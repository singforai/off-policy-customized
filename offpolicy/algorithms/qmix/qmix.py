import torch
import copy
from offpolicy.utils.util import soft_update, huber_loss, mse_loss, to_torch
from offpolicy.algorithms.qmix.algorithm.q_mixer import QMixer
from offpolicy.algorithms.vdn.algorithm.vdn_mixer import VDNMixer
from offpolicy.algorithms.base.trainer import Trainer
from offpolicy.utils.popart import PopArt
import numpy as np

import sys

class QMix(Trainer):
    def __init__(self, args, num_agents, batch_size, policies, policy_mapping_fn, device=torch.device("cuda:0"), episode_length=None, vdn=False):
        """
        Trainer class for recurrent QMix/VDN. See parent class for more information.
        :param episode_length: (int) maximum length of an episode.
        :param vdnl: (bool) whether the algorithm being used is VDN.
        """
        self.args = args
        self.use_popart = self.args.use_popart #False
        self.use_value_active_masks = self.args.use_value_active_masks #False
        self.use_per = self.args.use_per
        self.per_eps = self.args.per_eps
        self.use_huber_loss = self.args.use_huber_loss
        self.huber_delta = self.args.huber_delta

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device) # 2개의 key-value 쌍을 가지는 딕셔너리 생성
        self.lr = self.args.lr  #5e-4
        self.tau = self.args.tau #0.005
        self.opti_eps = self.args.opti_eps #RMSprop optimizer epsilon default: 1e-5
        self.weight_decay = self.args.weight_decay #default=0

        if episode_length is None:
            self.episode_length = self.args.episode_length
        else:
            self.episode_length = episode_length #80
        self.num_agents = num_agents
        self.policies = policies
        self.policy_mapping_fn = policy_mapping_fn 
        self.policy_ids = sorted(list(self.policies.keys())) #[policy_0] 
        self.policy_agents = {policy_id: sorted( #{'policy_0': [0, 1, 2]}
            [agent_id for agent_id in range(self.num_agents) if self.policy_mapping_fn(agent_id) == policy_id])
            for policy_id in self.policies.keys()}
        if self.use_popart: # 손실함수의 정규화를 수행 
            self.value_normalizer = {policy_id: PopArt(1) for policy_id in self.policies.keys()}

        self.use_same_share_obs = self.args.use_same_share_obs #True

        multidiscrete_list = None
        if any([isinstance(policy.act_dim, np.ndarray) for policy in self.policies.values()]):
            # multidiscrete
            multidiscrete_list = [len(self.policies[p_id].act_dim) *
                                  len(self.policy_agents[p_id]) for p_id in self.policy_ids]

        # mixer network
        if vdn:
            self.mixer = VDNMixer(args, self.num_agents, self.policies['policy_0'].central_obs_dim, self.device, multidiscrete_list=multidiscrete_list)
        else:
            self.mixer = QMixer(
                args, self.num_agents, self.policies['policy_0'].central_obs_dim, self.device, multidiscrete_list=multidiscrete_list)

        # target policies/networks
        self.target_policies = {p_id: copy.deepcopy(self.policies[p_id]) for p_id in self.policy_ids}
        self.target_mixer = copy.deepcopy(self.mixer) # 완전히 별개의 새로운 변수에 self.mixer와 완전히 같은 변수값을 가지게 복사 

        # collect all trainable parameters: each policy parameters, and the mixer parameters
        self.parameters = []
        for policy in self.policies.values():
            self.parameters += policy.parameters()
        self.parameters += self.mixer.parameters()
        self.optimizer = torch.optim.Adam(
            params=self.parameters, lr=self.lr, eps=self.opti_eps)
        if self.args.use_double_q:
            print("double Q learning will be used")

        self.batch_size = batch_size

    def train_policy_on_batch(self, batch, update_policy_id=None):
        """See parent class."""
        # unpack the batch
        obs_batch, cent_obs_batch, \
        act_batch, rew_batch, \
        dones_batch, dones_env_batch, \
        avail_act_batch, \
        importance_weights, idxes = batch

        if self.use_same_share_obs:   # True
            cent_obs_batch = to_torch(cent_obs_batch[self.policy_ids[0]]) #61 x batch_size x 48
        else:
            choose_agent_id = 0
            cent_obs_batch = to_torch(cent_obs_batch[self.policy_ids[0]][choose_agent_id]) #61

        dones_env_batch = to_torch(dones_env_batch[self.policy_ids[0]]).to(**self.tpdv) # 60 x batch_size x 1
        
        # individual agent q value sequences: each element is of shape (ep_len, batch_size, 1)
        agent_q_seq = []
        # individual agent next step q value sequences
        agent_nq_seq = []
        batch_size = None

        for p_id in self.policy_ids: #policy_0 요소 하나밖에 없음
            policy = self.policies[p_id]
            target_policy = self.target_policies[p_id]
            
            # get data related to the policy id
            pol_obs_batch = to_torch(obs_batch[p_id]) #3x61x32x64
            curr_act_batch = to_torch(act_batch[p_id]).to(**self.tpdv) #**은 들어가는 인자의 개수를 한정하고 싶지 않을 때 

            # stack over policy's agents to process them at once
            stacked_act_batch = torch.cat(list(curr_act_batch), dim=-2) #-2차원을 중심으로 concat
            stacked_obs_batch = torch.cat(list(pol_obs_batch), dim=-2) # 3x61x32x64 => 61x96x64 가장 겉에 있는 3개의 텐서를 -2 기준으로 결합
           
            if avail_act_batch[p_id] is not None:
                curr_avail_act_batch = to_torch(avail_act_batch[p_id])
                stacked_avail_act_batch = torch.cat(list(curr_avail_act_batch), dim=-2)
            else:
                stacked_avail_act_batch = None

            # [num_agents, episode_length, episodes, dim]
            batch_size = pol_obs_batch.shape[2]
            
            total_batch_size = batch_size * len(self.policy_agents[p_id]) #96: 32x3

            sum_act_dim = int(sum(policy.act_dim)) if policy.multidiscrete else policy.act_dim #9
            pol_prev_act_buffer_seq = torch.cat((torch.zeros(1, total_batch_size, sum_act_dim).to(**self.tpdv), #61x96x9
                                                 stacked_act_batch)) #torch.cat의 default: 

            # sequence of q values for all possible actions
            pol_all_q_seq, _ = policy.get_q_values(stacked_obs_batch, pol_prev_act_buffer_seq, #[61, 96, 9]
                                                                            policy.init_hidden(-1, total_batch_size))
            # get only the q values corresponding to actions taken in action_batch. Ignore the last time dimension.
            if policy.multidiscrete:
                pol_all_q_curr_seq = [q_seq[:-1] for q_seq in pol_all_q_seq]
                pol_q_seq = policy.q_values_from_actions(pol_all_q_curr_seq, stacked_act_batch)
            else:
                """
                [:-1]은 첫번째 차원의 마지막 요소를 제외한 요소만을 선택하겠다는 것이다. 
                """ 
                pol_q_seq = policy.q_values_from_actions(pol_all_q_seq[:-1], stacked_act_batch) # 60x96x9
            #############################################################################
            agent_q_out_sequence = pol_q_seq.split(split_size=batch_size, dim=-2)
            agent_q_seq.append(torch.cat(agent_q_out_sequence, dim=-1))

            with torch.no_grad():
                if self.args.use_double_q:
                    # choose greedy actions from live, but get corresponding q values from target
                    greedy_actions, _ = policy.actions_from_q(pol_all_q_seq, available_actions=stacked_avail_act_batch)
                    target_q_seq, _ = target_policy.get_q_values(stacked_obs_batch, pol_prev_act_buffer_seq, target_policy.init_hidden(-1, total_batch_size), action_batch=greedy_actions)
                else:
                    _, _, target_q_seq = target_policy.get_actions(stacked_obs_batch, pol_prev_act_buffer_seq, target_policy.init_hidden(-1, total_batch_size))
            # don't need the first Q values for next step
            target_q_seq = target_q_seq[1:]
            agent_nq_sequence = target_q_seq.split(split_size=batch_size, dim=-2)
            agent_nq_seq.append(torch.cat(agent_nq_sequence, dim=-1))

        # combine agent q value sequences to feed into mixer networks
        agent_q_seq = torch.cat(agent_q_seq, dim=-1)
        agent_nq_seq = torch.cat(agent_nq_seq, dim=-1)

        # get curr step and next step Q_tot values using mixer
        Q_tot_seq = torch.reshape(self.mixer(agent_q_seq, cent_obs_batch[:-1]),(-1,self.batch_size,1))#.squeeze(-1)
        next_step_Q_tot_seq = torch.reshape(self.target_mixer(agent_nq_seq, cent_obs_batch[1:]),(-1,self.batch_size,1))#.squeeze(-1)
                
        # agents share reward
        rewards = to_torch(rew_batch[self.policy_ids[0]][0]).to(**self.tpdv)
        # form bad transition mask
        bad_transitions_mask = torch.cat((torch.zeros(1, batch_size, 1).to(**self.tpdv), dones_env_batch[:self.episode_length - 1, :, :]))

        # bootstrapped targets
        Q_tot_target_seq = rewards + (1 - dones_env_batch) * self.args.gamma * next_step_Q_tot_seq
        # Bellman error and mask out invalid transitions
        error = (Q_tot_seq - Q_tot_target_seq.detach()) * (1 - bad_transitions_mask)

        if self.use_per:
            # Form updated priorities for prioritized experience replay using the Bellman error
            importance_weights = to_torch(importance_weights).to(**self.tpdv)
            if self.use_huber_loss:
                per_batch_error = huber_loss(error, self.huber_delta).sum(dim=0).flatten()
            else:
                per_batch_error = mse_loss(error).sum(dim=0).flatten()
            importance_weight_error = per_batch_error * importance_weights
            loss = importance_weight_error.sum() / (1 - bad_transitions_mask).sum()

            # new priorities are a combination of the maximum TD error across sequence and the mean TD error across sequence (see R2D2 paper)
            td_errors = error.abs().cpu().detach().numpy()
            new_priorities = ((1 - self.args.per_nu) * td_errors.mean(axis=0) +
                              self.args.per_nu * td_errors.max(axis=0)).flatten() + self.per_eps
        else:
            if self.use_huber_loss:
                loss = huber_loss(error, self.huber_delta).sum() / (1 - bad_transitions_mask).sum()
            else:
                loss = mse_loss(error).sum() / (1 - bad_transitions_mask).sum()
            new_priorities = None

        # backward pass and gradient step
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters, self.args.max_grad_norm)
        self.optimizer.step()
        # log
        train_info = {}
        train_info['loss'] = loss
        train_info['grad_norm'] = grad_norm
        train_info['Q_tot'] = (Q_tot_seq * (1 - bad_transitions_mask)).mean()

        return train_info, new_priorities, idxes


    def hard_target_updates(self):
        """Hard update the target networks."""
        print("hard update targets")
        for policy_id in self.policy_ids:
            self.target_policies[policy_id].load_state(self.policies[policy_id])
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def soft_target_updates(self):
        """Soft update the target networks."""
        for policy_id in self.policy_ids:
            soft_update(self.target_policies[policy_id], self.policies[policy_id], self.tau)
        if self.mixer is not None:
            soft_update(self.target_mixer, self.mixer, self.tau)

    def prep_training(self):
        """See parent class."""
        for p_id in self.policy_ids:
            self.policies[p_id].q_network.train()
            self.target_policies[p_id].q_network.train()
        self.mixer.train()
        self.target_mixer.train()

    def prep_rollout(self):
        """See parent class."""
        for p_id in self.policy_ids:
            self.policies[p_id].q_network.eval()
            self.target_policies[p_id].q_network.eval()
        self.mixer.eval()
        self.target_mixer.eval()
