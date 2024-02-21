import os
import numpy as np
import wandb
import torch
from tensorboardX import SummaryWriter

import time
import datetime

from offpolicy.utils.rec_buffer import RecReplayBuffer, PrioritizedRecReplayBuffer
from offpolicy.utils.util import DecayThenFlatSchedule

import sys

class RecRunner(object):
    """Base class for training recurrent policies."""

    def __init__(self, config):
        """
        Base class for training recurrent policies.
        :param config: (dict) Config dictionary containing parameters for training.
        """
        self.args = config["args"]
        self.device = config["device"]
        self.q_learning = ["qmix","vdn"]

        self.map_name = self.args.map_name
        self.share_policy = self.args.share_policy
        self.algorithm_name = self.args.algorithm_name
        self.env_name = self.args.env_name
        self.num_env_steps = self.args.num_env_steps
        self.use_wandb = self.args.use_wandb
        self.use_reward_normalization = self.args.use_reward_normalization
        self.use_popart = self.args.use_popart
        self.use_per = self.args.use_per #false
        self.per_alpha = self.args.per_alpha
        self.per_beta_start = self.args.per_beta_start
        self.buffer_size = self.args.buffer_size
        self.batch_size = self.args.batch_size
        self.hidden_size = self.args.hidden_size
        self.use_soft_update = self.args.use_soft_update
        self.hard_update_interval_episode = self.args.hard_update_interval_episode
        self.popart_update_interval_step = self.args.popart_update_interval_step
        self.actor_train_interval_step = self.args.actor_train_interval_step
        self.train_interval_episode = self.args.train_interval_episode #1
        self.train_interval = self.args.train_interval
        self.use_eval = self.args.use_eval
        self.eval_interval = self.args.eval_interval
        self.save_interval = self.args.save_interval
        self.log_interval = self.args.log_interval

        self.total_env_steps = 0  # total environment interactions collected during training
        self.num_episodes_collected = 0  # total episodes collected during training
        self.total_train_steps = 0  # number of gradient updates performed
        self.last_train_episode = 0  # last episode after which a gradient update was performed
        self.last_eval_T = 0  # last episode after which a eval run was conducted
        self.last_save_T = 0  # last epsiode after which the models were saved
        self.last_log_T = 0 # last timestep after which information was logged
        self.last_hard_update_episode = 0 # last episode after which target policy was updated to equal live policy

        self.ExecuteTime = ExpectedExecuteTime()

        if config.__contains__("take_turn"):
            self.take_turn = config["take_turn"]
        else:
            self.take_turn = False

        if config.__contains__("use_same_share_obs"):
            self.use_same_share_obs = config["use_same_share_obs"]
        else:
            self.use_same_share_obs = False

        if config.__contains__("use_available_actions"):
            self.use_avail_acts = config["use_available_actions"]
        else:
            self.use_avail_acts = False

        if config.__contains__("buffer_length"):
            self.episode_length = config["buffer_length"]
            if self.args.use_naive_recurrent_policy:
                self.data_chunk_length = config["buffer_length"]
            else:
                self.data_chunk_length = self.args.data_chunk_length
        else:
            self.episode_length = self.args.episode_length
            if self.args.use_naive_recurrent_policy:
                self.data_chunk_length = self.args.episode_length
            else:
                self.data_chunk_length = self.args.data_chunk_length

        self.policy_info = config["policy_info"]
        self.policy_ids = sorted(list(self.policy_info.keys()))
        self.policy_mapping_fn = config["policy_mapping_fn"]

        self.num_agents = config["num_agents"]
        self.agent_ids = [i for i in range(self.num_agents)]

        self.env = config["env"]
        self.eval_env = config["eval_env"]
        # no parallel envs
        self.num_envs = 1

        # dir
        self.model_dir = self.args.model_dir
        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
        else:
            self.run_dir = config["run_dir"]
            self.log_dir = str(self.run_dir / 'logs')
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writter = SummaryWriter(self.log_dir)
            self.save_dir = str(self.run_dir / 'models')
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        # initialize all the policies and organize the agents corresponding to each policy
        if self.algorithm_name == "rmatd3":
            from offpolicy.algorithms.r_matd3.algorithm.rMATD3Policy import R_MATD3Policy as Policy
            from offpolicy.algorithms.r_matd3.r_matd3 import R_MATD3 as TrainAlgo
        elif self.algorithm_name == "rmaddpg":
            assert self.actor_train_interval_step == 1, (
                "rmaddpg only supports actor_train_interval_step=1.")
            from offpolicy.algorithms.r_maddpg.algorithm.rMADDPGPolicy import R_MADDPGPolicy as Policy
            from offpolicy.algorithms.r_maddpg.r_maddpg import R_MADDPG as TrainAlgo
        elif self.algorithm_name == "rmasac":
            assert self.actor_train_interval_step == 1, (
                "rmasac only support actor_train_interval_step=1.")
            from offpolicy.algorithms.r_masac.algorithm.rMASACPolicy import R_MASACPolicy as Policy
            from offpolicy.algorithms.r_masac.r_masac import R_MASAC as TrainAlgo
        elif self.algorithm_name == "qmix":
            from offpolicy.algorithms.qmix.algorithm.QMixPolicy import QMixPolicy as Policy
            from offpolicy.algorithms.qmix.qmix import QMix as TrainAlgo
        elif self.algorithm_name == "vdn":
            from offpolicy.algorithms.vdn.algorithm.VDNPolicy import VDNPolicy as Policy
            from offpolicy.algorithms.vdn.vdn import VDN as TrainAlgo
        else:
            raise NotImplementedError

        self.collecter = self.collect_rollout
        self.saver = self.save_q if self.algorithm_name in self.q_learning else self.save        
        self.restorer = self.restore_q if self.algorithm_name in self.q_learning else self.restore
        self.train = self.batch_train_q if self.algorithm_name in self.q_learning else self.batch_train

        self.policies = {p_id: Policy(config, self.policy_info[p_id]) for p_id in self.policy_ids}
        
        if self.model_dir is not None:
            self.restorer()

        # initialize trainer class for updating policies
        self.trainer = TrainAlgo(self.args, self.num_agents,self.batch_size, self.policies, self.policy_mapping_fn,
                                 device=self.device, episode_length=self.episode_length)

        # map policy id to agent ids controlled by that policy
        self.policy_agents = {policy_id: sorted(
            [agent_id for agent_id in self.agent_ids if self.policy_mapping_fn(agent_id) == policy_id]) for policy_id in
            self.policies.keys()}

        self.policy_obs_dim = {
            policy_id: self.policies[policy_id].obs_dim for policy_id in self.policy_ids}
        self.policy_act_dim = {
            policy_id: self.policies[policy_id].act_dim for policy_id in self.policy_ids}
        self.policy_central_obs_dim = {
            policy_id: self.policies[policy_id].central_obs_dim for policy_id in self.policy_ids}

        num_train_episodes = (self.num_env_steps / self.episode_length) / (self.train_interval_episode)
        self.beta_anneal = DecayThenFlatSchedule(
            self.per_beta_start, 1.0, num_train_episodes, decay="linear")

        if self.use_per:
            self.buffer = PrioritizedRecReplayBuffer(self.per_alpha,
                                                     self.policy_info,
                                                     self.policy_agents,
                                                     self.buffer_size,
                                                     self.episode_length,
                                                     self.use_same_share_obs,
                                                     self.use_avail_acts,
                                                     self.use_reward_normalization)
        else:
            self.buffer = RecReplayBuffer(self.policy_info,
                                          self.policy_agents,
                                          self.buffer_size,
                                          self.episode_length,
                                          self.use_same_share_obs,
                                          self.use_avail_acts,
                                          self.use_reward_normalization)
    
    def run(self):
        self.ExecuteTime.start_time = time.perf_counter()
        self.ExecuteTime.prior_step = self.total_env_steps


        if self.args.eval_mode:
            print("================================== evaluation mode ===================================")
            self.eval()
            self.total_env_steps = self.num_env_steps+1
            print("======================================================================================")
            return self.total_env_steps

        """Collect a training episode and perform appropriate training, saving, logging, and evaluation steps."""
        # collect data
        self.trainer.prep_rollout()
        env_info = self.collecter(explore=True, training_episode=True, warmup=False) 
        for k, v in env_info.items():
            self.env_infos[k].append(v)

        # train
        if ((self.num_episodes_collected - self.last_train_episode) / self.train_interval_episode) >= 1 or self.last_train_episode == 0:
            self.train() #batch_train_q
            self.total_train_steps += 1
            self.last_train_episode = self.num_episodes_collected

        self.ExecuteTime.end_time = time.perf_counter()
        self.ExecuteTime.execution_time = self.ExecuteTime.end_time - self.ExecuteTime.start_time
        self.ExecuteTime.bias_correction = 1 - self.ExecuteTime.weight**(self.total_env_steps)
        self.ExecuteTime.weighted_average_time = (self.ExecuteTime.weight * (
            self.ExecuteTime.weighted_average_time)) + ((1 - self.ExecuteTime.weight) * (self.ExecuteTime.execution_time/(self.total_env_steps-self.ExecuteTime.prior_step)))/(self.ExecuteTime.bias_correction)
        self.ExecuteTime.sec_per_step = self.ExecuteTime.weighted_average_time*(self.num_env_steps-self.total_env_steps)
        self.ExecuteTime.predicted_time = str(datetime.timedelta(seconds = self.ExecuteTime.sec_per_step))
        

        # save
        if (self.total_env_steps - self.last_save_T) / self.save_interval >= 1 or self.total_env_steps >= self.num_env_steps:
            self.saver()
            self.last_save_T = self.total_env_steps


        # log
        if ((self.total_env_steps - self.last_log_T) / self.log_interval) >= 1 or self.total_env_steps >= self.num_env_steps:
            
            self.log()
            print(f"Predicted program execution time: {self.ExecuteTime.predicted_time}")
            print("================================================================================")
            self.last_log_T = self.total_env_steps

        # eval
        if self.use_eval and ((self.total_env_steps - self.last_eval_T) / self.eval_interval) >= 1 or self.total_env_steps >= self.num_env_steps:
            self.eval()
            self.last_eval_T = self.total_env_steps

        return self.total_env_steps
    
    def warmup(self, num_warmup_episodes):
        """
        Fill replay buffer with enough episodes to begin training.

        :param: num_warmup_episodes (int): number of warmup episodes to collect.
        """
        self.trainer.prep_rollout()
        warmup_rewards = []
        print("warm up...")
        for _ in range((num_warmup_episodes // self.num_envs) + 1):
            env_info = self.collecter(explore=True, training_episode=False, warmup=True)
            warmup_rewards.append(env_info['average_episode_rewards'])
        warmup_reward = np.mean(warmup_rewards)
        print("warmup average episode rewards: {}".format(warmup_reward))

    def batch_train(self):
        """Do a gradient update for all policies."""
        self.trainer.prep_training()

        # gradient updates
        self.train_infos = []
        update_actor = False
        for p_id in self.policy_ids:
            if self.use_per:
                beta = self.beta_anneal.eval(self.total_train_steps)
                sample = self.buffer.sample(self.batch_size, beta, p_id)
            else:
                sample = self.buffer.sample(self.batch_size)

            update_method = self.trainer.shared_train_policy_on_batch if self.use_same_share_obs else self.trainer.cent_train_policy_on_batch
            
            train_info, new_priorities, idxes = update_method(p_id, sample)
            update_actor = train_info['update_actor']

            if self.use_per:
                self.buffer.update_priorities(idxes, new_priorities, p_id)

            self.train_infos.append(train_info)

        if self.use_soft_update and update_actor:
            for pid in self.policy_ids:
                self.policies[pid].soft_target_updates()
        else:
            if ((self.num_episodes_collected - self.last_hard_update_episode) / self.hard_update_interval_episode) >= 1:
                for pid in self.policy_ids:
                    self.policies[pid].hard_target_updates()
                self.last_hard_update_episode = self.num_episodes_collected

    def batch_train_q(self):
        """Do a q-learning update to policy (used for QMix and VDN)."""
        self.trainer.prep_training()
        # gradient updates
        self.train_infos = []

        for p_id in self.policy_ids:
            if self.use_per:
                beta = self.beta_anneal.eval(self.total_train_steps)
                sample = self.buffer.sample(self.batch_size, beta, p_id)
            else:
               
                sample = self.buffer.sample(self.batch_size)
            train_info, new_priorities, idxes = self.trainer.train_policy_on_batch(sample) 

            if self.use_per:
                self.buffer.update_priorities(idxes, new_priorities, p_id)

            self.train_infos.append(train_info)

        if self.use_soft_update:
            self.trainer.soft_target_updates()
        else:
            if (self.num_episodes_collected - self.last_hard_update_episode) / self.hard_update_interval_episode >= 1:
                self.trainer.hard_target_updates()
                self.last_hard_update_episode = self.num_episodes_collected

    def save(self):
        """Save all policies to the path specified by the config."""
        for pid in self.policy_ids:
            policy_critic = self.policies[pid].critic
            critic_save_path = self.save_dir + '/' + str(pid)
            if not os.path.exists(critic_save_path):
                os.makedirs(critic_save_path)
            torch.save(policy_critic.state_dict(),
                       critic_save_path + '/critic.pt')

            policy_actor = self.policies[pid].actor
            actor_save_path = self.save_dir + '/' + str(pid)
            if not os.path.exists(actor_save_path):
                os.makedirs(actor_save_path)
            torch.save(policy_actor.state_dict(),
                       actor_save_path + '/actor.pt')

    def save_q(self):
        """Save all policies to the path specified by the config. Used for QMix and VDN."""
        for pid in self.policy_ids:
            policy_Q = self.policies[pid].q_network
            p_save_path = self.save_dir + '/' + str(pid)
            if not os.path.exists(p_save_path):
                os.makedirs(p_save_path)
            torch.save(policy_Q.state_dict(), p_save_path + f'/{self.map_name}_{self.total_env_steps}_q_network.pt')
        print(f'{self.map_name}_{self.total_env_steps}_q_network.pt file saves complete')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        torch.save(self.trainer.mixer.state_dict(),
                   self.save_dir + f'/{self.map_name}_{self.total_env_steps}_mixer.pt')
        print(f'{self.map_name}_{self.total_env_steps}_mixer.pt file saves complete' )

    def restore(self):
        """Load policies policies from pretrained models specified by path in config."""
        for pid in self.policy_ids:
            path = str(self.model_dir) + str(pid)
            print("load the pretrained model from {}".format(path))
            policy_critic_state_dict = torch.load(path + '/critic.pt')
            policy_actor_state_dict = torch.load(path + '/actor.pt')

            self.policies[pid].critic.load_state_dict(policy_critic_state_dict)
            self.policies[pid].actor.load_state_dict(policy_actor_state_dict)

    def restore_q(self):
        """Load policies policies from pretrained models specified by path in config. Used for QMix and VDN."""
        for pid in self.policy_ids:
            path = str(self.model_dir) + str(pid)
            print("load the pretrained model from {}".format(path))
            policy_q_state_dict = torch.load(path + '/' +self.args.q_network_name)  
            self.policies[pid].q_network.load_state_dict(policy_q_state_dict)
            
        policy_mixer_state_dict = torch.load(str(self.model_dir) + self.args.mixer_name)
        #self.trainer.mixer.load_state_dict(policy_mixer_state_dict) # QMIX가 아닌 VDN을 실행할 때는 MIXING network가 필요 없으므로 차단해도 된다. 

    def log(self):
        """Log relevent training and rollout colleciton information.."""
        raise NotImplementedError

    def log_clear(self):
        """Clear logging variables so they do not contain stale information."""
        raise NotImplementedError

    def log_env(self, env_info, suffix=None):
        """
        Log information related to the environment.
        :param env_info: (dict) contains logging information related to the environment.
        :param suffix: (str) optional string to add to end of keys in env_info when logging. 
        """
        for k, v in env_info.items():
            if len(v) > 0:
                v = np.mean(v)
                suffix_k = k if suffix is None else suffix + k 
                print(suffix_k + " is " + str(v))
                if self.use_wandb:
                    wandb.log({suffix_k: v}, step=self.total_env_steps)
                else:
                    self.writter.add_scalars(suffix_k, {suffix_k: v}, self.total_env_steps)

    def log_train(self, policy_id, train_info):
        """
        Log information related to training.
        :param policy_id: (str) policy id corresponding to the information contained in train_info.
        :param train_info: (dict) contains logging information related to training.
        """
        for k, v in train_info.items():
            policy_k = str(policy_id) + '/' + k
            if self.use_wandb:
                wandb.log({policy_k: v}, step=self.total_env_steps)
            else:
                self.writter.add_scalars(policy_k, {policy_k: v}, self.total_env_steps)

    def collect_rollout(self):
        """Collect a rollout and store it in the buffer."""
        raise NotImplementedError # 해당 메서드가 아직 구현되지 않았음을 의미하는 에러 

class ExpectedExecuteTime():
    def __init__(self):
        self.total_num_steps = 0
        self.weighted_average_time = 0.0
        self.weight = 0.99
        self.start_time = None
        self.prior_step = None
        self.end_time = None
        self.execution_time = 0
        self.bias_correction = None
        self.sec_per_step = None
        self.predicted_time = None

        