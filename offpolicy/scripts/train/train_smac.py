import sys
import os
import numpy as np
from pathlib import Path
import wandb
import socket
import setproctitle
import torch
from offpolicy.config import get_config #parser(hyper-parameter)를 정의 및 저장
from offpolicy.utils.util import get_cent_act_dim, get_dim_from_space # 각 dim을 return
from offpolicy.envs.starcraft2.StarCraft2_Env import StarCraft2Env #starcraft2 env를 가져옴
from offpolicy.envs.starcraft2.smac_maps import get_map_params #맵 정보를 가진 dictionary에서 해당하는 map을 return함
from offpolicy.envs.env_wrappers import ShareDummyVecEnv, ShareSubprocVecEnv #2개의 정체 불명의 class를 가져옴


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "StarCraft2":
                env = StarCraft2Env(all_args) #env setting에 필요한 모든 argumen를 정의
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        '''
        init_env()가 아닌 init_env는 closure(함수가 정의된 시점에서의 환경(context)를 기억하고 이 환경에 있는 변수에 접근 가능한 함수)로써 rank/all_args값을 기억하고 있으며 이를 사용하여 함수를 객체로 다루면, 해당 함수를 다른 함수의 인자로 전달하거나 반환값으로 사용할 수 있다. 
        '''
        return init_env
    """
    rollout은 에이전트가 현재 상태에서 행동을 선택하고, 환경과 상호작용하여 얻은 다음 상태 및 보상을 기반으로 학습을 진>행하는 과정이다. rollout_threads 하이퍼파라미터는 동시에 실행되는 롤아웃 스레드의 수를 지정하는 역할을 하는 것으로 추정>된다.
    """
    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)]) #get_env_fn에서 return한 env를 ShareDummyVecEnv에 입력
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "StarCraft2":
                env = StarCraft2Env(all_args)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_eniv
    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)]) #get_env_fn에서 return한 env를 ShareDummyVecEnv에 입력
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)]) #list comprehension


def parse_args(args, parser):
    parser.add_argument('--map_name', type=str, default='3m',
                        help="Which smac map to run on")
    parser.add_argument('--use_available_actions', action='store_false',
                        default=True, help="Whether to use available actions")
    parser.add_argument('--use_same_share_obs', action='store_false',
                        default=True, help="Whether to use available actions")
    parser.add_argument('--use_global_all_local_state', action='store_true',
                        default=False, help="Whether to use available actions")
    
    '''
    parse_known_args 함수는 args에 입력된 변수값들 중에 이미 parser에 존재하는 변수들의 값은 
    첫번째 튜플에 저장되며 저장되어 있지 않은 변수의 값은 두 번째 튜플에 저장된다. 
    (Namespace(이미 있는 변수들의 튜플),[저장되지 않은 변수값들의 리스트])
    따라서 all_args에는 이미 parser에 변수가 저장된 변수값들만 존재한다. 
    '''
    all_args = parser.parse_known_args(args)[0]
    '''
    all_args = Namespace(algorithm_name='vdn', experiment_name='check', seed=1, cuda=True,~~~~~~)
    '''
    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    # cuda and # threads
    if all_args.cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        '''
        하이퍼스레딩: 원래는 한개의 코어는 하나의 쓰레드로써 한 가지의 일만 할 수 있었지만 하이퍼스레딩 기술을 적용해 한 개의 코어를 2개의 쓰레드로 나눔으로써 두 가지의 작업을 동시에 할 수 있게 만든 기술이다. 4코어 8쓰레드는 4개의 코어를 각각 2개로 분할해 8개의 쓰레드를 돌린다는 말이다. 
        '''
        if all_args.cuda_deterministic:
            '''
            CuDNN은 딥러닝에 특화된 CUDA 라이브러리로 이 라이브러리의 Randomness를 제어하기 위한 설정법이 아래 두 줄이다. 
            '''
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # setup file to output tensorboard, hyperparameters, and saved models
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                   0] + "/results") / all_args.env_name / all_args.map_name / all_args.algorithm_name / all_args.experiment_name
    '''
    run_dir: /home/uosai/바탕화면/off-policy/offpolicy/scripts/results/StarCraft2/3m/vdn/check
    '''
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if all_args.use_wandb:
        # init wandb
        run = wandb.init(project=all_args.env_name,
                         entity='singfor7012',
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                         str(all_args.experiment_name) +
                         "_seed" + str(all_args.seed),
                         group=all_args.map_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[
                                 1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))
    #setproctitle 라이브러리의 setproctitle 함수를 사용하여 프로세스의 제목을 설정하는 역할
    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + str(
        all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    # set seeds
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    env = make_train_env(all_args)
    buffer_length = get_map_params(all_args.map_name)["limit"] #limit: 60
    num_agents = get_map_params(all_args.map_name)["n_agents"] # n_agents:3

    # create policies and mapping fn
    if all_args.share_policy:
        print(env.share_observation_space[0])
        policy_info = {
            'policy_0': {"cent_obs_dim": get_dim_from_space(env.share_observation_space[0]),
                         "cent_act_dim": get_cent_act_dim(env.action_space),
                         "obs_space": env.observation_space[0],
                         "share_obs_space": env.share_observation_space[0],
                         "act_space": env.action_space[0]}
        }

        def policy_mapping_fn(id): return 'policy_0'
    else:
        policy_info = {
            'policy_' + str(agent_id): {"cent_obs_dim": get_dim_from_space(env.share_observation_space[agent_id]),
                                        "cent_act_dim": get_cent_act_dim(env.action_space),
                                        "obs_space": env.observation_space[agent_id],
                                        "share_obs_space": env.share_observation_space[agent_id],
                                        "act_space": env.action_space[agent_id]}
            for agent_id in range(num_agents)
        }

        def policy_mapping_fn(agent_id): return 'policy_' + str(agent_id)

    # choose algo
    if all_args.algorithm_name in ["rmatd3", "rmaddpg", "rmasac", "qmix", "vdn"]:
        from offpolicy.runner.rnn.smac_runner import SMACRunner as Runner
        assert all_args.n_rollout_threads == 1, ("only support 1 env in recurrent version.")
        eval_env = make_train_env(all_args)
    elif all_args.algorithm_name in ["matd3", "maddpg", "masac", "mqmix", "mvdn"]:
        from offpolicy.runner.mlp.smac_runner import SMACRunner as Runner
        eval_env = make_eval_env(all_args)
    else:
        raise NotImplementedError

    config = {"args": all_args,
              "policy_info": policy_info,
              "policy_mapping_fn": policy_mapping_fn,
              "env": env,
              "eval_env": eval_env,
              "num_agents": num_agents,
              "device": device,
              "run_dir": run_dir,
              "buffer_length": buffer_length,
              "use_same_share_obs": all_args.use_same_share_obs,
              "use_available_actions": all_args.use_available_actions}

    total_num_steps = 0
    runner = Runner(config=config)
    while total_num_steps < all_args.num_env_steps:
        total_num_steps = runner.run()


    env.close()
    if all_args.use_eval and (eval_env is not env):
        eval_env.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    '''
    sys.argv[1:]: 파이썬 명령행의 인자값을 받는 변수로 sys.argv는 아래와 같이 구성된다. 
    ['train_smac.py', '--algorithm_name', 'vdn', '--use_wandb', 'False']
    '''
    main(sys.argv[1:]) 
