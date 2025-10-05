#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import gymnasium as gym
from envs.kukaGymEnv import KukaGymEnv
import torch as th
from stable_baselines3 import DQN
from sb3_contrib import QRDQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

import datetime

def main(renders):
  """
  Train a deep q-learning model in Stable Baselines3
  """
  os.environ['OPENAI_LOGDIR'] = "./logs/dqn"

  # --------------------------------------------------------------
  # Place your code here! 
  # --------------------------------------------------------------  
  #policy_kwargs = dict(activation_fn=?, net_arch=[?,?,...], n_quantiles=50)

  env = KukaGymEnv(renders=renders, isDiscrete=True)
  #model = QRDQN("MlpPolicy", env,
  #                  learning_rate=?,
  #                  learning_starts=?,
  #                  batch_size=?,
  #                  gamma=?,
  #                  target_update_interval=?,
  #                  exploration_final_eps=?,
  #                  policy_kwargs=policy_kwargs, 
  #                  seed=0,
  #                  verbose=1)
  model.learn(total_timesteps=100000, log_interval=4)  
  # --------------------------------------------------------------
  model.save("kuka_model_dqn")
  print("Saving model to kuka_model_dqn")
  
  # Evaluate the policy
  mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10, return_episode_rewards=True)
  print(mean_reward)
  print(std_reward)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--norender', '-nr', action='store_true')
    args = parser.parse_args() 
    
    main(not(args.norender))
