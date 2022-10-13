# -*- coding: utf-8 -*-
# MIT License
#
# Copyright (c) 2017 Kai Arulkumaran
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# ==============================================================================
from __future__ import division
import argparse
import bz2
from datetime import datetime
import os
import pickle

import atari_py
import numpy as np
import torch
from tqdm import trange
import random
import json
from agent import Agent
from env import Env
from memory import ReplayMemory
from test import test
# import setproctitle

# Note that hyperparameters may originally be reported in ATARI game frames instead of agent steps
def get_parser():
    parser = argparse.ArgumentParser(description='Rainbow')
    # parser.add_argument('--id', type=str, default='default', help='Experiment ID')
    parser.add_argument('--work-dir', type=str, default='./test')
    parser.add_argument('--seed', type=int, default=-1, help='Random seed')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--game', type=str, default='ms_pacman', choices=atari_py.list_games(), help='ATARI game')
    parser.add_argument('--T-max', type=int, default=int(1e5), metavar='STEPS', help='Number of training steps (4x number of frames)')
    parser.add_argument('--max-episode-length', type=int, default=int(108e3), metavar='LENGTH', help='Max episode length in game frames (0 to disable)')
    parser.add_argument('--history-length', type=int, default=4, metavar='T', help='Number of consecutive states processed')
    parser.add_argument('--architecture', type=str, default='data-efficient', choices=['canonical', 'data-efficient'], metavar='ARCH', help='Network architecture')
    parser.add_argument('--hidden-size', type=int, default=256, metavar='SIZE', help='Network hidden size')
    parser.add_argument('--noisy-std', type=float, default=0.1, metavar='σ', help='Initial standard deviation of noisy linear layers')
    parser.add_argument('--atoms', type=int, default=51, metavar='C', help='Discretised size of value distribution')
    parser.add_argument('--V-min', type=float, default=-10, metavar='V', help='Minimum of value distribution support')
    parser.add_argument('--V-max', type=float, default=10, metavar='V', help='Maximum of value distribution support')
    parser.add_argument('--model', type=str, metavar='PARAMS', help='Pretrained model (state dict)')
    parser.add_argument('--memory-capacity', type=int, default=int(1e5), metavar='CAPACITY', help='Experience replay memory capacity')
    parser.add_argument('--replay-frequency', type=int, default=1, metavar='k', help='Frequency of sampling from memory')
    parser.add_argument('--priority-exponent', type=float, default=0.5, metavar='ω', help='Prioritised experience replay exponent (originally denoted α)')
    parser.add_argument('--priority-weight', type=float, default=0.4, metavar='β', help='Initial prioritised experience replay importance sampling weight')
    parser.add_argument('--multi-step', type=int, default=20, metavar='n', help='Number of steps for multi-step return')
    parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
    parser.add_argument('--target-update', type=int, default=int(2e3), metavar='τ', help='Number of steps after which to update target network')
    parser.add_argument('--reward-clip', type=int, default=1, metavar='VALUE', help='Reward clipping (0 to disable)')
    parser.add_argument('--learning-rate', type=float, default=0.0001, metavar='η', help='Learning rate')
    parser.add_argument('--adam-eps', type=float, default=1.5e-4, metavar='ε', help='Adam epsilon')
    parser.add_argument('--batch-size', type=int, default=32, metavar='SIZE', help='Batch size')
    parser.add_argument('--norm-clip', type=float, default=10, metavar='NORM', help='Max L2 norm for gradient clipping')
    parser.add_argument('--learn-start', type=int, default=int(1600), metavar='STEPS', help='Number of steps before starting training')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
    parser.add_argument('--evaluation-interval', type=int, default=10000, metavar='STEPS', help='Number of training steps between evaluations')
    parser.add_argument('--evaluation-episodes', type=int, default=10, metavar='N', help='Number of evaluation episodes to average over')
    # TODO: Note that DeepMind's evaluation method is running the latest agent for 500K frames ever every 1M steps
    parser.add_argument('--evaluation-size', type=int, default=500, metavar='N', help='Number of transitions to use for validating Q')
    parser.add_argument('--render', action='store_true', help='Display screen (testing only)')
    parser.add_argument('--enable-cudnn', action='store_true', help='Enable cuDNN (faster but nondeterministic)')
    parser.add_argument('--checkpoint-interval', default=0, help='How often to checkpoint the model, defaults to 0 (never checkpoint)')
    parser.add_argument('--memory', help='Path to save/load the memory from')
    parser.add_argument('--disable-bzip-memory', action='store_true', help='Don\'t zip the memory file. Not recommended (zipping is a bit slower and much, much smaller)')
    return parser.parse_args()

def main(weights, tid=0, seed=None, task_str=None):
    # Setup
    args = get_parser()

    if seed is not None:
        args.seed = seed

    def set_seed_everywhere(seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic=True

    if args.seed < 0:
        args.seed = np.random.randint(0, 10000)
    set_seed_everywhere(args.seed)

    # xid = 'curl-' + args.game + '-' + str(seed)
    process_title = 'ELo-Rainbow'
    env_name = args.game

    weights_str = '_'.join([f"{w:.2f}" for w in weights])

    xid = f'{process_title}-{args.game}-t{tid}-w{weights_str}-b{args.batch_size}-s{args.seed}'

    if tid is not None:
        xid += f''

    args.id = xid
    # setproctitle.setproctitle(f"python {process_title}")

    print(' ' * 26 + 'Options')
    for k, v in vars(args).items():
      print(' ' * 26 + k + ': ' + str(v))
    results_dir = os.path.join(args.work_dir, args.id)
    if not os.path.exists(results_dir):
      os.makedirs(results_dir)
    metrics = {'steps': [], 'rewards': [], 'Qs': [], 'best_avg_reward': -float('inf')}

    # Simple ISO 8601 timestamped logger
    def log(s):
      print('[' + str(datetime.now().strftime('%Y-%m-%dT%H:%M:%S')) + '] ' + s)


    def load_memory(memory_path, disable_bzip):
      if disable_bzip:
        with open(memory_path, 'rb') as pickle_file:
          return pickle.load(pickle_file)
      else:
        with bz2.open(memory_path, 'rb') as zipped_pickle_file:
          return pickle.load(zipped_pickle_file)


    def save_memory(memory, memory_path, disable_bzip):
      if disable_bzip:
        with open(memory_path, 'wb') as pickle_file:
          pickle.dump(memory, pickle_file)
      else:
        with bz2.open(memory_path, 'wb') as zipped_pickle_file:
          pickle.dump(memory, zipped_pickle_file)

    # save config
    with open(os.path.join(results_dir, 'args.json'), 'w') as f:
      arg_json = vars(args)
      json.dump(arg_json, f, sort_keys=True, indent=4)

    if task_str is not None:
        with open(os.path.join(results_dir, 'task.json'), 'w') as f:
            f.write(task_str)

    if torch.cuda.is_available() and not args.disable_cuda:
      args.device = torch.device('cuda')
      # torch.cuda.manual_seed(np.random.randint(1, 10000))
      # torch.backends.cudnn.enabled = args.enable_cudnn
    else:
      args.device = torch.device('cpu')


    # Environment
    env = Env(args)
    env.train()
    action_space = env.action_space()

    # Agent
    dqn = Agent(args, env, weights)

    # If a model is provided, and evaluate is fale, presumably we want to resume, so try to load memory
    if args.model is not None and not args.evaluate:
      if not args.memory:
        raise ValueError('Cannot resume training without memory save path. Aborting...')
      elif not os.path.exists(args.memory):
        raise ValueError('Could not find memory file at {path}. Aborting...'.format(path=args.memory))

      mem = load_memory(args.memory, args.disable_bzip_memory)

    else:
      mem = ReplayMemory(args, args.memory_capacity)

    priority_weight_increase = (1 - args.priority_weight) / (args.T_max - args.learn_start)


    # Construct validation memory
    val_mem = ReplayMemory(args, args.evaluation_size)
    T, done = 0, True
    while T < args.evaluation_size:
      if done:
        state, done = env.reset(), False

      next_state, _, done = env.step(np.random.randint(0, action_space))
      val_mem.append(state, None, None, done)
      state = next_state
      T += 1

    if args.evaluate:
      dqn.eval()  # Set DQN (online network) to evaluation mode
      avg_reward, avg_Q = test(args, 0, dqn, val_mem, metrics, results_dir, evaluate=True)  # Test
      print('Avg. reward: ' + str(avg_reward) + ' | Avg. Q: ' + str(avg_Q))
    else:
      # Training loop
      dqn.train()
      T, done = 0, True
      for T in trange(1, args.T_max + 1):
        if done:
          state, done = env.reset(), False

        if T % args.replay_frequency == 0:
          dqn.reset_noise()  # Draw a new set of noisy weights

        action = dqn.act(state)  # Choose an action greedily (with noisy weights)
        next_state, reward, done = env.step(action)  # Step
        if args.reward_clip > 0:
          reward = max(min(reward, args.reward_clip), -args.reward_clip)  # Clip rewards
        mem.append(state, action, reward, done)  # Append transition to memory

        # Train and test
        if T >= args.learn_start:
          mem.priority_weight = min(mem.priority_weight + priority_weight_increase, 1)  # Anneal importance sampling weight β to 1

          if T % args.replay_frequency == 0:
            #for _ in range(4):
            dqn.learn(mem)  # Train with n-step distributional double-Q learning
            dqn.update_momentum_net() # MoCo momentum upate
          """
          if T % args.evaluation_interval == 0:
            dqn.eval()  # Set DQN (online network) to evaluation mode
            avg_reward, avg_Q = test(args, T, dqn, val_mem, metrics, results_dir)  # Test
            log('T = ' + str(T) + ' / ' + str(args.T_max) + ' | Avg. reward: ' + str(avg_reward) + ' | Avg. Q: ' + str(avg_Q))
            dqn.train()  # Set DQN (online network) back to training mode

            # If memory path provided, save it
            if args.memory is not None:
              save_memory(mem, args.memory, args.disable_bzip_memory)
          """
          # Update target network
          if T % args.target_update == 0:
            dqn.update_target_net()

          # Checkpoint the network
          if (args.checkpoint_interval != 0) and (T % args.checkpoint_interval == 0):
            dqn.save(results_dir, 'checkpoint.pth')

        state = next_state

    dqn.eval()  # Set DQN (online network) to evaluation mode
    avg_reward = test(args, T, dqn, val_mem, metrics, results_dir)  # Test
    # log('T = ' + str(T) + ' / ' + str(args.T_max) + ' | Avg. reward: ' + str(avg_reward) + ' | Avg. Q: ' + str(avg_Q))
    env.close()
    return avg_reward

if __name__ == '__main__':
  main([1.0] * 5)

