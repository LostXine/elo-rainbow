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
import os
import numpy as np
import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_
# import kornia.augmentation as aug
import torchvision.transforms.transforms as transforms
import torch
import torch.nn as nn
from model import DQN, AuxDQN
from torch.nn import functional as F

# random_shift = nn.Sequential(aug.RandomCrop((80, 80)), nn.ReplicationPad2d(4), aug.RandomCrop((84, 84)))
# aug = random_shift

augs = torch.nn.Sequential(
    transforms.RandomCrop(80),
    nn.ReplicationPad2d(4),
    transforms.RandomCrop(84),
)

aug = torch.jit.script(augs)

class Agent():
  def __init__(self, args, env, weights):
    self.args = args
    self.weights = weights
    self.action_space = env.action_space()
    self.atoms = args.atoms
    self.Vmin = args.V_min
    self.Vmax = args.V_max
    self.support = torch.linspace(args.V_min, args.V_max, self.atoms).to(device=args.device)  # Support (range) of z
    self.delta_z = (args.V_max - args.V_min) / (self.atoms - 1)
    self.batch_size = args.batch_size
    self.n = args.multi_step
    self.discount = args.discount
    self.norm_clip = args.norm_clip
    self.coeff = 0.01 if args.game in ['pong', 'boxing', 'private_eye', 'freeway'] else 1.

    self.online_net = AuxDQN(args, self.action_space).to(device=args.device)
    self.momentum_net = AuxDQN(args, self.action_space).to(device=args.device)
    if args.model:  # Load pretrained model if provided
      if os.path.isfile(args.model):
        state_dict = torch.load(args.model, map_location='cpu')  # Always load tensors onto CPU by default, will shift to GPU if necessary
        if 'conv1.weight' in state_dict.keys():
          for old_key, new_key in (('conv1.weight', 'convs.0.weight'), ('conv1.bias', 'convs.0.bias'), ('conv2.weight', 'convs.2.weight'), ('conv2.bias', 'convs.2.bias'), ('conv3.weight', 'convs.4.weight'), ('conv3.bias', 'convs.4.bias')):
            state_dict[new_key] = state_dict[old_key]  # Re-map state dict for old pretrained models
            del state_dict[old_key]  # Delete old keys for strict load_state_dict
        self.online_net.load_state_dict(state_dict)
        print("Loading pretrained model: " + args.model)
      else:  # Raise error if incorrect model path provided
        raise FileNotFoundError(args.model)

    self.online_net.train()
    self.initialize_momentum_net()
    self.momentum_net.train()

    self.target_net = AuxDQN(args, self.action_space).to(device=args.device)
    self.update_target_net()
    self.target_net.train()
    for param in self.target_net.parameters():
      param.requires_grad = False


    for param in self.momentum_net.parameters():
      param.requires_grad = False
    self.optimiser = optim.Adam(self.online_net.parameters(), lr=args.learning_rate, eps=args.adam_eps)
    
    self.mse_loss = nn.MSELoss()
    self.ce_loss = nn.CrossEntropyLoss()

    
  # Resets noisy weights in all linear layers (of online net only)
  def reset_noise(self):
    self.online_net.reset_noise()

  # Acts based on single state (no batch)
  def act(self, state):
    with torch.no_grad():
      a = self.online_net(state.unsqueeze(0))
      return (a * self.support).sum(2).argmax(1).item()

  # Acts with an ε-greedy policy (used for evaluation only)
  def act_e_greedy(self, state, epsilon=0.001):  # High ε can reduce evaluation scores drastically
    return np.random.randint(0, self.action_space) if np.random.random() < epsilon else self.act(state)

  def axu_loss_forward(self, obs, next_obs, action, reward, obs_aug, weights):
    assert obs.shape == next_obs.shape
    assert len(weights) == 5
    
    obs_h = self.online_net.convs(obs)
    h_shape = obs_h.shape
    obs_h = obs_h.view(-1, self.online_net.conv_output_size)
    
    next_obs_h = self.online_net.convs(next_obs)
    next_obs_h = next_obs_h.view(-1, self.online_net.conv_output_size)

    loss = 0
    if weights[0] > 0:
        # predict future
        action = F.one_hot(action, num_classes=self.action_space).float()
        act_h = self.online_net.act(action)
        pre_t = torch.cat([obs_h, act_h], dim=1)
        # print(self.online_net.pred_future(pre_t).shape, next_obs_h.detach().shape)
        pred_future_loss = self.mse_loss(self.online_net.pred_future(pre_t), next_obs_h.detach())
        # print(pred_future_loss)
        loss += weights[0] * pred_future_loss
      
    if weights[1] > 0:
        # extract reward
        pre_t = torch.cat([obs_h, next_obs_h], dim=1)
        extract_reward_loss = self.mse_loss(self.online_net.extract_reward(pre_t), reward.reshape(-1, 1))
        loss += weights[1] * extract_reward_loss
        # print(extract_reward_loss)
    
    if weights[2] > 0:
        # BYOL
        obs_h_2 = self.online_net.convs(obs_aug)
        obs_h_2 = obs_h_2.view(-1, self.online_net.conv_output_size)

        p1, p2 = self.online_net.predict(obs_h), self.online_net.predict(obs_h_2)
        t1, t2 = self.momentum_net.project(obs_h), self.momentum_net.project(obs_h_2)

        def loss_fn(x, y):
            x = F.normalize(x, dim=-1, p=2)
            y = F.normalize(y, dim=-1, p=2)
            return 2 - 2 * (x * y).sum(dim=-1)

        byol_loss = (0.5 * loss_fn(p1, t2) + 0.5 * loss_fn(p2, t1))
        byol_loss = byol_loss.mean()
        # print(byol_loss)
        loss += weights[2] * byol_loss
    
    if weights[3] > 0:
        # AE
        
        h = self.online_net.hnet(obs_h)
        r = h.reshape(h_shape)
        rec = self.online_net.decoder(r)
        r_shape = rec.shape

        rec_loss = self.mse_loss(rec, obs[:, :, :r_shape[-2], :r_shape[-1]])
        latent_loss = (0.5 * h.pow(2).sum(1)).mean()

        ae_loss = rec_loss + latent_loss * (1e-6) # latent lambda
        # print(ae_loss)
        loss += weights[3] * ae_loss

    
    if weights[4] > 0:
        # Rotation
        b = obs.size(0)
        labels = torch.arange(4, dtype=torch.long, device=obs.device).repeat_interleave(b)
        obs_cat = obs.repeat(4, 1, 1, 1)
        for i in range(4):
          obs_cat[i * b: (i + 1) * b] = torch.rot90(obs_cat[i * b:(i + 1) * b], i, [-2, -1])

        obs_cat = self.online_net.convs(obs_cat)
        rot_loss = self.ce_loss(self.online_net.rot_cls(obs_cat), labels)
        # print(rot_loss)
        loss += weights[4] * rot_loss
    
    return loss

  def learn(self, mem):
    # Sample transitions
    idxs, states, actions, returns, next_states, nonterminals, weights = mem.sample(self.batch_size)
    # print(states.shape)
    states = states.to(device=self.args.device)
    next_states = next_states.to(device=self.args.device)
    
    # augmentation for extract ar
    aug_states = aug(states).to(device=self.args.device)
    aug_next_states = aug(next_states).to(device=self.args.device)
    aug_states_2 = aug(states).to(device=self.args.device)
    
    aux_loss = self.axu_loss_forward(aug_states, aug_next_states, actions, returns, aug_states_2,self.weights)
    # print(logits.shape, actions.shape, returns.shape)
    
    
    # Calculate current state probabilities (online network noise already sampled)
    log_ps = self.online_net(states, log=True)  # Log probabilities log p(s_t, ·; θonline)

    log_ps_a = log_ps[range(self.batch_size), actions]  # log p(s_t, a_t; θonline)

    with torch.no_grad():
      # Calculate nth next state probabilities
      pns = self.online_net(next_states)  # Probabilities p(s_t+n, ·; θonline)
      dns = self.support.expand_as(pns) * pns  # Distribution d_t+n = (z, p(s_t+n, ·; θonline))
      argmax_indices_ns = dns.sum(2).argmax(1)  # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
      self.target_net.reset_noise()  # Sample new target net noise
      pns = self.target_net(next_states)  # Probabilities p(s_t+n, ·; θtarget)
      pns_a = pns[range(self.batch_size), argmax_indices_ns]  # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)

      # Compute Tz (Bellman operator T applied to z)
      Tz = returns.unsqueeze(1) + nonterminals * (self.discount ** self.n) * self.support.unsqueeze(0)  # Tz = R^n + (γ^n)z (accounting for terminal states)
      Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)  # Clamp between supported values
      # Compute L2 projection of Tz onto fixed support z
      b = (Tz - self.Vmin) / self.delta_z  # b = (Tz - Vmin) / Δz
      l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
      # Fix disappearing probability mass when l = b = u (b is int)
      l[(u > 0) * (l == u)] -= 1
      u[(l < (self.atoms - 1)) * (l == u)] += 1

      # Distribute probability of Tz
      m = states.new_zeros(self.batch_size, self.atoms)
      offset = torch.linspace(0, ((self.batch_size - 1) * self.atoms), self.batch_size).unsqueeze(1).expand(self.batch_size, self.atoms).to(actions)
      m.view(-1).index_add_(0, (l + offset).view(-1), (pns_a * (u.float() - b)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
      m.view(-1).index_add_(0, (u + offset).view(-1), (pns_a * (b - l.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)

    loss = -torch.sum(m * log_ps_a, 1)  # Cross-entropy loss (minimises DKL(m||p(s_t, a_t)))
    aux_loss = loss + (aux_loss * self.coeff)
    self.online_net.zero_grad()
    (weights * aux_loss).mean().backward()  # Backpropagate importance-weighted minibatch loss
    clip_grad_norm_(self.online_net.parameters(), self.norm_clip)  # Clip gradients by L2 norm
    self.optimiser.step()

    mem.update_priorities(idxs, loss.detach().cpu().numpy())  # Update priorities of sampled transitions

  def update_target_net(self):
    self.target_net.load_state_dict(self.online_net.state_dict())

  def initialize_momentum_net(self):
    for param_q, param_k in zip(self.online_net.parameters(), self.momentum_net.parameters()):
      param_k.data.copy_(param_q.data) # update
      param_k.requires_grad = False  # not update by gradient

  # Code for this function from https://github.com/facebookresearch/moco
  @torch.no_grad()
  def update_momentum_net(self, momentum=0.999):
    for param_q, param_k in zip(self.online_net.parameters(), self.momentum_net.parameters()):
      param_k.data.copy_(momentum * param_k.data + (1.- momentum) * param_q.data) # update

  # Save model parameters on current device (don't move model between devices)
  def save(self, path, name='model.pth'):
    torch.save(self.online_net.state_dict(), os.path.join(path, name))

  # Evaluates Q-value based on single state (no batch)
  def evaluate_q(self, state):
    with torch.no_grad():
      a = self.online_net(state.unsqueeze(0))
      return (a * self.support).sum(2).max(1)[0].item()

  def train(self):
    self.online_net.train()

  def eval(self):
    self.online_net.eval()
