
import argparse
from collections import deque
import itertools
import random
import time

import gym
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


class ReplayMemory:
    __slots__ = ['buffer']

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, *transition):
        # (state, action, reward, next_state, done)
        self.buffer.append(tuple(map(tuple, transition)))

    def sample(self, batch_size, device):
        '''sample a batch of transition tensors'''
        transitions = random.sample(self.buffer, batch_size)
        return (torch.tensor(x, dtype=torch.float, device=device) for x in zip(*transitions))


class Net(nn.Module):
    def __init__(self, args,env):
        super().__init__()
        self.action_dim = env.action_space.n
        self.observation_dim = env.observation_space.shape[0]
        self.batch_size = args.batch_size
        self.feature_dim = args.feature_dim
        self.alpha = 0.1
    #def feature_block(self,x):
        self.feature_block = nn.Sequential(
                            nn.Linear(self.observation_dim, self.feature_dim),
                            nn.ReLU()
                            )
        
    #def policy_block(self,f):
        self.A_block = nn.Sequential(          
                            nn.Linear(self.feature_dim, self.action_dim),
                            )
        
    #def V_block(self,f):
        self.V_block = nn.Sequential(
                            nn.Linear(self.feature_dim , 1),
                            )
        
    def forward(self, x):
        f = self.feature_block(x)
        A = self.A_block(f)
        probs = F.softmax(A,dim=1)
        V = self.V_block(f)
        A = A - (probs.detach() * A).sum()
        Q = A + V
        return Q
    
class PGQL:
    def __init__(self, args, env):
        # Two network use the same structure
        self._behavior_net = Net(args,env).to(args.device) # Main network
        self._target_net = Net(args,env).to(args.device) # update using main network's parameters after a while
        # initialize target network
        self._target_net.load_state_dict(self._behavior_net.state_dict())

        self._optimizer = optim.AdamW(self._behavior_net.parameters(), lr = args.lr)

        # memory
        self._memory = ReplayMemory(capacity=args.capacity)

        ## config ##
        self.device = args.device
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.freq = args.freq
        self.target_freq = args.target_freq


    def select_action(self, state, epsilon, action_space):
        '''epsilon-greedy based on behavior network'''

        state = torch.unsqueeze(torch.FloatTensor(state), 0).to(self.device)
        if np.random.uniform() < epsilon:
            action = action_space.sample()
        else:
            action_value = self._behavior_net.forward(state)
            action = torch.max(action_value, 1)[1].cpu().data.numpy()[0] 

        return action



    def append(self, state, action, reward, next_state, done):
        self._memory.append(state, [action], [reward], next_state, [int(done)])


    def update(self, total_steps):
        if total_steps % self.freq == 0:   
            self._update_behavior_network(self.gamma)
        if total_steps % self.target_freq == 0: 
            self._update_target_network()


    def _update_behavior_network(self, gamma):
        # sample a minibatch of transitions
        state, action, reward, next_state, done = self._memory.sample(self.batch_size, self.device)
        action = action.long() # change to LongTensor

        q_value = self._behavior_net(state).gather(1, action)
        with torch.no_grad(): 
            q_next = self._target_net(next_state)
            q_target = reward + gamma * q_next.max(1)[0].view(self.batch_size, 1)
            
        criterion = nn.MSELoss()
        loss = criterion(q_value, q_target)


        # optimize
        self._optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._behavior_net.parameters(), 5) # clip gradient, max_norm = 5
        self._optimizer.step()

    def _update_target_network(self):
        '''update target network by copying from behavior network'''
        
        self._target_net.load_state_dict(self._behavior_net.state_dict())


    def save(self, model_path, checkpoint=False):
        if checkpoint:
            torch.save(
                {
                    'behavior_net': self._behavior_net.state_dict(),
                    'target_net': self._target_net.state_dict(),
                    'optimizer': self._optimizer.state_dict(),
                }, model_path)
        else:
            torch.save({
                'behavior_net': self._behavior_net.state_dict(),
            }, model_path)


    def load(self, model_path, checkpoint=False):
        model = torch.load(model_path)
        self._behavior_net.load_state_dict(model['behavior_net'])
        if checkpoint:
            self._target_net.load_state_dict(model['target_net'])
            self._optimizer.load_state_dict(model['optimizer'])


def train(args, env, agent):
    print('Start Training')
    action_space = env.action_space
    total_steps, epsilon = 0, 1.
    ewma_reward = 0

    for episode in range(args.episode):
        total_reward = 0
        total_reward_original = 0
        state = env.reset()
        # itertools.count: infinite loop, benefit of itertools is that it does not actually use memory like range()
        for t in itertools.count(start=1):  # t is length of the landing process
            # select action
            if total_steps < args.warmup: # warmup
                action = action_space.sample()
            else:
                action = agent.select_action(state, epsilon, action_space)
                epsilon = max(epsilon * args.eps_decay, args.eps_min) # from 1 -> 0.01
            
            #print('action_train:', action)
            # execute action
            next_state, reward, done, _ = env.step(action)
            total_reward_original += 1

            # store transition
            if done:
                reward = -1

            agent.append(state, action, reward, next_state, done)
            if total_steps >= args.warmup: # start update after warm up
                agent.update(total_steps)  # update behavior net every 4 
            total_reward += reward
            state = next_state
            
            total_steps += 1

            if args.render:
                env.render()
                
            if done:
                ewma_reward = 0.05 * total_reward_original + (1 - 0.05) * ewma_reward

                
                print(
                    'Step: {}\tEpisode: {}\tLength: {:3d}\tTotal reward: {:.2f}\tEwma reward: {:.2f}\tEpsilon: {:.3f}'
                    .format(total_steps, episode, t, total_reward_original, ewma_reward, epsilon))
                
                break

        if ewma_reward > env.spec.reward_threshold: 
            
            print("Solved! Running reward is now {} and "
                      "the last episode runs to {} time steps!".format(ewma_reward, t))
            
            break

    env.close()
 
def test(args, env, agent):
    print('Start Testing')
    action_space = env.action_space
    epsilon = args.test_epsilon
    seeds = (args.seed + i for i in range(10))
    rewards = []
    for n_episode, seed in enumerate(seeds):
        total_reward = 0
        env.seed(seed)
        state = env.reset()
        ## TODO ##
        for t in itertools.count(start=1): 
            action = agent.select_action(state, epsilon, action_space)
            state, reward, done, _ = env.step(action)
            total_reward += reward
            if args.render:
                env.render()
            if done:
                print('Test/Episode Reward', total_reward, n_episode)
                rewards.append(total_reward)
                break

    print('Average Reward', np.mean(rewards))
    env.close()


def main():
    ## arguments ##
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--device', default='cpu')
    parser.add_argument('-m', '--model', default='PGQL.pth')
    # train
    parser.add_argument('--warmup', default=500, type=int)
    parser.add_argument('--episode', default=4000, type=int)
    parser.add_argument('--capacity', default=100000, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=.001 , type=float)
    parser.add_argument('--feature_dim', default=256 , type=float)
    parser.add_argument('--eps_decay', default=.9995, type=float)
    parser.add_argument('--eps_min', default=.001, type=float)
    parser.add_argument('--gamma', default=.8, type=float)
    parser.add_argument('--freq', default=1, type=int)
    parser.add_argument('--target_freq', default=4, type=int)
    # test
    # action='story_true': 如果這個選項存在,代表test_only = True
    # 除此之外,不能給這個選項賦值,不然會error
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--seed', default=20210628, type=int)
    parser.add_argument('--test_epsilon', default=.001, type=float)
    args = parser.parse_args()

    
    ## main ##
    env = gym.make('CartPole-v0')
    agent = PGQL(args, env)
    train(args, env, agent)
                
    agent.save(args.model)
    agent.load(args.model)
    test(args, env, agent)


if __name__ == '__main__':
    main()
