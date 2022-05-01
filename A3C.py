import argparse
from torch import nn
import torch
import numpy as np
import torch.nn.functional as F
import torch.multiprocessing as mp
import gym
import os
import itertools


def push_and_pull(opt, local_net, global_net, done, next_state, buffer_a,buffer_s,buffer_r, gamma):
    if done:
        v_s_ = 0.               # terminal
    else:
        v_s_ = local_net.forward(v_wrap(next_state[None, :]))[-1].data.numpy()[0, 0]

    buffer_v_target = []
    for r in buffer_r[::-1]:    # reverse buffer r
        v_s_ = r + gamma * v_s_
        buffer_v_target.append(v_s_)
    buffer_v_target.reverse()

    loss = local_net.loss_func(
    v_wrap(np.vstack(buffer_s)),
    v_wrap(np.array(buffer_a), dtype=np.int64) if np.array(buffer_a).dtype == np.int64 else v_wrap(np.vstack(buffer_a)),
    v_wrap(np.array(buffer_v_target)[:, None]))

    # calculate local gradients and push local parameters to global
    opt.zero_grad()
    loss.backward()
    for lp, gp in zip(local_net.parameters(), global_net.parameters()):
        gp._grad = lp.grad
    opt.step()

    # pull global parameters
    local_net.load_state_dict(global_net.state_dict())

def v_wrap(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)

def record(global_ep, global_ep_r, ep_r, res_queue, name,global_step):
    with global_ep.get_lock():
        global_ep.value += 1
        global_step.value += ep_r
    with global_ep_r.get_lock():
        if global_ep_r.value == 0.:
            global_ep_r.value = ep_r
        else:
            global_ep_r.value = global_ep_r.value * 0.95 + ep_r * 0.05
    res_queue.put(global_ep_r.value)
    print(name,
        'Step: {}\tEpisode: {}\tLength: {:3d}\tTotal reward: {:.2f}\tEwma reward: {:.2f}'
        .format(global_step.value, global_ep.value, ep_r, ep_r, global_ep_r.value))
    
class SharedAdam(torch.optim.AdamW):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
              weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

            # share in memory
            state['exp_avg'].share_memory_()
            state['exp_avg_sq'].share_memory_()

def set_init(layers):
    for layer in layers:
        nn.init.normal_(layer.weight, mean=0., std=0.1)
        nn.init.constant_(layer.bias, 0.)
    
class Net(nn.Module):
    def __init__(self,state_dim,action_dim):
        super(Net, self).__init__()
        self.observation_dim = state_dim
        self.action_dim = action_dim
        self.feature_dim = 256
        self.distribution = torch.distributions.Categorical
        self.feature_block = nn.Sequential(
                            nn.Linear(self.observation_dim, self.feature_dim),
                            nn.ReLU()
                            )
        
    #def policy_block(self,f):
        self.A_block = nn.Sequential(          
                            nn.Linear(self.feature_dim, self.action_dim),
                            nn.Softmax()
                            )
        
    #def V_block(self,f):
        self.V_block = nn.Sequential(
                            nn.Linear(self.feature_dim , 1),
                            )
        
    def forward(self, x):
        f = self.feature_block(x)
        probs = self.A_block(f)
        V = self.V_block(f)
        return probs , V
        
    def select_action(self, state):
        self.eval()
        probs , _ = self.forward(state)
        m = self.distribution(probs)
        return m.sample().item()

    def loss_func(self, state, action, v_t):
        self.train()
        probs, state_value = self.forward(state)
        td = v_t - state_value
        c_loss = td.pow(2)
      
        m = self.distribution(probs)
        exp_v = m.log_prob(action) * td.detach().squeeze()
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()
        return total_loss


class Worker(mp.Process):
    def __init__(self,state_dim,action_dim,global_net, opt, global_ep, global_ep_r, res_queue, num,env_name, args,global_step):
        super(Worker, self).__init__()
        self.env = gym.make(env_name)
        self.name = 'w%02i' % num
        print("worker_build",self.name)
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.global_step = global_step
        self.global_net, self.opt = global_net, opt
        self.local_net = Net(state_dim,action_dim )
        self.epi = args.episode
        self.gamma = args.gamma
        self.step = 0
        self.freq = args.freq
    def run(self):
        while self.g_ep.value < self.epi:
            state = self.env.reset()
            buffer_a = []
            buffer_s = []
            buffer_r = []
            total_reward = 0
            done = False
            for t in itertools.count(start=1):
                action = self.local_net.select_action(v_wrap(state[None, :]))
                next_state , reward , done, _ = self.env.step(action)
                
                if done: 
                    reward = -1
                total_reward += 1
                buffer_a.append(action)
                buffer_s.append(state)
                buffer_r.append(reward)
                # update global_net
                if t % self.freq == 0 or done:  
                    push_and_pull(self.opt, self.local_net, 
                                  self.global_net, done, next_state, 
                                  buffer_a,buffer_s,buffer_r,self.gamma)
                    if done :
                        record(self.g_ep, self.g_ep_r, total_reward, self.res_queue, self.name,self.global_step)
                        break
                state = next_state
                self.step += 1
                
            if self.g_ep_r.value > self.env.spec.reward_threshold:
                break
        print(self.name," in run done :", self.step)
        self.res_queue.put(None)                
                
def train(args,action_dim,state_dim):
    print('Start Training')
    global_net = Net(state_dim,action_dim)
    global_net.share_memory()
    opt = SharedAdam(global_net.parameters(), lr=args.lr)
    global_ep = mp.Value('i', 0)
    global_ep_r = mp.Value('d', 0.)
    global_step = mp.Value("i",0)
    res_queue = mp.Queue()
    workers = [Worker(state_dim,action_dim, global_net, opt, global_ep, global_ep_r, res_queue, i,args.env_name, args,global_step) for i in range(1,args.worker_num)]
    # share the global parameters in multiprocessing
    [worker.start() for worker in workers]
    res = []
    # send worker's process
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    [worker.join() for worker in workers]

    
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--device', default='cpu')
    parser.add_argument('-m', '--model', default='A3C.pth')
    parser.add_argument('--warmup', default=500, type=int)
    parser.add_argument('--episode', default=2000, type=int)
    parser.add_argument('--capacity', default=100000, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=.001 , type=float)
    parser.add_argument('--feature_dim', default=128 , type=float)
    parser.add_argument('--eps_decay', default=.999, type=float)
    parser.add_argument('--eps_min', default=.01, type=float)
    parser.add_argument('--gamma', default=.8, type=float)
    parser.add_argument('--freq', default=4, type=int)
    parser.add_argument('--target_freq', default=10, type=int)
    # test
    # action='story_true': 如果這個選項存在,代表test_only = True
    # 除此之外,不能給這個選項賦值,不然會error
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--seed', default=20200519, type=int)
    parser.add_argument('--test_epsilon', default=.001, type=float)
    parser.add_argument('--worker_num', default=5, type=int)
    parser.add_argument('--env_name', default="CartPole-v0")
    args = parser.parse_args()

    
    ## main ##
    #env = gym.make('CartPole-v0')
    #agent = PGQL(args, env)
    

    env = gym.make(args.env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    train(args,action_dim,state_dim)
                
if __name__ == "__main__":
    main()