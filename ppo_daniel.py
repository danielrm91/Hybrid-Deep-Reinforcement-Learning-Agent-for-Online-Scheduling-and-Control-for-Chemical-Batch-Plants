

'''
PPO class to train and learn the policy
'''

import gym
import gym_kondili
import time
import csv
import numpy as np
import torch
import random
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import torch.nn.functional as F

#OBS:
# OBSERVATION: [RI F RII S Ca Cb Cb Cd Ce Cf ptf pts ptRI ptRII E prfRI prfRII Ca0  Cd0 Ce  t]
#              [0  1  2  3 4  5  6  7  8  9  10  11  12   13    14  15    16    17  18  19 20]

class PPO:
    def __init__(self, policy_class, value_class, env, **hyperparameters):
        self.max_vals_obs = torch.ones(47, dtype=torch.float32)         #21
        self._init_hyperparameters(hyperparameters)
        self.env = env
        self.obs_dim =      47
        self.sen_dim =      100
        self.out_ac1_dim =  5       # 4 actions + idle      
        self.out_ac2_dim =  3       # 3 actions 
        self.out_ac3_dim =  3       # 3 actions 
        self.out_ac4_dim =  3
        # Initialize Actor and Critic
        self.actor = policy_class(self.obs_dim, self.sen_dim, self.out_ac1_dim, self.out_ac2_dim, self.out_ac3_dim, self.out_ac4_dim, self.seq_len)
        self.critic = value_class(self.obs_dim, 1, self.seq_len)
        # Initialize optimizers for Actor and Critic
        self.actor_optim  = Adam(self.actor.parameters(), lr= self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)
        
        # This logger will help for printing the summary
        self.logger = {
            'delta_t': time.time_ns(),
            't_so_far':         0,
            'i_so_far':         0, 
            'batch_lens':       [],
            'batch_rews':       [],
            'actord_losses1':   [],
            'actord_losses2':   [],
            'actord_losses3':   [],
            'actord_losses4':   [],
        }
        
    def learn(self, total_timesteps):
        print(f"Learning...Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
        print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
        t_so_far = 0
        i_so_far = 0
        while t_so_far < total_timesteps:

            t_it = total_timesteps / self.timesteps_per_batch                       # Number of total iterations in this training                        

            #Learning rate annealing
            new_lr = self.lr * (1e-6/self.lr)**(i_so_far/t_it)                      # initial * (final/initial)***(t/T)  
            self.actor_optim.param_groups[0]["lr"] = new_lr                         # 5e-8 min value of lr   
            self.critic_optim.param_groups[0]["lr"] = new_lr

            #Temperature annealing
            self.temperature = self.temp * (0.001/self.temp)**(i_so_far/t_it)                   # initial * (final/initial)***(t/T)

            #Entropy annealing
            self.ent_coeff = self.ent_init * (0.00001/self.ent_init)**(i_so_far/t_it)  # initial * (final/initial)***(t/T)

            #Calling Rollout
            batch_obs, batch_mask_vec, batch_dacts1, batch_dacts2, batch_dacts3,batch_dacts4, batch_log_probsd1, batch_log_probsd2, batch_log_probsd3, batch_log_probsd4, \
                batch_lens, batch_h, batch_c, batch_h_c, batch_c_c, batch_vals, batch_rews, done_flags = self.rollout() 

            t_so_far += np.sum(batch_lens)
            i_so_far += 1
            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far

            batch_h = batch_h.detach()              # Consider them as constants so the backpropagation does not alter them while updating
            batch_c = batch_c.detach()              # Consider them as constants so the backpropagation does not alter them while updating
            batch_h_c = batch_h_c.detach()          # Consider them as constants so the backpropagation does not alter them while updating
            batch_c_c = batch_c_c.detach()          # Consider them as constants so the backpropagation does not alter them while updating


            #Normalize the obs by dividing by max values
            batch_obs_n = np.divide(batch_obs, self.max_vals_obs)
            batch_obs_n = batch_obs_n.reshape(-1,self.seq_len,self.obs_dim)
            
            V = self.evaluate_V(batch_obs_n, batch_h_c, batch_c_c)

            #Minibatch construction
            b_size = batch_obs_n.size(0)
            idxs = np.arange(b_size)
            mini_batchsize = b_size//self.num_minibatches

            # Generalized Advantage Estimation function 
            A_k = self.GAE(batch_rews, batch_vals, done_flags)
            V = V.squeeze()
            batch_rtgs = A_k + V.detach()

            # For a more stable convergence
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            # Loop for updating the network n number of epochs
            for _ in range(self.n_updates_per_iteration):
                np.random.shuffle(idxs)
                for start in range(0, b_size, mini_batchsize):
                    end = start + mini_batchsize
                    id = idxs[start:end]
                    min_batch_obs_n          = batch_obs_n[id,:,:]
                    min_batch_mask_vec       = batch_mask_vec[id,:]
                    min_batch_dacts1         = batch_dacts1[id,:]
                    min_batch_dacts2         = batch_dacts2[id,:]
                    min_batch_dacts3         = batch_dacts3[id,:]
                    min_batch_dacts4         = batch_dacts4[id,:]
                    min_batch_log_probsd1    = batch_log_probsd1[id] 
                    min_batch_log_probsd2    = batch_log_probsd2[id]
                    min_batch_log_probsd3    = batch_log_probsd3[id]
                    min_batch_log_probsd4    = batch_log_probsd4[id]
                    min_batch_rtgs           = batch_rtgs[id]
                    min_batch_h              = batch_h[:,id,:]
                    min_batch_c              = batch_c[:,id,:]
                    min_batch_h_c            = batch_h_c[:,id,:]
                    min_batch_c_c            = batch_c_c[:,id,:]
                    min_A_k                  = A_k[id]

                    # Update Discrete 1
                    curr_log_probsd1, entropyd1 = self.evaluate_d1(min_batch_obs_n, min_batch_mask_vec, min_batch_dacts1, min_batch_h, min_batch_c)
                    logratios_d1 = curr_log_probsd1 - min_batch_log_probsd1
                    ratiod1 = torch.exp(logratios_d1)
                    surr1d1 = ratiod1 * min_A_k
                    surr2d1 = torch.clamp(ratiod1, 1-self.clip, 1+self.clip) * min_A_k
                    actord_loss1 = (-torch.min(surr1d1, surr2d1)).mean() - entropyd1.mean() * self.ent_coeff
                    self.logger['actord_losses1'].append(actord_loss1.detach())
                    # Deactivate the gradients from the other two discrete actions
                    for i, param in enumerate(self.actor.parameters()):
                        if i in [22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39]:
                            param.requires_grad = False
                    # Optimize the whole network (with the frozen params)
                    self.actor_optim.zero_grad()
                    actord_loss1.backward(retain_graph=True)
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    self.actor_optim.step()
                    # Reactivate everything
                    for i, param in enumerate(self.actor.parameters()):
                        param.requires_grad = True

                    # Update Discrete 2
                    curr_log_probsd2, entropyd2 = self.evaluate_d2(min_batch_obs_n, min_batch_mask_vec, min_batch_dacts2, min_batch_h, min_batch_c)
                    logratios_d2 = curr_log_probsd2 - min_batch_log_probsd2
                    ratiod2 = torch.exp(logratios_d2)
                    surr1d2 = ratiod2 * min_A_k
                    surr2d2 = torch.clamp(ratiod2, 1-self.clip, 1+self.clip) * min_A_k
                    actord_loss2 = (-torch.min(surr1d2, surr2d2)).mean() - entropyd2.mean() * self.ent_coeff
                    self.logger['actord_losses2'].append(actord_loss2.detach())
                    #Code to see the names of the layers
                    #for name, param in self.actor.named_parameters(): 
                    #    if param.requires_grad: 
                    #        print(name,param.data.shape)
                    #for i, param in enumerate(self.actor.parameters()):
                    #    print(i,param.data.shape )
                    # Deactivate the gradients from the other two discrete actions
                    for i, param in enumerate(self.actor.parameters()):
                        if i in [16,17,18,19,20,21,28,29,30,31,32,33,34,35,36,37,38,39]:
                            param.requires_grad = False
                    # Optimize the whole network (with the frozen params)
                    self.actor_optim.zero_grad()
                    actord_loss2.backward(retain_graph=True)
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    self.actor_optim.step()
                    # Reactivate everything
                    for i, param in enumerate(self.actor.parameters()):
                        param.requires_grad = True
                        
                    # Update Discrete 3
                    curr_log_probsd3, entropyd3 = self.evaluate_d3(min_batch_obs_n, min_batch_mask_vec, min_batch_dacts3, min_batch_h, min_batch_c)
                    logratios_d3 = curr_log_probsd3 - min_batch_log_probsd3
                    ratiod3 = torch.exp(logratios_d3)
                    surr1d3 = ratiod3 * min_A_k
                    surr2d3 = torch.clamp(ratiod3, 1-self.clip, 1+self.clip) * min_A_k
                    actord_loss3 = (-torch.min(surr1d3, surr2d3)).mean() - entropyd3.mean() * self.ent_coeff
                    self.logger['actord_losses3'].append(actord_loss3.detach())
                    # Deactivate the gradients from the other two discrete actions
                    for i, param in enumerate(self.actor.parameters()):
                        if i in [16,17,18,19,20,21,22,23,24,25,26,27,34,35,36,37,38,39]:
                            param.requires_grad = False
                    # Optimize the whole network (with the frozen params)
                    self.actor_optim.zero_grad()
                    actord_loss3.backward(retain_graph=True)
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    self.actor_optim.step()
                    # Reactivate everything
                    for i, param in enumerate(self.actor.parameters()):
                        param.requires_grad = True

                    # Update Discrete 4
                    curr_log_probsd4, entropyd4 = self.evaluate_d4(min_batch_obs_n, min_batch_mask_vec, min_batch_dacts4, min_batch_h, min_batch_c)
                    logratios_d4 = curr_log_probsd4 - min_batch_log_probsd4
                    ratiod4 = torch.exp(logratios_d4)
                    surr1d4 = ratiod4 * min_A_k
                    surr2d4 = torch.clamp(ratiod4, 1-self.clip, 1+self.clip) * min_A_k
                    actord_loss4 = (-torch.min(surr1d4, surr2d4)).mean() - entropyd4.mean() * self.ent_coeff
                    self.logger['actord_losses4'].append(actord_loss4.detach())
                    # Deactivate the gradients from the other two discrete actions
                    for i, param in enumerate(self.actor.parameters()):
                        if i in [16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33]:
                            param.requires_grad = False
                    # Optimize the whole network (with the frozen params)
                    self.actor_optim.zero_grad()
                    actord_loss4.backward(retain_graph=True)
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    self.actor_optim.step()
                    # Reactivate everything
                    for i, param in enumerate(self.actor.parameters()):
                        param.requires_grad = True
                    
                    # Update the Critic 
                    V = self.evaluate_V(min_batch_obs_n, min_batch_h_c, min_batch_c_c)
                    critic_loss = nn.MSELoss()(V, min_batch_rtgs)
                    self.critic_optim.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    self.critic_optim.step()
                
            # Print a summary
            self._log_summary()
            # Save the model if it's time
            if i_so_far % self.save_freq == 0:
                torch.save(self.actor.state_dict(),  './ppo_actor_exp4.pth')
                torch.save(self.critic.state_dict(), './ppo_critic_exp4.pth')
    
    def rollout(self):
        batch_obs =         []
        batch_dacts1 =      []
        batch_dacts2 =      []
        batch_dacts3 =      []
        batch_dacts4 =      []
        batch_log_probs1 =  []
        batch_log_probs2 =  []
        batch_log_probs3 =  []
        batch_log_probs4 =  []
        batch_rews =        []
        batch_lens =        []
        batch_mask_vec =    []
        done_flags =        []
        batch_vals =        []
        batch_h =           []
        batch_c =           []
        batch_h_c =         []
        batch_c_c =         []

        t = 0
        # Episode run
        while t < self.timesteps_per_batch:
            ep_rews = []
            ep_vals = []
            ep_dones = []
            done = False
            prev = torch.zeros([self.seq_len,self.obs_size])  
            mask_vec = self.env.reset_m()
            hist = torch.zeros(12)
            h, c     = (torch.zeros((3,1,25)),torch.zeros((3,1,25)))      #LAYERS, BATCHES, HIDDEN SIZE
            h_c, c_c = (torch.zeros((3,1,25)),torch.zeros((3,1,25)))  
            for ep_t in range(self.max_timesteps_per_episode):    
                t += 1     
                batch_obs.append(torch.clone(prev)) #prev_m
                batch_mask_vec.append(mask_vec)
                batch_h.append(h)
                batch_c.append(c)
                batch_h_c.append(h_c)
                batch_c_c.append(c_c)
                d_1, d_2, d_3, d_4, log_probd1, log_probd2, log_probd3, log_probd4, h1, c1 = self.get_action(prev, mask_vec, h, c)  #prev_m
                val,h_c1,c_c1 = self.critic(prev, h_c, c_c)   #prev_m
                h_c = h_c1
                c_c = c_c1
                h = h1
                c = c1
                call = torch.clone(prev[-1])
                obs1, rew, done, mask_vec,hist = self.env.step1(d_1, d_2, d_3, d_4, call, hist)
                ep_dones.append(done)
                obs1 = obs1.reshape([1,self.obs_size])
                prev = torch.cat((prev,obs1),0)        #prev_m
                prev = prev[1:self.seq_len+1]
                ep_rews.append(rew)
                ep_vals.append(val.flatten())
                batch_dacts1.append(d_1)
                batch_dacts2.append(d_2)
                batch_dacts3.append(d_3)
                batch_dacts4.append(d_4)
                batch_log_probs1.append(log_probd1)
                batch_log_probs2.append(log_probd2) 
                batch_log_probs3.append(log_probd3) 
                batch_log_probs4.append(log_probd4)
                if done:
                    break
            batch_vals.append(ep_vals)
            done_flags.append(ep_dones)
            batch_lens.append(ep_t+1)
            batch_rews.append(ep_rews)
        # Reshape data as tensors in the appropiate shape
        batch_h         = torch.concat(batch_h, dim=1)
        batch_c         = torch.concat(batch_c, dim=1)  
        batch_h_c         = torch.concat(batch_h_c, dim=1)
        batch_c_c         = torch.concat(batch_c_c, dim=1) 
        batch_obs        = torch.concat(batch_obs)
        batch_mask_vec   = np.array(batch_mask_vec)
        batch_mask_vec   = torch.tensor(batch_mask_vec, dtype=bool)
        batch_dacts1      = np.array(batch_dacts1)
        batch_dacts1      = torch.tensor(batch_dacts1, dtype=torch.int)
        batch_dacts2      = np.array(batch_dacts2)
        batch_dacts2      = torch.tensor(batch_dacts2, dtype=torch.int)
        batch_dacts3      = np.array(batch_dacts3)
        batch_dacts3      = torch.tensor(batch_dacts3, dtype=torch.int)
        batch_dacts4      = np.array(batch_dacts4)
        batch_dacts4      = torch.tensor(batch_dacts4, dtype=torch.int)
        batch_log_probsd1 = torch.tensor(batch_log_probs1, dtype=torch.float)
        batch_log_probsd2 = torch.tensor(batch_log_probs2, dtype=torch.float)
        batch_log_probsd3 = torch.tensor(batch_log_probs3, dtype=torch.float)
        batch_log_probsd4 = torch.tensor(batch_log_probs4, dtype=torch.float)

        # Save the episodic returns and lenghts
        self.logger['batch_rews'] = batch_rews
        self.logger['batch_lens'] = batch_lens
        return batch_obs, batch_mask_vec, batch_dacts1, batch_dacts2, batch_dacts3, batch_dacts4, batch_log_probsd1, batch_log_probsd2, batch_log_probsd3,batch_log_probsd4,\
              batch_lens, batch_h, batch_c, batch_h_c, batch_c_c, batch_vals, batch_rews, done_flags
    
    def GAE(self, rewards, values, dones):
        advs = []
        for ep_rews, ep_vals, ep_done in zip (rewards,values,dones):
            advantages = []
            last_adv = 0
            for t in reversed(range(len(ep_rews))):
                if t + 1 < len(ep_rews):
                    delta = ep_rews[t] + self.gamma * ep_vals[t+1] * (1 - ep_done[t+1]) - ep_vals[t]
                else:
                    delta = ep_rews[t] - ep_vals[t]    
                advantage = delta + self.gamma * self.lam * (1-ep_done[t]) * last_adv
                last_adv = advantage
                advantages.insert(0,advantage)
            advs.extend(advantages)
        return torch.tensor(advs, dtype=torch.float)
    
    def get_action(self, obs2, mask_vec, h,c):
        # Normalize the obs by dividing by max values
        obs_n = torch.divide(obs2,self.max_vals_obs)
        # Get action
        act_d1, act_d2, act_d3, act_d4, h, c= self.actor(obs_n, mask_vec, h, c)
        # Apply Temperature
        act_d1 = act_d1/self.temperature
        act_d1 = F.softmax(act_d1, dim=-1)
        cat1 = Categorical(act_d1)

        act_d2 = act_d2/self.temperature
        act_d2 = F.softmax(act_d2, dim=-1)
        cat2 = Categorical(act_d2)

        act_d3 = act_d3/self.temperature
        act_d3 = F.softmax(act_d3, dim=-1)
        cat3 = Categorical(act_d3)

        act_d4 = act_d3/self.temperature
        act_d4 = F.softmax(act_d4, dim=-1)
        cat4 = Categorical(act_d4)
        # Sample an action from the distributions
        action_d1 = cat1.sample()
        action_d2 = cat2.sample()
        action_d3 = cat3.sample()
        action_d4 = cat4.sample()
        # Calculate the log probability
        log_prob_d1 = cat1.log_prob(action_d1)
        log_prob_d2 = cat2.log_prob(action_d2)
        log_prob_d3 = cat3.log_prob(action_d3)
        log_prob_d4 = cat4.log_prob(action_d4)
        # Detach and make np arrays
        action_d1 = action_d1.detach().numpy()
        action_d2 = action_d2.detach().numpy()
        action_d3 = action_d3.detach().numpy()
        action_d4 = action_d4.detach().numpy()

        log_prob_d1 = log_prob_d1.detach()
        log_prob_d2 = log_prob_d2.detach()
        log_prob_d3 = log_prob_d3.detach()  
        log_prob_d4 = log_prob_d4.detach()   
        return action_d1, action_d2, action_d3, action_d4, log_prob_d1, log_prob_d2, log_prob_d3, log_prob_d4, h, c 
    
    def evaluate_V(self, batch_obsv, batch_h, batch_c):
        V, _, _ = self.critic(batch_obsv, batch_h, batch_c)
        V = V.squeeze()
        return V
    
    def evaluate_d1(self, batch_obsd, batch_mask_vec, batch_dacts, batch_h, batch_c):
        act_d, _ , _ , _, _, _= self.actor(batch_obsd, batch_mask_vec, batch_h, batch_c)
        act_d = act_d/self.temperature
        act_d = F.softmax(act_d, dim=-1)
        meand = torch.tensor([])
        mean_en = torch.tensor([])
        for i in range(len(batch_obsd)):
            cat = Categorical(act_d[i])
            meand = torch.cat((meand, cat.log_prob(batch_dacts[i])))
            cat_ent = torch.tensor([cat.entropy().detach()])
            mean_en = torch.cat((mean_en,cat_ent))
        return meand, mean_en
        
    def evaluate_d2(self, batch_obsd, batch_mask_vec, batch_dacts, batch_h, batch_c):
        _, act_d , _ , _, _, _= self.actor(batch_obsd, batch_mask_vec, batch_h, batch_c)
        act_d = act_d/self.temperature
        act_d = F.softmax(act_d, dim=-1)
        meand = torch.tensor([])
        mean_en = torch.tensor([])
        for i in range(len(batch_obsd)):
            cat = Categorical(act_d[i])
            meand = torch.cat((meand, cat.log_prob(batch_dacts[i])))
            cat_ent = torch.tensor([cat.entropy().detach()])
            mean_en = torch.cat((mean_en,cat_ent))
        return meand, mean_en
    
    def evaluate_d3(self, batch_obsd, batch_mask_vec, batch_dacts, batch_h, batch_c):
        _, _ , act_d , _, _, _ = self.actor(batch_obsd, batch_mask_vec, batch_h, batch_c)
        act_d = act_d/self.temperature
        act_d = F.softmax(act_d, dim=-1)
        meand = torch.tensor([])
        mean_en = torch.tensor([])
        for i in range(len(batch_obsd)):
            cat = Categorical(act_d[i])
            meand = torch.cat((meand, cat.log_prob(batch_dacts[i])))
            cat_ent = torch.tensor([cat.entropy().detach()])
            mean_en = torch.cat((mean_en,cat_ent))
        return meand, mean_en
    
    def evaluate_d4(self, batch_obsd, batch_mask_vec, batch_dacts, batch_h, batch_c):
        _, _ , _ , act_d, _, _= self.actor(batch_obsd, batch_mask_vec, batch_h, batch_c)
        act_d = act_d/self.temperature
        act_d = F.softmax(act_d, dim=-1)
        meand = torch.tensor([])
        mean_en = torch.tensor([])
        for i in range(len(batch_obsd)):
            cat = Categorical(act_d[i])
            meand = torch.cat((meand, cat.log_prob(batch_dacts[i])))
            cat_ent = torch.tensor([cat.entropy().detach()])
            mean_en = torch.cat((mean_en,cat_ent))
        return meand, mean_en
    
    def _init_hyperparameters(self, hyperparameters):
        self.timesteps_per_batch =          120
        self.seq_len =                        4 
        self.obs_size =                      47
        self.max_steps_per_episodes =        30
        self.n_updates_per_iteration =        5
        self.lr =                          1e-4
        self.gamma =                       0.99
        self.clip =                         0.2
        self.save_freq =                     10
        self.seed =                        None
        self.var =                         1e-2
        self.temp =                           1
        self.ent_coeff =                   0.20
        self.num_minibatches =                5
        self.max_grad_norm =                0.5
        self.lam =                         0.98
        self.target_kl =                   0.02
        
        for param, val in hyperparameters.items():
            exec('self.' + param + '=' + str(val))

        if self.seed != None:
            assert(type(self.seed) == int)
            torch.manual_seed(self.seed)
            print(f'Succesfully set used to {self.seed}')
    
    def _log_summary(self):
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        delta_t = str(round(delta_t,2))
        
        t_so_far = self.logger['t_so_far']
        i_so_far = self.logger['i_so_far']
        avg_ep_lens = np.mean(self.logger['batch_lens'])
        avg_ep_rews2 = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
        avg_actord_loss1 = np.mean([losses.float().mean() for losses in self.logger['actord_losses1']])
        avg_actord_loss2 = np.mean([losses.float().mean() for losses in self.logger['actord_losses2']])
        avg_actord_loss3 = np.mean([losses.float().mean() for losses in self.logger['actord_losses3']])
        avg_actord_loss4 = np.mean([losses.float().mean() for losses in self.logger['actord_losses4']])
        
        # Rounding to decimal places
        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(avg_ep_rews2, 2))       
        avg_actord_loss1 = str(round(avg_actord_loss1, 5))
        avg_actord_loss2 = str(round(avg_actord_loss2, 5))
        avg_actord_loss3 = str(round(avg_actord_loss3, 5))
        avg_actord_loss4 = str(round(avg_actord_loss4, 5))

        # Save values of rewards
        avg_ep_rews1 = (round(avg_ep_rews2,2))
        f = open('rewards_exp1_2', 'a')
        writer = csv.writer(f, lineterminator = '\n')
        writer.writerow([avg_ep_rews1])
        f.close()
        
        # Print 
        print(flush=True)
        print(f"----------------- Iteration #{i_so_far}-----------------", flush=True)
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Average Discrete1 Loss: {avg_actord_loss1}", flush=True) 
        print(f"Average Discrete2 Loss: {avg_actord_loss2}", flush=True)
        print(f"Average Discrete3 Loss: {avg_actord_loss3}", flush=True)
        print(f"Average Discrete4 Loss: {avg_actord_loss4}", flush=True)
        print(f"Timesteps so Far: {t_so_far}", flush=True)
        print(f"Iteration took: {delta_t}", flush=True)
        print(f"--------------------------------------------------------", flush=True)
        print(flush=True)
        
        # Reset batch-specific logging data
        self.logger['batch_lens'] = []
        self.logger['batch_rews'] = []
        self.logger['actord_losses1'] = []
        self.logger['actord_losses2'] = []
        self.logger['actord_losses3'] = []
        self.logger['actord_losses4'] = []
