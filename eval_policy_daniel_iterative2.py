'''
Use it for evaluate the policy
'''
import numpy as np
import torch
import pandas as pd
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from scipy import stats as st
import torch.nn.functional as F

def _log_summary(ep_len, ep_ret, ep_num):
    #Round the values
    ep_len = str(round(ep_len,2))
    #ep_ret = str(round(ep_ret,2))
    
    #Print statements
    print(flush=True)
    print(f"-------------------- Episode #{ep_num} --------------------", flush=True)
    print(f"Episodic Length: {ep_len}", flush=True)
    print(f"Episodic Return: {ep_ret}", flush=True)
    print(f"------------------------------------------------------", flush=True)
    print(flush=True)
    
def rollout(policy, env):
    #Rollout until user kills process
    max_timesteps_per_episode = 30
    prev = torch.zeros([4,47])      #21
    mask_vec = env.reset_m()
    done = False
    t = 0
    ep_len = 0
    ep_ret = 0
    f = []
    h_, c_ = (torch.zeros((3,1,25)),torch.zeros((3,1,25)))
    hist = torch.zeros(12)
    for ep_t in range(max_timesteps_per_episode + 1):  
        max_vals_obs = torch.ones(47, dtype=torch.float32)      #21
        t += 1
        obs_normalized = np.divide(prev, max_vals_obs)
        daction1, daction2, daction3, daction4, h_, c_ = get_action(policy, obs_normalized, mask_vec, h_, c_)                            
        # Sample an action from the distributions
        a = torch.clone(prev[-1][:22])
        a = a.numpy()
        b = [daction1[0][0]] #this is a numpy
        c = [daction2[0][0]] 
        d = [daction3[0][0]]
        d1 =[daction4[0][0]]
        e = np.concatenate((a,b,c,d,d1))
        f.append(e)  

        calle = torch.clone(prev[-1]) 
        obs3, rew, done, mask_vec, hist = env.step1(daction1[0], daction2[0],daction3[0],daction4[0], calle,hist)
        ep_ret += rew
        obs3 = obs3.reshape([1,47])         #21
        prev = torch.cat((prev,obs3),0)    #prev_m
        prev = prev[1:4+1] 

    #Last observation withs actions 10 and 0
    a = torch.clone(prev[-1][:22])
    a = a.numpy()
    b = [np.array(0)] #this is a numpy
    c = [np.array(0)]
    d = [np.array(0)]
    d1 =[np.array(0)]
    e = np.concatenate((a,b,c,d,d1))
    e = np.round(e, 3)
    f.append(e)
    df = pd.DataFrame(f) #convert to a dataframe
    df.columns = ['RI', 'F', 'RII', 'S', 'Ca', 'Cb', 'Cb', 'Cd', 'Ce', 'Cf', 'ptf', 'pts', 'ptRI', 'ptRII', 'E', 'prfRI', 'prfRI' ,'prfRII', 'Ca0', 'Cd0', 'Ce', 't','Disc1','Disc2','Disc3','Disc4']
    df.to_csv("test_file.csv", index=False) #save to file
    ep_len = t
    yield ep_len, ep_ret
    
    
def get_action(policy, obs_n, mask_vec, h, c):
    d_iter1 = []
    d_iter2 = []  
    d_iter3 = []
    d_iter4 = []
    # Get action
    act_d1, act_d2, act_d3, act_d4, h, c = policy(obs_n, mask_vec, h, c)
    act_d1 = act_d1/0.001
    act_d2 = act_d2/0.001
    act_d3 = act_d3/0.001
    act_d4 = act_d4/0.001
    act_d1 = F.softmax(act_d1,dim=-1)
    act_d2 = F.softmax(act_d2,dim=-1)
    act_d3 = F.softmax(act_d3,dim=-1)
    act_d4 = F.softmax(act_d4,dim=-1)
    cat1 = Categorical(act_d1)
    cat2 = Categorical(act_d2)
    cat3 = Categorical(act_d3)
    cat4 = Categorical(act_d4)
    ###################### Maximum #################
    action_d1 = torch.argmax(cat1.probs)
    action_d2 = torch.argmax(cat2.probs)
    action_d3 = torch.argmax(cat3.probs)
    action_d4 = torch.argmax(cat4.probs)
    ###################### Sampling ################   Erase when not using samples from the covariance or the model
    action_d1 = cat1.sample()
    action_d2 = cat2.sample()
    action_d3 = cat3.sample()
    action_d4 = cat4.sample()
    ################################################
    action_d1 = torch.reshape(action_d1,(1,))
    action_d1 = action_d1.detach().numpy()
    d_iter1.append(action_d1)
    daction1 = d_iter1##############################
    action_d2 = torch.reshape(action_d2,(1,))
    action_d2 = action_d2.detach().numpy()
    d_iter2.append(action_d2)
    daction2 = d_iter2##############################
    action_d3 = torch.reshape(action_d3,(1,))
    action_d3 = action_d3.detach().numpy()
    d_iter3.append(action_d3)
    daction3 = d_iter3##############################
    action_d4 = torch.reshape(action_d4,(1,))
    action_d4 = action_d4.detach().numpy()
    d_iter4.append(action_d4)
    daction4 = d_iter4##############################

    return daction1, daction2, daction3, daction4, h, c
    
def eval_policy(policy, env, render=False):
    open('test_file.csv','w').close
    for ep_num, (ep_len, ep_ret) in enumerate(rollout(policy,env)):
        _log_summary(ep_len=ep_len, ep_ret=ep_ret, ep_num=ep_num)