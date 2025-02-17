import gym 
import numpy as np
from gym import spaces
import torch
from random import uniform
from TF_profiles import profiles
from scipy.integrate import solve_ivp

# Define the constants for R1
k1      = 1e7    # h**-1
E1      = 5000   # K
k2      = 1e10   # h**-1
E2      = 8000   # K
DH1     = -1e-4  # kJ/kmol
DH2     = -1e-6  # kJ/kmol
rho1    = 1e3    # kg/m3
cr1     = 2.5    # kJ/kg/K
UA      = 8e4    # kJ/m2/K/h
Vr      = 5      # m3             ????
Vj      = 1.5    # m3
rho2    = 1e3    # kg/m3
cj2     = 2.5    # Kj/kg/K
Fh      = 1      # m3/h
Fc      = 1      # m3/h
Th      = 350
Tc      = 294

# Define the constants for R2
k3      = 2      # m3/kmol/h
k4      = 1      # m3/kmol/h
cdf     = 0.8     # kmol/m3

RI_CBout    = 0.7   # Limits for concentration
RII_CEout   = 0.5
lim_CF      = 0.15

# Time span for the integration (start, end)
t_span = (0,2.0)

#capacities       
sepcap  =  5 

#times
RI_pt   = 2
filt_pt = 2
sep_pt  = 2
RII_pt  = 2

# rewards for H and C profiles 
h_rev  = [0,3,6,9]  #revenue for using: no, high, med, and low
c_rev  = [0,3,6,9]  #revenue for using: no, high, med, and low
dp_rev = [0,6,12,18]  #revenue for using: no, high, med, and low

# OBSERVATION: [RI F RII S Ca Cb Cb Cd Ce Cf ptf pts ptRI ptRII E prfRI prfRII Ca0  Cd0 Ce  t]
#              [0  1  2  3 4  5  6  7  8  9  10  11  12   13    14  15    16    17  18  19 20]

##################################### Move products to next stage #################################

class KondiliEnv(gym.Env):           #action    #accumulated times
    def __init__(self):   
        self.observation_space = spaces.Dict({"agent" : spaces.Box(0, 1, shape=(2,), dtype=int), "target": spaces.Box(0, 1, shape=(2,), dtype=int),})
        self.action_space = spaces.Discrete(5)
        self.max_steps = 30 #for a decision every half hour
        self.max_val = 1.0

    def systemRI(self, t, y, h1, h2, h3, h4, c1, c2, c3, c4):
        ca, cb, TR, TJ = y
        Fh_t = profiles('hot',  h1, h2, h3, h4, t)  # Fh(t)
        Fc_t = profiles('cold', c1, c2, c3, c4, t)  # Fc(t)
        r1 = k1 * np.exp(-E1 / TR) * ca
        r2 = k2 * np.exp(-E2 / TR) * cb
        dAdt = -r1 
        dBdt = r1 - r2
        dTRdt = -((DH1 * r1 + DH2 * r2)/(rho1 * cr1)) + ((UA * (TJ - TR))/(Vr * rho1 * cr1))
        dTJdt = ((Fh_t * (Th - TJ)) / Vj) + ((Fc_t * (Tc - TJ)) / Vj) + ((UA * (TR - TJ)) / (Vj * rho2 * cr1))
        return [dAdt, dBdt, dTRdt, dTJdt]

    def systemRII (self,t, y, d1, d2, d3, d4, CDF):
        V, cb, cd, ce, cf = y
        F_in = profiles('d_prof',d1, d2, d3, d4, t)
        r3 = k3 * cb * cd
        r4 = k4 * cd**2
        dVdt = F_in
        dBdt = -r3 - (F_in/Vr) * cb 
        dDdt = -r3 - 2 * r4 + (F_in/Vr) * (CDF - cd)
        dEdt =  r3 - (F_in/Vr) * ce
        dFdt =  r4 - (F_in/Vr) * cf
        return [dVdt, dBdt, dDdt, dEdt, dFdt]

    def step1(self, d_action1, d_action2, d_action3, d_action4, observations, hist):  
        unit       = d_action1             # hot profile
        hot        = d_action2 +1          # hot and cold profile
        cold       = d_action3 +1          # hot and cold profile
        D_p        = d_action4 +1          # D profile
        obs        = observations  
        r          = 0
        done       = False

        Ca0     =  uniform(0.88, 0.94)    # Uncertainty parameter  0.91
        Cd0     =  uniform(0.78, 0.91)     # Uncertainty parameter 0.845

        # Initial conditions [ca0, cb0, TR0, TJ0] and [V,cb,cd,ce,cf]
        #y0r1 = [Ca0, 0.0, 294.0, 294.0]  
        #y0r2 = [Vr,  1.0, Cd0, 0.0, 0.0]

        # Mask
        available_actions = np.array([1,1,1,1,1], dtype=np.int32)
        # rewards:
        rew_h = h_rev[int(hot)]
        rew_c = c_rev[int(cold)]
        rew_d = dp_rev[int(D_p)]
        
    ##################################### Move products to next stage #################################
        #Reaction 1
        if obs[0] == 1:
            obs[12] -= 0.5
            y0r1 = [obs[18], 0.0, 294.0, 294.0]     # Initial conditions
            if obs[12] == 0.5:                            # If it ends check concentration and give reward
                obs[15] = torch.from_numpy(hot)
                obs[16] = torch.from_numpy(cold)
                ix = int(4 - (obs[12] / 0.5))
                hist[ix-1] = torch.from_numpy(hot)
                hist[3 + ix] = torch.from_numpy(cold) 
                sol = solve_ivp(self.systemRI, t_span, y0r1, args=(hist[:8]), t_eval=[RI_pt]) 
                obs[4] = torch.from_numpy(sol.y[0])
                obs[5] = torch.from_numpy(sol.y[1])
                CB_f = obs[5]         
                hist[[0,1,2,3,4,5,6,7]] = 0        
                if CB_f >= RI_CBout:
                    r += rew_h + rew_c + 35
                else:
                    obs[[0,4,5,12,15,16,18]] = 0                                            
                    hist[[0,1,2,3,4,5,6,7]] = 0
                    r -= ix * h_rev[3]
            elif obs[12] > 0.5:                        # update the value in the state if we're in the middle of the process
                obs[15] = torch.from_numpy(hot)
                obs[16] = torch.from_numpy(cold)
                ix = int(4 - (obs[12] / 0.5))
                hist[ix-1] = torch.from_numpy(hot)
                hist[3 + ix] = torch.from_numpy(cold)  
                sol = solve_ivp(self.systemRI, t_span, y0r1, args=(hist[:8]), t_eval=[RI_pt - obs[12] + 0.5])      
                obs[4] = torch.from_numpy(sol.y[0])
                obs[5] = torch.from_numpy(sol.y[1])
                r += rew_h + rew_c
            elif obs[12] < 0:
                r -= 10

        #Filtration 
        if obs[1] == 1:
            obs[10] -= 0.5         #Value decreased at each time interval
            if obs[10] > 0:
                r += 10
            elif obs[10] < 0:
                r -= 10      # give a penalty for every time interval that the product is inside the state
            elif obs[10] == 0:
                r += 15
            #if obs[10] == filt_pt - 1.0:
            #    obs[[4,5,12,15,16,18]] = 0      # take it away from the reactor I
            

        #Reaction 2
        if obs[2] == 1:
            obs[13] -= 0.5
            y0r2 = [Vr,  1.0, obs[19], 0.0, 0.0]
            if obs[13] == 0.5:
                obs[17] = torch.from_numpy(D_p)
                ix = int(4 - (obs[13] / 0.5))
                hist[7 + ix] = torch.from_numpy(D_p)
                args = torch.cat((hist[8:],obs[19].view(1)))  
                sol = solve_ivp(self.systemRII, t_span, y0r2, args=args, t_eval=[RII_pt])
                obs[6] = torch.from_numpy(sol.y[1])   #Cb
                obs[7] = torch.from_numpy(sol.y[2])   #Cd
                obs[8] = torch.from_numpy(sol.y[3])   #Ce
                obs[9] = torch.from_numpy(sol.y[4])   #Cf
                #vol = torch.from_numpy(sol.y[0])
                CE_f = obs[8]
                CF_f = obs[9]                               #convert the np.array into an scalar
                hist[[8,9,10,11]] = 0
                if CE_f >= RII_CEout and CF_f < lim_CF:
                    r += rew_d + 30
                else:
                    obs[[2,6,7,8,9,13,17,19]] = 0
                    hist[[8,9,10,11]] = 0
                    r -= ix * dp_rev[3]
                #obs[8] = obs[8] * vol #the last one becomes the volume times the conc, then we have the kmols  CHECK
            elif obs[13] > 0.5:
                obs[17] = torch.from_numpy(D_p)
                ix = int(4 - (obs[13] / 0.5))
                hist[7 + ix] = torch.from_numpy(D_p)  
                args = torch.cat((hist[8:],obs[19].view(1)))  
                sol = solve_ivp(self.systemRII, t_span, y0r2, args=args, t_eval=[RII_pt - obs[13] + 0.5])
                obs[6] = torch.from_numpy(sol.y[1])   #Cb
                obs[7] = torch.from_numpy(sol.y[2])   #Cd
                obs[8] = torch.from_numpy(sol.y[3])   #Ce
                obs[9] = torch.from_numpy(sol.y[4])   #Cf
                if obs[9] > lim_CF:
                    obs[[2,6,7,8,9,13,17,19]] = 0
                    hist[[8,9,10,11]] = 0
                    r -= ix * dp_rev[3]
                else: r += rew_d
            elif obs[13] < 0:
                r -= 10

        #Separation
        if obs[3] == 1:
            obs[11] -= 0.5      #Value decreased at each time interval
            if obs[11] > 0:
                r += 40
            elif obs[11] == 0:
                obs[14] += obs[20] 
                obs[[3,11,20]] = 0
                r += 40
            #if obs[11] == sep_pt - 1.0:
            #    obs[[6,7,8,9,13,17,19]] = 0

    ############################################## New actions ##########################################         
        #choosing the first reactor
        if unit == 0:
            if obs[0] == 0:
                obs[0] = 1             
                obs[12] = RI_pt 
                obs[15] = torch.from_numpy(hot)
                hist[0] = torch.from_numpy(hot)
                obs[16] = torch.from_numpy(cold)
                hist[4] = torch.from_numpy(cold)
                obs[18] = Ca0
                # Solve the system of differential equations
                y0r1 = [obs[18], 0.0, 294.0, 294.0] 
                sol = solve_ivp(self.systemRI, t_span, y0r1, args=(hist[:8]), t_eval=[0.5])  #t_eval=[RI_pt - obs[12]]    
                obs[4] = torch.from_numpy(sol.y[0])
                obs[5] = torch.from_numpy(sol.y[1])          #compared the expected value with the final concentration and if it is less we give a penalty and throw it away
                r += rew_h + rew_c
            else: r = -10                              # a reward is not provided at this point because we still dont know the outcome #CHANGED += -10
            
        # choosing filter
        if unit == 1:
            if obs[1] == 0:
                if obs[0] == 1 and obs[12] <= 0:      # there is something in the reactor 1 and already done
                    obs[1] = 1
                    obs[10] = filt_pt 
                    #obs[0] = 0
                    obs[[0,4,5,12,15,16,18]] = 0      # take it away from the reactor I
                    r += 10
                else: r = -10  #CHANGED  +=-15
            else: r = -10      #CHANGED   +=-15
                
        #choosing the second reactor
        if unit == 2:
            if obs[2] == 0:
                if obs[1] == 1 and obs[10] <= 0:     #there is something in the filter and already done
                    obs[[1,10]] = 0                  # take it away from the filter
                    obs[2] = 1
                    obs[13] = RII_pt 
                    obs[17] = torch.from_numpy(D_p)
                    hist[8] = torch.from_numpy(D_p)
                    obs[19] = Cd0 
                    args = torch.cat((hist[8:],obs[19].view(1)))
                    # Solve the system of differential equations
                    y0r2 = [Vr,  1.0, obs[19], 0.0, 0.0]
                    sol = solve_ivp(self.systemRII, t_span, y0r2, args=args, t_eval=[0.5])
                    obs[6] = torch.from_numpy(sol.y[1])   #Cb
                    obs[7] = torch.from_numpy(sol.y[2])   #Cd
                    obs[8] = torch.from_numpy(sol.y[3])   #Ce
                    obs[9] = torch.from_numpy(sol.y[4])   #Cf
                    r += rew_d + 5
                    if obs[9] > lim_CF:
                        obs[[2,6,7,8,9,13,17,19]] = 0
                        r += -10
                else: r = -10    # CHANGED += -10
            else: r = -10        # CHANGED += -10

        # choosing the separation unit
        if unit == 3:
            if obs[3] == 0:
                if obs[2] == 1 and obs[13] <= 0:
                    obs[3] = 1
                    obs[11] = sep_pt 
                    obs[20] = obs[8]
                    #obs[2] = 0
                    obs[[2,6,7,8,9,13,17,19]] = 0
                    r += 40
                else: r = -10   # CHANGED += -10
            else: r = -10        # CHANGED += -10

        ################################# Masking ####################################
        obs[21] += 0.5
        if obs[21] == self.max_steps:
            done = True
        
        # Masking
        if obs[21] >= 9:
            available_actions[0] = 0
        if obs[21] >= 11:
            available_actions[1] = 0
        if obs[21] >= 13:
            available_actions[2] = 0
        

        ################################# One-hot encoded ############################
        vector_list = [
            torch.tensor([0, 0, 0, 0]),
            torch.tensor([0, 0, 0, 1]),
            torch.tensor([0, 0, 1, 0]),
            torch.tensor([0, 0, 1, 1]),
            torch.tensor([0, 1, 0, 0]),
            torch.tensor([0, 1, 0, 1]),
            torch.tensor([0, 1, 1, 0]),
            torch.tensor([0, 1, 1, 1]),
            torch.tensor([1, 0, 0, 0]),
            torch.tensor([1, 0, 0, 1]),
            torch.tensor([1, 0, 1, 0]),
            torch.tensor([1, 0, 1, 1]),
            torch.tensor([1, 1, 0, 0]),
            torch.tensor([1, 1, 0, 1]),
            torch.tensor([1, 1, 1, 0]),
            torch.tensor([1, 1, 1, 1])]

        # New vector as tensor
        new_vector = obs[:4]

        # Find the index of new_vector in the list and substitute the last 16 values for the new one hot encode
        index = next((i for i, vec in enumerate(vector_list) if new_vector.equal(vec)), None)
        one_h = torch.zeros(16)
        one_h[index] = 1

        two_h = torch.zeros(3)
        three_h = torch.zeros(3)
        four_h = torch.zeros(3)

        two_h[int(hot)-1] = 1
        three_h[int(cold)-1] = 1
        four_h[int(D_p)-1] = 1

        obs = obs[:22]
        obs = torch.cat((obs,one_h))
        obs = torch.cat((obs,two_h))
        obs = torch.cat((obs,three_h))
        obs = torch.cat((obs,four_h))
        
        ##############################################################################    
        
        return obs, r, done, available_actions, hist
        ##############################################################################        
    def reset(self):
        initial_states = torch.zeros(38, dtype=torch.float32)   #21
        self.done = False     
        return initial_states
    
    def reset_m(self):
        self.available_inactions = np.array([1,0,0,0,0], dtype=np.int32)
        return self.available_inactions
       
    def render (self):
        pass
