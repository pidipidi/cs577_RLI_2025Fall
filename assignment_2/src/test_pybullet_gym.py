#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import time
import gymnasium as gym
import logging

#gym.logger.set_level(logging.ERROR) 
import numpy as np
from envs.kukaGymEnv import KukaGymEnv

# Load the KUKA Robotic Grasping Environment
env = KukaGymEnv(renders=True, isDiscrete=True)
_ = env.render(mode='human')
env.isRender=True
s = env.reset()
cnt,rsum = 0,0

# Run random movements
while 1:
    cnt += 1 # increase counter
    
    a = env.action_space.sample()
    obs, r, done, _, _ = env.step(action=a)
    ## print (a, obs)
            
    rsum += r
    still_open = env.render(mode='human')
    if done:
        env.reset()
        
    time.sleep(1./240.)
    #time.sleep(1./10.)
p.disconnect()
env.close()
print ("Done. ravg:[%.3f]"%(rsum/cnt))

