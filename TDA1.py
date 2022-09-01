import csv

import gym
import numpy as np
import pandas as pd
from gym.envs.box2d import CarRacing
from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
import gudhi as gd
import gudhi.representations
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

if __name__=='__main__':


    #env = getattr(environments, env)
    #env = DummyVecEnv([env])
    import matplotlib.pyplot as plt
    import skimage.transform
    #y='/Users/paulinanouwou/Downloads/marl_ppo-main/cartra.png'
    #y = cv.imread("cartra.png")
    #plt.imshow(y)

    #h, w, c = y.shape
    #x = skimage.transform.resize(y, (256, int((w * 256) / h)), preserve_range=False)
    #plt.imshow(x)


    def _is_outside(self):
        right = self.info['count_right'] > 0
        left  = self.info['count_left']  > 0
        if (left|right).sum() == 0:
            return True
        else:
            return False

    def check_outside(self,reward,done):
        if self._is_outside():
            # In case it is outside the track
            return True
        else:
            return False


    env = make_vec_env("CarRacing-v0", n_envs=1)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=1000)
    model.save("ppo_car")

    del model  # remove to demonstrate saving and loading

    model = PPO.load("ppo_car")
    num_pts = 1000
    run0=0
    if run0==0:


        num_pts = 1000

        X = np.empty([num_pts, 2])

        path = '/Users/paulinanouwou/Downloads/'
        for i in range(num_pts):
            dones2 = False
            env2 = CarRacing()
            obs2 = env2.reset()
            obs = env.reset()
            buffer = []
            label=[]
            ep_obs=[]
            ep_pos=[]
            ep_act=[]
            ep_cumR=0
            functional_Fault=False
            iter=0
            t=0
            while not dones2 :
                iter+=1
                obs2 = obs2[np.newaxis]
                #action, _states = model.predict(obs)
                action2, _states2 = model.predict(obs2.copy())
                x,y=env2.car.hull.position
                ep_pos.append([x,y])
                ep_act.append(action2)
                ep_obs.append(obs2)


                #print(iter, functional_Fault, dones2, env2.tile_visited_count, t, [x, y], obs2.shape)
                from skimage.measure.entropy import shannon_entropy
                file = open(f'{path}/carRacing.csv', 'a+', newline='')
                with file:
                    write = csv.writer(file)
                    write.writerow([x, y,obs2,obs2.shape,shannon_entropy(np.squeeze(obs2)[:, :, 0]),shannon_entropy(np.squeeze(obs2)[:, :, 1]),
                      shannon_entropy(np.squeeze(obs2)[:, :, 2]),t,functional_Fault])  # each element is the probability of taking each action of the trained agent
                   # write.writerow(obs2)  # each element is the total visited state of the maze
                   # write.writerow(t)
                   # write.writerow(functional_Fault)
                if functional_Fault:
                    break
                #obs, rewards, dones, info = env.step(action)

                obs2, rewards2, dones2, info2 = env2.step(np.squeeze(action2,axis=0))

                ep_cumR+=rewards2

                #if abs(x) > 2000/6.0 or abs(y) > 2000/6.0 :
                 #   functional_Fault=True
                if functional_Fault:
                    label.append(1)
                t+=1
                if env2.tile_visited_count < t:
                    functional_Fault = True
                if(env2.tile_visited_count<len(env2.track)):
                    if env2.tile_visited_count<t:
                        functional_Fault = True
                if iter==200:
                    break
