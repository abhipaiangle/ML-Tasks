# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 21:09:59 2020

@author: Abhishek Pai Angle
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv("Ads_CTR_Optimisation.csv")
import math
N=10000
d=10
ads_selected=[]
n_selected=[0]*d
sum_of_rewards=[0]*d
total_reward=0
for n in range(0,N): 
    ad=0
    max_bound=0
    for i in range(0,d):
        if(n_selected[i]>0):
          avg=sum_of_rewards[i]/n_selected[i]
          delta=math.sqrt(3/2*math.log(n+1)/n_selected[i])
          upper_bound=avg+delta
        else:
            upper_bound=1e400
        if(upper_bound>max_bound):
            max_bound=upper_bound
            ad=i
    ads_selected.append(ad)
    n_selected[ad]=n_selected[ad]+1
    reward=dataset.values[n,ad]
    sum_of_rewards[ad]=sum_of_rewards[ad]+reward
    total_reward=total_reward+reward     
    
plt.hist(ads_selected)
plt.Title("Favoured Advertisement")
plt.show()    