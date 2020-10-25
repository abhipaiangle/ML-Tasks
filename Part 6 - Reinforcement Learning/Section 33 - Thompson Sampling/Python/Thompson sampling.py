import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv("Ads_CTR_Optimisation.csv")
import random
N=10000
d=10
ads_selected=[]
n_selected_1=[0]*d
n_selected_0=[0]*d
total_reward=0
for n in range(0,N): 
    ad=0
    max_beta=0
    for i in range(0,d):
        random_beta=random.betavariate(n_selected_1[i]+1,n_selected_0[i]+1)
        
        if(random_beta>max_beta):
            max_beta=random_beta
            ad=i
    ads_selected.append(ad)
    reward=dataset.values[n,ad]
    if(reward==1):
        n_selected_1[ad]=n_selected_1[ad]+1
    else:
        n_selected_0[ad]=n_selected_0[ad]+1
    total_reward=total_reward+reward     
    
plt.hist(ads_selected)
plt.title("Favoured Advertisement")
plt.show()  