import numpy as np
from dynamicProgramming import *
import matplotlib.pyplot as plt

# Set up P and R
def get_P_and_R_gambler(prob_head):
    # initialization    
    P = np.zeros([101,100,101])
    R = np.zeros([101,100,101])
    for s in range(1,101):
        for a in range(1,min(s, 100-s)+1):
            if s+a <= 100:
                P[s][a][s+a]=prob_head
            if s-a >= 0:
                P[s][a][s-a]=1-prob_head 
    for s in range(101):
        for a in range(100):
            R[s][a][100]=1
    return P,R    

# initial value function
initial_v = np.zeros(101)
initial_v[0] = 0
initial_v[-1]=1



gamma = 1
theta = 1e-6
ph = 0.40

P, R = get_P_and_R_gambler(ph) 

opt_actions,opt_v = valueIteration(P,R,gamma,theta,initial_v,1000)

plt.figure(figsize=(8,12))
plt.subplot(211)
plt.plot(opt_actions,'s')
plt.xlabel('capital')
plt.ylabel('stake')
plt.title('p_h = '+str(ph))
plt.grid()
plt.subplot(212)
plt.plot(opt_v,'.-')
plt.xlabel('capital')
plt.ylabel('value')
plt.grid()
plt.savefig('gambler_example_p=' + str(ph) +'.pdf',dpi=180)

#Upon checking the value functions are all correct except for state=100.

#for Ph=0.25 i.e <0.5 and gamma=1, the the algorithm doesn't really mind taking its time to win, 
#and so it tries to maximize value at every step:
#So it tries to play it safe when is in a state >50, which accounts for the submitted shape.


#for Ph=0.55, for states <50, the algorithm finds it difficult to get an action which increases 
#the winning chance. Then at 50, it is able to bet a lot as there is win-win situation and after that it 
#slowly decreases.
