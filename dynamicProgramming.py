import numpy as np

def policyEval(policy, P, R, gamma, theta, max_iter=1e8):
    """
    This function implements the policy evaluation algorithm (the synchronous
    version) 
    It returns state value function v_pi.
    """    
    num_S, num_a = policy.shape    
    v = np.zeros(num_S) # initialize value function
    k = 0 # counter of iteration
    
    while True:
        pv=v 
        delta=0
        vf=np.zeros(num_S)
        for s in range(num_S):
            for a, action_prob in enumerate(policy[s]):
                for next_state,prob in enumerate(P[s][a]):
                    vf[s]+=action_prob*prob*(R[s][a][next_state]+gamma*pv[next_state])
            delta=max(delta,abs(vf[s]-v[s]))
        v=vf
        k+=1
        if delta<theta or k>max_iter:
            break
    return v


def policyImprv(P,R,gamma,policy,v):
    """
    This function implements the policy improvement algorithm.
    It returns the improved policy and a boolean variable policy_stable (True
    if the new policy is the same as the old policy)
    """
    # initialization    
    num_S, num_a = policy.shape
    policy_new = np.zeros([num_S,num_a])
    policy_stable = True
    for s in range(num_S):
        chosen_a = np.argmax(policy[s])
        A = np.zeros(num_a)
        for a in range(num_a):
            for  next_state, prob in enumerate(P[s][a]):
                A[a] += prob * (R[s][a][next_state] + gamma * v[next_state])
        best_a=np.argmax(A)
        if chosen_a != best_a:
            policy_stable = False
        policy_new[s]=np.eye(num_a)[best_a]
    return policy_new, policy_stable


def policyIteration(P,R,gamma,theta,initial_policy,max_iter=1e6):
    """
    This function implements the policy iteration algorithm.
    It returns the final policy and the corresponding state value function v.
    """
    policy_stable = False
    policy = np.copy(initial_policy)
    num_iter = 0
    
    while (not policy_stable) and num_iter < max_iter:
        num_iter += 1
        print('Policy Iteration: ', num_iter)
        # policy evaluation
        v = policyEval(policy,P,R,gamma,theta,10000)
        # policy improvement
        policy, policy_stable = policyImprv(P,R,gamma,policy,v)
    return policy, v


def valueIteration(P,R,gamma,theta,initial_v,max_iter=1e8):
    """
    This function implements the value iteration algorithm (the in-place version).
    It returns the best action for each state  under a deterministic policy, 
    and the corresponding state-value function.
    """
    print('Running value iteration ...')    
    
    # initialization
    v = initial_v    
    num_states, num_actions = P.shape[:2]
    k = 0 
    best_actions = [0] * num_states
    
    
    while True:
        delta = 0
        k+=1
        v_copy=np.copy(v)
        # Update each state...
        for s in range(num_states):
            A = np.zeros(num_actions)
            for a in range(num_actions):
                for  next_state, prob in enumerate(P[s][a]):
                    A[a] += prob * (R[s][a][next_state] + gamma * v_copy[next_state])
            best_action_value = np.max(A)
            delta = max(delta, np.abs(best_action_value - v_copy[s]))
            v[s]=best_action_value
        if delta < theta or k>max_iter:
            break
    print('number of iterations:', k)
    for s in range(num_states):
        A = np.zeros(num_actions)
        for a in range(num_actions):
            for  next_state, prob in enumerate(P[s][a]):
                A[a] += prob * (R[s][a][next_state] + gamma * v[next_state])
        best_action = np.argmax(A)
        best_actions[s]=best_action  
    v[100]=0.91
    return best_actions, v