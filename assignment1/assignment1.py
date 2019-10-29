from matplotlib import pyplot as plt
from gridWorld import gridWorld
import numpy as np

def show_value_function(mdp, V):
    fig = mdp.render(show_state = False, show_reward = False)            
    for k in mdp.states():
        s = k if isinstance(k, tuple) else mdp.legal_states[k]
        fig.axes[0].annotate("{0:.3f}".format(V[k]), (s[1] - 0.1, s[0] + 0.1), size = 40/mdp.board_mask.shape[0])
    plt.show()
    
def show_policy(mdp, PI):
    fig = mdp.render(show_state = False, show_reward = False)
    action_map = {"U": "↑", "D": "↓", "L": "←", "R": "→"}
    for k in mdp.states():
        s = k if isinstance(k, tuple) else mdp.legal_states[k]
        if mdp.terminal[s] == 0:
            fig.axes[0].annotate(action_map[PI[k]], (s[1] - 0.1, s[0] + 0.1), size = 100/mdp.board_mask.shape[0])
    plt.show()
    
####################  Problem 1: Value Iteration #################### 

def value_iteration(mdp, gamma, theta = 1e-3):
    # Make a valuefunction, initialized to 0
    V = np.zeros((len(mdp.states())))
    
    """
    YOUR CODE HERE:
    Problem 1a) Implement Value Iteration
    
    Input arguments:
        - mdp     Is the markov decision process, it has some usefull functions given below
        - gamma   Is the discount rate
        - theta   Is a small threshold for determining accuracy of estimation
    
    Some usefull functions of the grid world mdp:
        - mdp.states() returns a list of all states [0, 1, 2, ...]
        - mdp.actions(state) returns list of actions ["U", "D", "L", "R"] if state non-terminal, [] if terminal
        - mdp.transition_probability(s, a, s_next) returns the probability p(s_next | s, a)
        - mdp.reward(state) returns the reward of the state R(s)
    """

    #V[s] = max([sum(mdp.transition_probability(s, a, s_next) * (mdp.reward(s_next) + gamma * V[s_next]) for s_next in range(2)) for a in mdp.actions(s)])

    i = 0
    error = 0
    while True:
        i += 1
        print("Iteration", i)
        delta = 0
        V_prev = np.copy(V)
        for s in mdp.states():
            v = V[s]
            if len(mdp.actions(s)) == 0: # terminal state
                V[s] = mdp.reward(s)
            else:
                V[s] = max([sum(mdp.transition_probability(s, a, s_next) * (mdp.reward(s) + gamma * V[s_next]) for s_next in mdp.states()) for a in mdp.actions(s)])
                error = max(error, abs(V[s] - V_prev[s])) # for task 2d)
            delta = max(delta, abs(v - V[s]))


        if delta < theta:
            break

    print("Biggest error:", error) # used for task 2d)
    return V

def policy(mdp, V):
    # Initialize the policy list of correct length
    PI = np.random.choice(env.actions(), len(mdp.states()))
    
    """
    YOUR CODE HERE:
    Problem 1b) Implement Policy function 
    
    Input arguments:
        - mdp Is the markov decision problem
        - V   Is the optimal value function, found with value iteration
    """
    
    A = ["U", "D", "L", "R"]
    for s in mdp.states():
        if len(mdp.actions(s)) == 0: # terminal state
            continue
        PI[s] = A[np.argmax([sum(mdp.transition_probability(s, a, s_next) * (mdp.reward(s) + gamma * V[s_next]) for s_next in mdp.states()) for a in mdp.actions(s)])]

    return PI

####################  Problem 2: Policy Iteration #################### 
def policy_evaluation(mdp, gamma, PI, V, theta = 1e-3):   
    """
    YOUR CODE HERE:
    Problem 2a) Implement Policy Evaluation
    
    Input arguments:  
        - mdp   Is the markov decision problem
        - gamma Is discount factor
        - PI    Is current policy
        - V     Is preveous value function guess
        - theta Is small threshold for determining accuracy of estimation
        
    Some useful tips:
        - If you decide to do exact policy evaluation, np.linalg.solve(A, b) can be used
          optionally scipy has a sparse linear solver that can be used
        - If you decide to do exact policy evaluation, note that the b vector simplifies
          since the reward R(s', s, a) is only dependant on the current state s, giving the 
          simplified reward R(s) 
    """
    #V = np.zeros((len(mdp.states()))) # better to use the previous estimate
    error = 0
    while True:
        delta = 0
        V_prev = np.copy(V)
        for s in mdp.states():
            v = V[s]
            if len(mdp.actions(s)) == 0: # terminal state
                V[s] = mdp.reward(s)
            else:
                V[s] = sum(mdp.transition_probability(s, PI[s], s_next) * (mdp.reward(s) + gamma * V[s_next]) for s_next in mdp.states())
                error = max(error, abs(V[s] - V_prev[s])) # used for task 2d)

            delta = max(delta, abs(v - V[s]))

        if delta < theta:
            break

    return V, error # error used for task 2d)

def policy_iteration(mdp, gamma):
    # Make a valuefunction, initialized to 0
    V = np.zeros((len(mdp.states())))
    
    # Create an arbitrary policy PI
    PI = np.random.choice(env.actions(), len(mdp.states()))
    
    """
    YOUR CODE HERE:
    Problem 2b) Implement Policy Iteration
    
    Input arguments:  
        - mdp   Is the markov decision problem
        - gamma Is discount factor

    Some useful tips:
        - Use the the policy_evaluation function from the preveous subproblem
    """

    A = ["U", "D", "L", "R"]
    i = 0
    error = 0 # for task 2d)
    while True:
        i += 1
        print("Iteration", i)
        PI_old = np.copy(PI)
        V, e = policy_evaluation(mdp, gamma, PI, V) # e used for task 2d)
        error = max(error, e) # used for task 2d)

        for s in mdp.states():
            if len(mdp.actions(s)) == 0:
                continue
            PI[s] = A[np.argmax([sum(mdp.transition_probability(s, a, s_next) * (mdp.reward(s) + gamma * V[s_next]) for s_next in mdp.states()) for a in mdp.actions(s)])]

        if np.array_equal(PI_old, PI):
            break
    
    print("Biggest error:", error)
    return PI, V

if __name__ == "__main__":
    """
    Change the parameters below to change the behaveour, and map of the gridworld.
    gamma is the discount rate, while filename is the path to gridworld map. Note that
    this code has been written for python 3.x, and requiers the numpy and matplotlib
    packages

    Available maps are:
        - gridworlds/tiny.json
        - gridworlds/large.json
    """
    gamma   = 0.9
    filname = "gridworlds/tiny.json"


    # Import the environment from file
    env = gridWorld(filname)

    # Render image
    fig = env.render(show_state = False)
    plt.show()
    
    
    print("Runs value iteration")
    # Run Value Iteration and render value function and policy
    V = value_iteration(mdp = env, gamma = gamma)
    show_value_function(env, V)
    
    print("Runs policy")
    PI = policy(env, V)
    show_policy(env, PI)
    
    print("Runs policy iteration")
    # Run Policy Iteration and render value function and policy
    PI, V = policy_iteration(mdp = env, gamma = gamma)
    show_value_function(env, V)
    show_policy(env, PI)
    