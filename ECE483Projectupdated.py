#!/usr/bin/env python
# coding: utf-8

# In[125]:


import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# In[126]:


X_t_test = np.array([[0,0,0.321,0]]).T
X = X_t_test.flatten()
print(X.shape)


# In[183]:


def swing_equation_ODE(t,X_t,K,P,I,gamma):
    """
    >The function takes as input:
    
    X_t: np.array of dimension 2Nx1, giving thetas and omegas at time t, it is of the form (theta_0,...,theta_N,omega_1,..omega_N)
    t: the time at which the integraion starts
    K: np.array of dimension NxN, representing the coupling strength between nodes
    P: np.array of dimension Nx1, giving the power at the nodes of the system
    I: np.array of dimension Nx1, giving the inertia constants at the nodes of the system
    gamma: np.array of dimension Nx1, giving the damping coefficients at the nodes of the system
    
    >The function returns the expression of the swing equation ODE
    """
    N = len(P)
    dX_dt = np.zeros(2*N)
    for i in range(N):
        #Computing the derivative of theta_i
            dX_dt[i] = X_t[N+i]
        # Computing the derivative of omega_i
            dX_dt[N+i] = (1/I[i])*(P[i] - gamma[i]*X_t[N+i]
                       + sum([K[i][j]*np.sin(X_t[j]-X_t[i]) for j in range(N)]))

    return dX_dt

def simulate_time_step(X_t,K,P,I,gamma,delta_t):
    """
    >The function takes as input:
    
    X_t: np.array of dimension 2Nx1 giving thetas and omegas at time t, it is of the form (theta_0,omega0,...,theta_N,omega_N)
    K: np.array of dimension NxN representing the coupling strength between nodes
    P: np.array of dimension Nx1 giving the power at the nodes of the system
    I: np.array of dimension Nx1 giving the inertia constants at the nodes of the system
    gamma: np.array of dimension Nx1 giving the damping coefficients at the nodes of the system
    
    >The function returns:
    X_t_plus_1 : np.array of dimension 2N+1 giving thetas and omegas at time t+1
    """
    # Integrating the initial value problem from t to t+delta_t
    integration = solve_ivp(swing_equation_ODE, (0.0,delta_t), X_t.flatten(),args =(K,P,I,gamma),t_eval= [0,delta_t_test])
    #Extracting the state at t+delta_t:
    X_t_plus_1 = integration.y.T[-1]
    return X_t_plus_1

#Test case inputs
#Input variables vector (theta_1,omega_1,theta_2,omega_2)
X_t_test = np.array([[0,-0.32128859,0,0]]).T 
#Constructing the coupling matrix K
B_test =  np.array([[-5,5],[5,-5]]) # Admittance matrix
V_test = np.array([[1,0.95]]) # Voltage amplitudes vector
K_test = B_test*(V_test.T@V_test) # Coupling matrix
#Real power vector
P_test = np.array([[1.5 ,-1.5]]).T 
#Moment of inertia vector
I_test = np.array([[1,1]]).T 
#Decay constants vector
gamma_test = np.array([[1,1]]).T 
#step_size
delta_t_test = float(0.1)
def test_case_benchmark():
    """ The function returns a benchmark value to asses the simulate_time_step function based on the bus system
    in question 2 of ECE483 homework 2 """
    benchmark =  simulate_time_step(X_t_test,K_test,P_test,I_test,gamma_test,delta_t_test)
    return benchmark
def test_simulate_time_step():
    """
    We run the current version of the code with the inputs corresponding to the bus-system in question 2 of homework 2
    and assert the value using this example.
    """
    benchmark = test_case_benchmark()
    assert simulate_time_step(X_t_test,K_test,P_test,I_test,gamma_test,delta_t_test).any() == benchmark.any()
    


# In[ ]:





# In[184]:


X_t_plus_1 = simulate_time_step(X_t_test,K_test,P_test,I_test,gamma_test,delta_t_test)
print('X_t_plus_1 is',X_t_plus_1)


# In[185]:


test_case_benchmark()
# The simulation doesn't change anything as we expect from a static system ! 


# In[186]:


test_simulate_time_step()


# In[ ]:




