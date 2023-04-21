import numpy as np
from src.modeling_cascading_failure import solve


def test_simulate_time_step():
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

    assert (
        solve.simulate_time_step(
            X_t_test, K_test, P_test, I_test, gamma_test, delta_t_test
        )[0].any()
        == solve.simulate_time_step(
            X_t_test, K_test, P_test, I_test, gamma_test, delta_t_test
        )[0].any()
    )
