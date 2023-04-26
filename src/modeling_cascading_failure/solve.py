import numpy as np
from scipy.integrate import solve_ivp


def swing_equation_ODE(t, X_t, K_G, K_B, P, I, gamma):
    """
    >The function takes as input:

    X_t: np.array of dimension 2Nx1, giving thetas and omegas at time t, it is of the form (theta_0,...,theta_N,omega_1,..omega_N)
    t: the time at which the integraion starts
    K: np.array of dimension NxN, representing the coupling strength between nodes
    P: np.array of dimension Nx1, giving the real power at the nodes of the system
    I: np.array of dimension Nx1, giving the inertia constants at the nodes of the system
    gamma: np.array of dimension Nx1, giving the damping coefficients at the nodes of the system

    >The function returns the expression of the swing equation ODE
    """
    N = len(P)
    dX_dt = np.zeros(2 * N)
    for i in range(N):
        # Computing the derivative of theta_i
        dX_dt[i] = X_t[N + i]
        # Computing the derivative of omega_i
        dX_dt[N + i] = (1 / I[i]) * (
            P[i]
            - gamma[i] * X_t[N + i]
            + sum([K_G[i][j] * np.cos(X_t[j] - X_t[i]) for j in range(N)])
            + sum([K_B[i][j] * np.sin(X_t[j] - X_t[i]) for j in range(N)])
        )

    return dX_dt


def simulate_time_step(X_t, K_G, K_B, P, I, gamma, delta_t):
    """
    >The function takes as input:

    X_t: np.array of dimension 2Nx1 giving thetas and omegas at time t, it is of the form (theta_0,...,theta_N,omega_1,..omega_N)
    K: np.array of dimension NxN representing the coupling strength between nodes
    P: np.array of dimension Nx1 giving the power at the nodes of the system
    I: np.array of dimension Nx1 giving the inertia constants at the nodes of the system
    gamma: np.array of dimension Nx1 giving the damping coefficients at the nodes of the system
    delta_t: float giving the timestep

    >The function returns:
    X_t_plus_1 : np.array of dimension 2N+1 giving thetas and omegas at time t+1
    """
    # Integrating the initial value problem from t to t+delta_t
    integration = solve_ivp(
        swing_equation_ODE,
        (0.0, delta_t),
        X_t.flatten(),
        args=(K_G, K_B, P, I, gamma),
        t_eval=[0, delta_t],
    )
    # Extracting the state at t+delta_t:
    X_t_plus_1 = integration.y.T[-1]
    return X_t_plus_1.reshape(-1, 1)
