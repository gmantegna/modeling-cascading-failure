import numpy as np
from scipy import integrate


def swing_equation_ODE(X_t, t, K, P, I, gamma):
    """
    >The function takes as input:

    X_t: np.array of dimension 2Nx1, giving thetas and omegas at time t, it is of the form (theta_0,omega0,...,theta_N,omega_N)
    t: the time at which the integraion starts
    K: np.array of dimension NxN, representing the coupling strength between nodes
    P: np.array of dimension Nx1, giving the power at the nodes of the system
    I: np.array of dimension Nx1, giving the inertia constants at the nodes of the system
    gamma: np.array of dimension Nx1, giving the damping coefficients at the nodes of the system
    delta_t: float, representing timestep at which to advance the simulation

    >The function returns the expression of the swing equation ODE
    """
    N = len(P)
    dX_dt = np.zeros(2 * N)
    for i in range(2 * N):
        if i % 2 == 0:
            dX_dt[i] = X_t[i + 1]
        else:
            dX_dt[i] = (1 / I[i // 2]) * (
                P[i // 2]
                - gamma[i // 2] * X_t[i]
                + sum(
                    [
                        K[i // 2][j] * np.sin(X_t[2 * j] - X_t[i - 1])
                        for j in range(N)
                    ]
                )
            )
    return dX_dt


def simulate_time_step(X_t, K, P, I, gamma, delta_t):
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
    # Obtaining the expression of the swing equation ODE
    X_t_plus_1 = integrate.odeint(
        swing_equation_ODE,
        X_t.flatten(),
        [0, delta_t],
        (K, P, I, gamma),
        full_output=True,
    )[0]
    return X_t_plus_1
