import numpy as np
import xarray as xr
import cmath
import importlib
from modeling_cascading_failure import solve

importlib.reload(solve)

nprect = np.vectorize(cmath.rect)
npphase = np.vectorize(cmath.phase)


def simulate_system(
    Y: np.array,
    PV_x: np.array,
    PQ_x: np.array,
    x_slack: np.array,
    V_abs: np.array,
    V_phase: np.array,
    P_input: np.array,
    Q_input: np.array,
    eps: float,
    max_iter: float,
    base_MVA: float,
    line_to_cut: tuple[int, int],
    cut_time: float,
    delta_t: float,
    alpha: float,
    I: np.array,
    gamma: np.array,
    t_max: float,
    include_resistive_losses: bool,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, list]:
    """

    Simulate the failure of a line in a given power system using the swing equation.

    Args:
        Y (np.array): Bus admittance matrix of dimensions NxN, complex valued (note real part will be ignored until resistive losses are added)
        PV_x (np.array): Nx1 vector representing indices of the PV buses (1 if PV bus, 0 o/w)
        PQ_x (np.array): Nx1 vector representing indices of the PQ buses (1 if PQ bus, 0 o/w)
        x_slack (np.array): Nx1 vector representing the index of the slack bus (1 if slack bus, 0 o/w)
        V_abs (np.array): Nx1 vector representing the p.u. voltage magnitudes at the PV buses and slack bus, AND initial guesses for the other voltages
        V_phase (np.array): Nx1 vector representing the voltage phase in radians for the slack bus, AND initial guesses for the other voltages
        P_input (np.array): Nx1 vector representing the real power data for all buses except the reference bus. NOTE: value for reference bus should
            be set to np.nan
        Q_input (np.array): Nx1 vector representing the reactive power data at the PQ buses. NOTE: values for other buses should be set to np.nan
        eps (float): Value of delta P or Q to stop iterating at (for N-R)
        max_iter (float): Maximum iterations (for N-R)
        base_MVA (float): Base MVA
        line_to_cut (tuple[int,int]): Tuple representing which line to cut (ints represent nodes i and j)
        cut_time (float): Time in seconds at which to cut the line
        delta_t (float): Timestep at which to advance the simulation
        alpha (float): Threshold for line shutoff relative to inferred maximum capacity (susceptance times voltages on either side)
        I (np.array): Nx1 vector representing inertia constant at each node
        gamma (np.array): Nx1 vector representing damping coefficient at each node
        t_max (float): Time in seconds at which to stop the simulation
        include_resistive_losses (bool): whether to include resistive losses

    Returns:
        tuple[xr.DataArray,xr.DataArray,xr.DataArray,list]:
            - theta (xr.DataArray): xarray with dimensions Nx(t_max/delta_t) giving evolution of thetas over time
            - omega (xr.DataArray): xarray with dimensions Nx(t_max/delta_t) giving evolution of omegas over time
            - F (xr.DataArray): xarray with dimensions NxNx(t_max/delta_t) giving evolution of line flows over time (in MW)
            - failure_time (list): list with length number_of_cuts where each value is a tuple showing when each line fails: [i,j,cut_time]
            - F_threshold (np.array): np.array with dimensions NxN giving the threshold for line cutting applied in the simulation (in MW)
    """

    # Assert that inputs are the proper size
    assert Y.shape[0] == Y.shape[1]
    N = Y.shape[0]
    correct_shape = (N, 1)
    for v in [PV_x, PQ_x, x_slack, V_abs, V_phase, P_input, Q_input, I, gamma]:
        assert v.shape == correct_shape

    # Solve initial state with Newton-Raphson

    G = np.real(Y)
    B = np.imag(Y)
    if not include_resistive_losses:
        G_multiplier = 1e-6
        Y_for_NR = G * G_multiplier + B * 1j
    else:
        Y_for_NR = Y
    V_abs, theta_0, P, _ = newton_raphson(
        Y_for_NR,
        PV_x,
        PQ_x,
        x_slack,
        V_abs,
        V_phase,
        P_input,
        Q_input,
        eps,
        max_iter,
    )
    K_G = G * (V_abs @ V_abs.T)
    if not include_resistive_losses:
        K_G = K_G * 0
    K_B = B * (V_abs @ V_abs.T)
    omega_0 = np.zeros((N, 1))
    X_t = np.vstack((theta_0, omega_0))

    # Variables to keep track of the evolution of X and F over time
    X = np.copy(X_t)
    F = K_G * np.cos(theta_0.T - theta_0) + K_B * np.sin(theta_0.T - theta_0)
    np.fill_diagonal(F, 0)
    F = F[np.newaxis, ...]

    # Run simulation until t=cut_time
    t = 0
    while t < cut_time:
        X_t = solve.simulate_time_step(X_t, K_G, K_B, P, I, gamma, delta_t)
        F_t = K_G * np.cos(X_t[:N].T - X_t[:N]) + K_B * np.sin(
            X_t[:N].T - X_t[:N]
        )
        np.fill_diagonal(F_t, 0)
        X = np.hstack((X, X_t))
        F = np.concatenate((F, F_t[np.newaxis, ...]), axis=0)
        t += delta_t
    # Cut line
    K_G_cut = np.copy(K_G)
    K_B_cut = np.copy(K_B)
    i = line_to_cut[0]
    j = line_to_cut[1]
    K_G_cut[i, j] = 0
    K_G_cut[j, i] = 0
    K_B_cut[i, j] = 0
    K_B_cut[j, i] = 0
    F_threshold = alpha * np.sqrt(np.square(K_G_cut) + np.square(K_B_cut))
    failure_time = [(cut_time, i, j)]
    frequency_failure = [] #list where forced disconnections due to frequency threshold limit violation are recorded

    # Run simulation with cut line and cut more lines as necessary
    while t < t_max:
        X_t = solve.simulate_time_step(
            X_t, K_G_cut, K_B_cut, P, I, gamma, delta_t
        )
        F_t = K_G_cut * np.cos(X_t[:N].T - X_t[:N]) + K_B_cut * np.sin(
            X_t[:N].T - X_t[:N]
        )
        np.fill_diagonal(F_t, 0)
        X = np.hstack((X, X_t))
        F = np.concatenate((F, F_t[np.newaxis, ...]), axis=0)
        t += delta_t

        # check for threshold exceedance and cut lines if necessary
        threshold_exceeded_mask = np.abs(F_t) > F_threshold
        K_G_cut[threshold_exceeded_mask] = 0
        K_B_cut[threshold_exceeded_mask] = 0

        # check frequency deviation and cut load/generator if necessary
        where_frequency_threshold_exceeded = np.where(X_t[N:] > 0.02*2*np.pi)[0] # nodes where the frequency deviation limit is exceeded
        K_cut[where_frequency_threshold_exceeded] = 0 #disconnect nodes where frequency limit is exceeded
        for i in range(N):
             K_cut[i][where_frequency_threshold_exceeded] = 0 # nodes connected to the shedded load/generator
        
        #record nodes forced to disconnect due to frequency deviation
        if where_frequency_threshold_exceeded.size != 0:      
            # get list of tuples where each tuple has (t,i) for the failed node
            frequency_failure.append([tuple([t] + [i]) for i in where_frequency_threshold_exceeded])
            

        # record lines that were cut in failure_time
        cuts = np.argwhere(threshold_exceeded_mask)
        if cuts.size != 0:
            # get list of tuples where each tuple has (t,i,j) for the cut (without duplicates)
            cuts_list = list({tuple([t] + sorted(cut)) for cut in cuts})
            failure_time += cuts_list

    # Prepare outputs
    T = np.arange(0, t_max + delta_t * 2, delta_t)
    i = np.arange(N)
    j = np.arange(N)
    theta = xr.DataArray(
        data=X[:N, :], dims=["node", "time"], coords=dict(node=i, time=T)
    )
    omega = xr.DataArray(
        data=X[N:, :], dims=["node", "time"], coords=dict(node=i, time=T)
    )
    flows = xr.DataArray(
        data=F * base_MVA,
        dims=["time", "node_i", "node_j"],
        coords=dict(time=T, node_i=i, node_j=j),
    )

    return theta, omega, flows, failure_time,frequency_failure, F_threshold * base_MVA


def newton_raphson(
    Y: np.array,
    PV_x: np.array,
    PQ_x: np.array,
    x_slack: np.array,
    V_abs: np.array,
    V_phase: np.array,
    P_input: np.array,
    Q_input: np.array,
    eps: float,
    max_iter: float,
) -> tuple[np.array, np.array, np.array, np.array]:
    """Function to solve power flow equations for a system of N buses using the Newton-Raphson method.
    N is implicit in the inputs.

    Args:
        Y (np.array): Bus admittance matrix of dimensions NxN, complex valued (note real part will be ignored until resistive losses are added)
        PV_x (np.array): Nx1 vector representing indices of the PV buses (1 if PV bus, 0 o/w)
        PQ_x (np.array): Nx1 vector representing indices of the PQ buses (1 if PQ bus, 0 o/w)
        x_slack (np.array): Nx1 vector representing the index of the slack bus (1 if slack bus, 0 o/w)
        V_abs (np.array): Nx1 vector representing the p.u. voltage magnitudes at the PV buses and slack bus, AND initial guesses for the other voltages
        V_phase (np.array): Nx1 vector representing the voltage phase in radians for the slack bus, AND initial guesses for the other voltages
        P_input (np.array): Nx1 vector representing the real power data for all buses except the reference bus. NOTE: value for reference bus should
            be set to np.nan
        Q_input (np.array): Nx1 vector representing the reactive power data at the PQ buses. NOTE: values for other buses should be set to np.nan
        eps (float): Value of delta P or Q to stop iterating at (for N-R)
        max_iter (float): Maximum iterations (for N-R)

    Returns:
        tuple[np.array, np.array, np.array, np.array]:
            - V_abs (np.array): dimensions Nx1, represents the p.u. voltage magnitudes at each bus (including
                the buses where voltage was specified by the problem)
            - V_phase (np.array): dimensions Nx1, represents the voltage phases in radians at each bus
            - P (np.array): dimensions Nx1, represents the real power at each bus (including where specified
                by the inputs)
            - Q (np.array): dimensions Nx1, represents the reactive power at each bus (including where
                specified by the inputs)
    """

    # assert Y is square
    assert Y.shape[0] == Y.shape[1]

    # get number of buses
    N = Y.shape[0]
    slack_index = np.where(x_slack == 1)[0][0]
    not_pq_indices = np.where(PQ_x == 0)[0]

    # initialize voltages and angles
    voltages = np.copy(V_abs)
    angles = np.copy(V_phase)

    # get G and B from Y
    G = np.real(Y)
    B = np.imag(Y)

    iterations = 0
    epsilon = 1
    while epsilon > eps:

        P = np.zeros((N, 1))
        Q = np.zeros((N, 1))

        for i in range(N):
            for j in range(N):
                P[i] += (
                    voltages[i]
                    * voltages[j]
                    * (
                        G[i, j] * np.cos(angles[i] - angles[j])
                        + B[i, j] * np.sin(angles[i] - angles[j])
                    )
                )
                Q[i] += (
                    voltages[i]
                    * voltages[j]
                    * (
                        G[i, j] * np.sin(angles[i] - angles[j])
                        - B[i, j] * np.cos(angles[i] - angles[j])
                    )
                )

        # calculate dP and dQ
        dP_all = P_input - P
        dQ_all = Q_input - Q
        dP = dP_all[x_slack == 0].reshape(-1, 1)
        dQ = dQ_all[PQ_x == 1].reshape(-1, 1)
        dp_dq = np.vstack((dP, dQ))

        # calculate Jacobian

        # calculate H = dP/dtheta
        # will calculate for all nodes, then take out the row and column for slack bus so that
        # H will have dimension (N-1,N-1)
        H = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if i == j:
                    H[i, j] = -Q[i] - B[i, j] * voltages[i] ** 2
                else:
                    H[i, j] = (
                        voltages[i]
                        * voltages[j]
                        * (
                            G[i, j] * np.sin(angles[i] - angles[j])
                            - B[i, j] * np.cos(angles[i] - angles[j])
                        )
                    )

        # calculate M = dQ/dtheta
        # similarly, will calculate for all nodes, but will end up with dimensions of (num_pq_buses, N-1)
        M = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if i == j:
                    M[i, j] = P[i] - G[i, j] * voltages[i] ** 2
                else:
                    M[i, j] = (
                        -voltages[i]
                        * voltages[j]
                        * (
                            G[i, j] * np.cos(angles[i] - angles[j])
                            + B[i, j] * np.sin(angles[i] - angles[j])
                        )
                    )

        # calculate N = dP/dV
        # will end up with dimensions of (N-1,num_pq_buses)
        N_mat = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if i == j:
                    N_mat[i, j] = P[i] + G[i, j] * voltages[i] ** 2
                else:
                    N_mat[i, j] = -M[i, j]

        # calculate L = dQ/dV
        # will end up with dimensions of (num_pq_buses,num_pq_buses)
        L = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if i == j:
                    L[i, j] = Q[i] - B[i, j] * voltages[i] ** 2
                else:
                    L[i, j] = H[i, j]

        H = np.delete(H, [slack_index], axis=0)
        H = np.delete(H, [slack_index], axis=1)
        M = np.delete(M, not_pq_indices, axis=0)
        M = np.delete(M, [slack_index], axis=1)
        N_mat = np.delete(N_mat, [slack_index], axis=0)
        N_mat = np.delete(N_mat, not_pq_indices, axis=1)
        L = np.delete(L, not_pq_indices, axis=0)
        L = np.delete(L, not_pq_indices, axis=1)

        J = np.vstack((np.hstack((H, N_mat)), np.hstack((M, L))))

        # solve for dtheta and dv
        X = np.linalg.solve(J, dp_dq)
        X = X.ravel()
        dtheta = X[0 : N - 1]
        dV = X[N - 1 :]

        # update angles and voltages
        angles[x_slack == 0] += dtheta
        voltages[PQ_x == 1] += dV

        # update iteration variables
        epsilon = max(abs(dp_dq))
        if iterations >= max_iter:
            break
        iterations += 1

    # calculate P and Q at remaining nodes

    # get vector voltages
    V = nprect(voltages, angles)

    # get current at each node
    I = Y @ V

    # get power at each node
    P = np.copy(P_input)
    Q = np.copy(Q_input)
    Sn = np.multiply(V, np.conj(I))

    P[x_slack == 1] = np.real(Sn)[x_slack == 1]
    Q[PQ_x == 0] = np.imag(Sn)[PQ_x == 0]

    V_abs = np.abs(V)
    V_phase = npphase(V)

    return V_abs, V_phase, P, Q
