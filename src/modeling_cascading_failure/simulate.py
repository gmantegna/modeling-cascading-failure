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
    lines_to_cut: list,
    nodes_to_cut: list,
    cut_time: float,
    delta_t: float,
    alpha: float,
    frequency_deviation_threshold: float,
    apply_freq_dev_during_sim: bool,
    I: np.array,
    H: float,
    gamma: np.array,
    t_max: float,
    include_resistive_losses: bool,
    ref_freq: float,
) -> tuple[
    xr.DataArray,
    xr.DataArray,
    xr.DataArray,
    xr.DataArray,
    list,
    list,
    np.array,
    list,
    xr.DataArray,
]:
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
        lines_to_cut (list or None): list of tuple[int,int]'s representing which line to cut (ints represent nodes i and j)
            - Set to None if no lines to cut
        nodes_to_cut (list or None): List of ints representing which nodes to cut (ints represent node indices)
            - Set to None if no nodes to cut
        cut_time (float): Time in seconds at which to cut the line
        delta_t (float): Timestep at which to advance the simulation
        alpha (float or None): Threshold for line shutoff relative to inferred maximum capacity (susceptance times voltages on either side)
            - Set to None if no threshold
        frequency_deviation_threshold (float): Threshold (in Hz) for frequency deviations relative to the reference frequency
        apply_freq_dev_during_sim (bool): Whether to apply frequency deviation cutoffs during the simulation
        I (np.array): Nx1 vector representing inertia constant at each node. Set to None if using H instead
        H (float): float representing the inertia constant (function will calculate I based on generator powers). Set to None if using I instead
        gamma (np.array): Nx1 vector representing damping coefficient at each node
        t_max (float): Time in seconds at which to stop the simulation
        include_resistive_losses (bool): whether to include resistive losses
        ref_freq (float): reference frequency in Hz

    Returns:
        tuple[xr.DataArray,xr.DataArray,xr.DataArray,list]:
            - theta (xr.DataArray): xarray with dimensions Nx(t_max/delta_t) giving evolution of thetas over time
            - omega (xr.DataArray): xarray with dimensions Nx(t_max/delta_t) giving evolution of omegas over time
            - F (xr.DataArray): xarray with dimensions NxNx(t_max/delta_t) giving evolution of line flows over time (in MW)
            - P (xr.DataArray): xarray with dimensions Nx(t_max/delta_t) giving evolution of node powers over time (in MW)
            - line_failures (list or None): list with length number_of_cuts where each value is a tuple showing when each line fails: [cut_time,i,j]
            - node_failures (list): list with length number_of_node_failures where each value is a tuple showing when each node fails: [cut_time,i]
                - If frequency deviation threshold is not being applied udring simulation, cut_time will be the final time
            - F_threshold (np.array): np.array with dimensions NxN giving the threshold for line cutting applied in the simulation (in MW)
            - line_failures_static (list or None): list with length number_of_cuts where each value is a tuple showing when each line fails
                in the static case: [iteration n,i,j]
            - flows_static (xr.DataArray): xarray with dimensions Nx(number_of_static_NR_iterations) giving evolution of line flows over time
                in the static analysis
    """

    # Assert that inputs are the proper size
    assert Y.shape[0] == Y.shape[1]
    N = Y.shape[0]
    correct_shape = (N, 1)
    for v in [PV_x, PQ_x, x_slack, V_abs, V_phase, P_input, Q_input, gamma]:
        assert v.shape == correct_shape
    assert not isinstance(I, np.ndarray) or H is None

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

    # Variables to keep track of the evolution of X, F, and P over time
    X = np.copy(X_t)
    F = K_G * np.cos(theta_0.T - theta_0) + K_B * np.sin(theta_0.T - theta_0)
    np.fill_diagonal(F, 0)
    F_static_all = F[np.newaxis, ...]
    F = F[np.newaxis, ...]
    P_track = np.copy(P)

    # Calculate I if it was not input
    if H is not None:
        P_multiplier = 2 * H / (ref_freq * 2 * np.pi) ** 2
        P_for_I = np.copy(P) * base_MVA * 1e6
        I = P_for_I * P_multiplier
        I[I <= 0] = 0.01
        print(I)

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
        P_track = np.hstack((P_track, P))
        t += delta_t

    # Cut line
    K_G_cut = np.copy(K_G)
    K_B_cut = np.copy(K_B)
    Y_for_NR_cut = np.copy(Y_for_NR)
    line_failures = []
    line_failures_static = []
    if lines_to_cut is not None:
        for cut in lines_to_cut:
            K_G_cut[cut] = 0
            K_B_cut[cut] = 0
            Y_for_NR_cut[cut] = 0
            K_G_cut[cut[::-1]] = 0
            K_B_cut[cut[::-1]] = 0
            Y_for_NR_cut[cut[::-1]] = 0
            line_failures += [(cut_time, cut[0], cut[1])]
            line_failures_static += [(0, cut[0], cut[1])]

    # Cut nodes, if applicable
    P_cut = np.copy(P)
    if nodes_to_cut is not None:
        P_cut[nodes_to_cut, :] = 0
        node_failures = [(cut_time, i) for i in nodes_to_cut]
    else:
        node_failures = []

    # Get threshold for line cutoffs
    if alpha is not None:
        F_threshold = alpha * np.sqrt(np.square(K_G_cut) + np.square(K_B_cut))
    else:
        F_threshold = None

    # Run simulation with cut line and cut more lines as necessary
    while t < t_max:
        X_t = solve.simulate_time_step(
            X_t, K_G_cut, K_B_cut, P_cut, I, gamma, delta_t
        )
        F_t = K_G_cut * np.cos(X_t[:N].T - X_t[:N]) + K_B_cut * np.sin(
            X_t[:N].T - X_t[:N]
        )
        np.fill_diagonal(F_t, 0)
        X = np.hstack((X, X_t))
        F = np.concatenate((F, F_t[np.newaxis, ...]), axis=0)
        P_track = np.hstack((P_track, P_cut))
        t += delta_t

        # check for threshold exceedance and cut lines if necessary
        if alpha is not None:
            threshold_exceeded_mask = np.abs(F_t) > F_threshold
            K_G_cut[threshold_exceeded_mask] = 0
            K_B_cut[threshold_exceeded_mask] = 0

            # record lines that were cut in line_failures
            cuts = np.argwhere(threshold_exceeded_mask)
            if cuts.size != 0:
                # get list of tuples where each tuple has (t,i,j) for the cut (without duplicates)
                cuts_list = list({tuple([t] + sorted(cut)) for cut in cuts})
                line_failures += cuts_list
        else:
            line_failures = None

        # check frequency deviation and cut load/generator if necessary
        if apply_freq_dev_during_sim:
            frequency_threshold_exceeded_mask = (
                np.abs(X_t[N:]) > frequency_deviation_threshold * 2 * np.pi
            )
            P_cut[frequency_threshold_exceeded_mask] = 0

            # record nodes forced to disconnect due to frequency deviation
            node_cuts = np.argwhere(frequency_threshold_exceeded_mask)
            if node_cuts.size != 0:
                # get list of tuples where each tuple has (t,i) for the failed node (without duplicates)
                cuts_list = list({tuple([t] + [cut[0]]) for cut in node_cuts})
                node_failures += cuts_list
    if not apply_freq_dev_during_sim:
        frequency_threshold_exceeded_mask = (
            np.abs(X_t[N:]) > frequency_deviation_threshold * 2 * np.pi
        )
        # record nodes forced to disconnect due to frequency deviation
        node_cuts = np.argwhere(frequency_threshold_exceeded_mask)
        if node_cuts.size != 0:
            # get list of tuples where each tuple has (t,i) for the failed node (without duplicates)
            cuts_list = list({tuple([t] + [cut[0]]) for cut in node_cuts})
            node_failures += cuts_list

    # Get static solution to disturbed state from N-R
    if alpha is not None:
        line_failures_static_minus1 = []
        n = 0
        while n == 0 or len(line_failures_static_minus1) != len(
            line_failures_static
        ):
            try:
                V_abs_static, theta_0_static, _, _ = newton_raphson(
                    Y_for_NR_cut,
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
                F_static = np.real(Y_for_NR_cut) * (
                    V_abs_static @ V_abs_static.T
                ) * np.cos(theta_0_static.T - theta_0_static[:N]) + np.imag(
                    Y_for_NR_cut
                ) * (
                    V_abs_static @ V_abs_static.T
                ) * np.sin(
                    theta_0_static.T - theta_0_static[:N]
                )
                F_static_all = np.concatenate(
                    (F_static_all, F_static[np.newaxis, ...]), axis=0
                )
                threshold_exceeded_mask = np.abs(F_static) > F_threshold
                Y_for_NR_cut[threshold_exceeded_mask] = 0
                # record lines that were cut in line_failures
                cuts = np.argwhere(threshold_exceeded_mask)
                line_failures_static_minus1 = list(line_failures_static)
                if cuts.size != 0:
                    # get list of tuples where each tuple has (n,i,j) for the cut (without duplicates)
                    cuts_list = list(
                        {tuple([n] + sorted(cut)) for cut in cuts}
                    )
                    line_failures_static += cuts_list
                n += 1
            except np.linalg.LinAlgError as e:
                print(
                    "Note N-R after cut could not solve due to error {}".format(
                        e
                    )
                )
                break

    else:
        line_failures_static = None

    # Prepare outputs
    T = np.arange(0, t_max + delta_t * 2, delta_t)
    i = np.arange(N)
    j = np.arange(N)
    n_index = np.arange(n + 1)
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
    real_power = xr.DataArray(
        data=P_track * base_MVA,
        dims=["node", "time"],
        coords=dict(node=i, time=T),
    )
    flows_static = xr.DataArray(
        data=F_static_all * base_MVA,
        dims=["iteration", "node_i", "node_j"],
        coords=dict(iteration=n_index, node_i=i, node_j=j),
    )

    return (
        theta,
        omega,
        flows,
        real_power,
        line_failures,
        node_failures,
        (F_threshold * base_MVA if alpha is not None else None),
        line_failures_static,
        flows_static,
    )


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
