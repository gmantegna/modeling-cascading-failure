import numpy as np
import cmath
from pathlib import Path
from src.modeling_cascading_failure import simulate


def test_simulate():
    # Prepare inputs for testing
    num_buses = 14
    eps = 0.0000001
    max_iter = 100
    I = 1
    gamma = 1

    # helper functions
    npphase = np.vectorize(cmath.phase)

    def convert_complex(x):
        return complex(x.replace("i", "j"))

    npconvert = np.vectorize(convert_complex)

    # load data
    test_folder = Path(__file__).resolve().parent
    data_folder = test_folder / "data"
    Y = np.loadtxt(data_folder / f"Y{num_buses}.csv", delimiter=",", dtype=str)
    Y = npconvert(Y)
    v = np.loadtxt(data_folder / f"v{num_buses}.csv", delimiter=",", dtype=str)
    v = npconvert(v)
    slack = np.loadtxt(
        data_folder / f"slack{num_buses}.csv",
        delimiter=",",
        dtype=str,
    )
    slack = int(slack) - 1
    s = np.loadtxt(data_folder / f"s{num_buses}.csv", delimiter=",", dtype=str)
    s = npconvert(s)
    pv = np.loadtxt(
        data_folder / f"pv{num_buses}.csv", delimiter=",", dtype=str
    )
    pv = pv.astype(int) - 1
    pq = np.loadtxt(
        data_folder / f"pq{num_buses}.csv", delimiter=",", dtype=str
    )
    pq = pq.astype(int) - 1
    base_mva = np.loadtxt(
        data_folder / f"base_mva{num_buses}.csv",
        delimiter=",",
        dtype=str,
    )
    base_mva = float(base_mva)

    # clean up data
    PV_x = np.array([1 if i in pv else 0 for i in range(Y.shape[0])]).reshape(
        -1, 1
    )
    PQ_x = np.array([1 if i in pq else 0 for i in range(Y.shape[0])]).reshape(
        -1, 1
    )
    x_slack = np.array(
        [1 if i == slack else 0 for i in range(Y.shape[0])]
    ).reshape(-1, 1)
    V_abs_full = abs(v).reshape(-1, 1)
    V_phase_full = npphase(v).reshape(-1, 1)
    P_full = np.real(s).reshape(-1, 1)
    Q_full = np.imag(s).reshape(-1, 1)
    V_abs = np.copy(V_abs_full)
    V_abs[PQ_x == 1] = 1
    V_phase = np.copy(V_phase_full)
    V_phase[x_slack == 0] = 0
    P_input = np.copy(P_full)
    P_input[x_slack == 1] = np.nan
    Q_input = np.copy(Q_full)
    Q_input[PQ_x == 0] = np.nan
    I_array = np.ones(num_buses).reshape(-1, 1) * I
    gamma_array = np.ones(num_buses).reshape(-1, 1) * gamma

    # run simulation with all settings turned on and check it matches reference
    reference = np.array(
        [
            3.39509193,
            3.30568646,
            3.14905934,
            3.27480629,
            3.27791941,
            54.51449994,
            3.43334434,
            3.35581235,
            3.61107126,
            3.75282305,
            28.08350174,
            32.13828985,
            54.49936453,
            16.30968438,
        ]
    )
    (
        theta,
        omega,
        F,
        P,
        line_failures,
        node_failures,
        F_threshold,
        _,
        _,
        _,
        _,
        _,
    ) = simulate.simulate_system(
        Y,
        PV_x,
        PQ_x,
        x_slack,
        V_abs,
        V_phase,
        P_input,
        Q_input,
        eps,
        max_iter,
        base_MVA=base_mva,
        lines_to_cut=[(12, 13)],
        nodes_to_cut=[1],
        cut_time=1,
        delta_t=0.1,
        alpha=0.7,
        frequency_deviation_threshold=0.5,
        apply_freq_dev_during_sim=True,
        I=I_array,
        H=None,
        gamma=gamma_array,
        t_max=10,
        include_resistive_losses=True,
        ref_freq=60,
    )
    assert np.max(np.abs(theta.sel(time=10).values - reference)) < 0.1

    # run simulation with no node cutting and check
    reference = np.array(
        [
            3.33455112,
            3.24617975,
            3.08300331,
            3.19912976,
            3.20635134,
            54.03800859,
            3.33056144,
            3.25401536,
            3.50282062,
            3.63259065,
            28.09057746,
            31.85205137,
            54.4995766,
            12.40006564,
        ]
    )
    (
        theta,
        omega,
        F,
        P,
        line_failures,
        node_failures,
        F_threshold,
        _,
        _,
        _,
        _,
        _,
    ) = simulate.simulate_system(
        Y,
        PV_x,
        PQ_x,
        x_slack,
        V_abs,
        V_phase,
        P_input,
        Q_input,
        eps,
        max_iter,
        base_MVA=base_mva,
        lines_to_cut=[(12, 13)],
        nodes_to_cut=None,
        cut_time=1,
        delta_t=0.1,
        alpha=0.7,
        frequency_deviation_threshold=0.5,
        apply_freq_dev_during_sim=True,
        I=I_array,
        H=None,
        gamma=gamma_array,
        t_max=10,
        include_resistive_losses=True,
        ref_freq=60,
    )
    assert np.max(np.abs(theta.sel(time=10).values - reference)) < 0.1

    # run simulation with no line cutting and check
    reference = np.array(
        [
            -1.40318317,
            -1.41497629,
            -1.50716535,
            -1.43381101,
            -1.41826864,
            -1.37136323,
            -1.38901583,
            -1.36472946,
            -1.39695364,
            -1.38607707,
            -1.36805387,
            -1.36060336,
            -1.3782089,
            -1.39655448,
        ]
    )
    (
        theta,
        omega,
        F,
        P,
        line_failures,
        node_failures,
        F_threshold,
        _,
        _,
        _,
        _,
        _,
    ) = simulate.simulate_system(
        Y,
        PV_x,
        PQ_x,
        x_slack,
        V_abs,
        V_phase,
        P_input,
        Q_input,
        eps,
        max_iter,
        base_MVA=base_mva,
        lines_to_cut=None,
        nodes_to_cut=[0],
        cut_time=1,
        delta_t=0.1,
        alpha=0.8,
        frequency_deviation_threshold=0.5,
        apply_freq_dev_during_sim=True,
        I=I_array,
        H=None,
        gamma=gamma_array,
        t_max=10,
        include_resistive_losses=True,
        ref_freq=60,
    )
    assert np.max(np.abs(theta.sel(time=10).values - reference)) < 0.1

    # run simulation without resistive losses and check
    reference = np.array(
        [
            -1.4688839,
            -1.47633446,
            -1.55851264,
            -1.48857646,
            -1.47596897,
            -1.43367971,
            -1.44002001,
            -1.41540215,
            -1.44619423,
            -1.43428617,
            -1.42119576,
            -1.41684692,
            -1.42952604,
            -1.43768977,
        ]
    )
    (
        theta,
        omega,
        F,
        P,
        line_failures,
        node_failures,
        F_threshold,
        _,
        _,
        _,
        _,
        _,
    ) = simulate.simulate_system(
        Y,
        PV_x,
        PQ_x,
        x_slack,
        V_abs,
        V_phase,
        P_input,
        Q_input,
        eps,
        max_iter,
        base_MVA=base_mva,
        lines_to_cut=None,
        nodes_to_cut=[0],
        cut_time=1,
        delta_t=0.1,
        alpha=0.9,
        frequency_deviation_threshold=1,
        apply_freq_dev_during_sim=True,
        I=I_array,
        H=None,
        gamma=gamma_array,
        t_max=10,
        include_resistive_losses=False,
        ref_freq=60,
    )
    assert np.max(np.abs(theta.sel(time=10).values - reference)) < 0.1
