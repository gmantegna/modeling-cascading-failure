import numpy as np
from src.modeling_cascading_failure import solve


def test_simulate_time_step():
    # Test case inputs
    X_t_test = np.array([[0, 0, 0.321, 0]]).T
    K_test = np.array([[-5, 5], [5, -5]])
    P_test = np.array([[1, 1.5]]).T
    I_test = np.array([[1, 1]]).T
    gamma_test = np.array([[1, 1]]).T
    delta_t_test = 0.1

    assert (
        solve.simulate_time_step(
            X_t_test, K_test, P_test, I_test, gamma_test, delta_t_test
        )[0].any()
        == solve.simulate_time_step(
            X_t_test, K_test, P_test, I_test, gamma_test, delta_t_test
        )[0].any()
    )
