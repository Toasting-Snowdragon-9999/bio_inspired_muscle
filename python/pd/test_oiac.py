"""
@brief Unit tests for the online impedance adaptation controller (ada_imp_ctrl).

Validates that the OIAC produces zero gains under zero error, that gains grow with
increasing tracking error, and that the fixed-gain (non-adaptive) PD path returns the
expected diagonal stiffness and damping.
"""
import numpy as np
from adaptive_imp import ada_imp_ctrl

def test_zero_error_gives_zero_gains():
    """
    @brief Assert that the OIAC yields zero stiffness and damping when desired equals measured state.
    """
    ctrl = ada_imp_ctrl(dof=3, params=(0.2, 5.0, 0.05), use_oiac=True)
    q = np.array([0.1, 0.2, 0.3])
    K, B = ctrl.update_impedance(q, q, np.zeros(3), np.zeros(3))  # q_d==q, dq_d==dq
    assert np.allclose(K, 0), f"Expected zero K, got {K}"
    assert np.allclose(B, 0), f"Expected zero B, got {B}"

def test_gains_grow_with_error():
    """
    @brief Assert that the adaptive stiffness magnitude increases as the tracking error grows.
    """
    ctrl = ada_imp_ctrl(dof=3, params=(0.2, 5.0, 0.05), use_oiac=True)
    small_err = np.array([0.01, 0.01, 0.01])
    large_err = np.array([0.1, 0.1, 0.1])
    K_small, _ = ctrl.update_impedance(np.zeros(3), small_err, np.zeros(3), np.zeros(3))
    K_large, _ = ctrl.update_impedance(np.zeros(3), large_err, np.zeros(3), np.zeros(3))
    assert np.linalg.norm(K_large) > np.linalg.norm(K_small), "K should grow with error"

def test_fixed_pd_matches_expected_gains():
    """
    @brief Assert that the non-adaptive PD path returns the expected fixed diagonal stiffness and damping.
    """
    ctrl = ada_imp_ctrl(dof=3, params=(0.2, 5.0, 0.05), use_oiac=False)
    K, B = ctrl.update_impedance(np.zeros(3), np.ones(3), np.zeros(3), np.ones(3))
    assert np.allclose(np.diag(K), 90.0)
    assert np.allclose(np.diag(B), 15.0)

def main():
    """
    @brief Run all OIAC unit tests in sequence and report success.
    """
    test_zero_error_gives_zero_gains()
    test_gains_grow_with_error()
    test_fixed_pd_matches_expected_gains()
    print("All tests passed!")

if __name__ == "__main__":
    main()