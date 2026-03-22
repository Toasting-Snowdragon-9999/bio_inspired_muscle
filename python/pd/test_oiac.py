import numpy as np
from adaptive_imp import ada_imp_ctrl

def test_zero_error_gives_zero_gains():
    ctrl = ada_imp_ctrl(dof=3, params=(0.2, 5.0, 0.05), use_oiac=True)
    q = np.array([0.1, 0.2, 0.3])
    K, B = ctrl.update_impedance(q, q, np.zeros(3), np.zeros(3))  # q_d==q, dq_d==dq
    assert np.allclose(K, 0), f"Expected zero K, got {K}"
    assert np.allclose(B, 0), f"Expected zero B, got {B}"

def test_gains_grow_with_error():
    ctrl = ada_imp_ctrl(dof=3, params=(0.2, 5.0, 0.05), use_oiac=True)
    small_err = np.array([0.01, 0.01, 0.01])
    large_err = np.array([0.1, 0.1, 0.1])
    K_small, _ = ctrl.update_impedance(np.zeros(3), small_err, np.zeros(3), np.zeros(3))
    K_large, _ = ctrl.update_impedance(np.zeros(3), large_err, np.zeros(3), np.zeros(3))
    assert np.linalg.norm(K_large) > np.linalg.norm(K_small), "K should grow with error"

def test_fixed_pd_matches_expected_gains():
    ctrl = ada_imp_ctrl(dof=3, params=(0.2, 5.0, 0.05), use_oiac=False)
    K, B = ctrl.update_impedance(np.zeros(3), np.ones(3), np.zeros(3), np.ones(3))
    assert np.allclose(np.diag(K), 90.0)
    assert np.allclose(np.diag(B), 15.0)

def main():
    test_zero_error_gives_zero_gains()
    test_gains_grow_with_error()
    test_fixed_pd_matches_expected_gains()
    print("All tests passed!")

if __name__ == "__main__":
    main()