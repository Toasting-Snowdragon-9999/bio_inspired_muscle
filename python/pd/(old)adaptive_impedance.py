import numpy as np


class ada_imp_ctrl( ):
    """Online impedance adaptation with bounded diagonal gains."""

    def __init__(
        self,
        dof: int,
        kp0=20.0,
        kd0=2.0,
        kp_bounds=(1.0, 120.0),
        kd_bounds=(0.1, 20.0),
        adapt_rate: float = 0.02,
        tau_limit=23.7,
        use_full_matrix: bool = False,
    ):
        self.DOF = dof
        self.use_full_matrix = use_full_matrix
        self.adapt_rate = float(adapt_rate)

        self.kp_min, self.kp_max = kp_bounds
        self.kd_min, self.kd_max = kd_bounds

        self.kp0 = self._to_vec(kp0)
        self.kd0 = self._to_vec(kd0)
        self.tau_limit = self._to_vec(tau_limit)

        self.kp = np.clip(self.kp0.copy(), self.kp_min, self.kp_max)
        self.kd = np.clip(self.kd0.copy(), self.kd_min, self.kd_max)

        self.k_mat = np.diag(self.kp)
        self.b_mat = np.diag(self.kd)

        # Scaling from error magnitude to adaptive gain target.
        self._kp_err_scale = 10.0
        self._kd_err_scale = 2.0

    def _to_vec(self, x) -> np.ndarray:
        arr = np.asarray(x, dtype=float).reshape(-1)
        if arr.size == 1:
            return np.full(self.DOF, float(arr[0]), dtype=float)
        if arr.size != self.DOF:
            raise ValueError(f"Expected scalar or length {self.DOF}, got {arr.size}")
        return arr

    def update_impedance(
        self,
        q: np.ndarray,
        q_d: np.ndarray,
        dq: np.ndarray,
        dq_d: np.ndarray,
        freeze_adaptation: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Update stiffness and damping matrices.
        Gains are kept positive, diagonal and bounded.
        """
        q = self._to_vec(q)
        q_d = self._to_vec(q_d)
        dq = self._to_vec(dq)
        dq_d = self._to_vec(dq_d)

        if not freeze_adaptation:
            e = q_d - q
            de = dq_d - dq

            kp_target = self.kp0 + self._kp_err_scale * np.abs(e)
            kd_target = self.kd0 + self._kd_err_scale * np.abs(de)

            self.kp += self.adapt_rate * (kp_target - self.kp)
            self.kd += self.adapt_rate * (kd_target - self.kd)

        self.kp = np.clip(self.kp, self.kp_min, self.kp_max)
        self.kd = np.clip(self.kd, self.kd_min, self.kd_max)

        if self.use_full_matrix:
            self.k_mat = np.diag(self.kp)
            self.b_mat = np.diag(self.kd)
        else:
            # For v1 we intentionally keep diagonal gains only.
            self.k_mat = np.diag(self.kp)
            self.b_mat = np.diag(self.kd)

        return self.k_mat, self.b_mat
