import numpy as np
import tensorflow as tf


class Kernel:
    def __init__(
        self,
        covar_matrix_func,
        covar_diag_func,
        descriptor,
        kernels=None
    ):

        self._descriptor = descriptor
        self._covar_matrix_func = covar_matrix_func
        self._covar_diag_func = covar_diag_func
        self._kernels = kernels

    @property
    def descriptor(self):
        return self._descriptor

    def covar_matrix(self, t_X, t_Z):
        return self._covar_matrix_func(t_X=t_X, t_Z=t_Z)

    def covar_diag(self, t_X):
        return self._covar_diag_func(t_X=t_X)


def create_matern32_kernel(t_alpha, t_gamma, dtype=tf.float64):
    def matrix_func(t_X, t_Z):
        xx = 0.5 * t_gamma * tf.reduce_sum(t_X * t_X, axis=1, keepdims=True)
        zz = 0.5 * t_gamma * tf.reduce_sum(t_Z * t_Z, axis=1, keepdims=True)
        sq_dist_xz = xx + tf.transpose(zz) - \
            t_gamma * tf.matmul(t_X, t_Z, transpose_b=True)
        sqrt_3 = tf.constant(np.sqrt(3.0), dtype=dtype)
        sqrt_3_dist_xz = sqrt_3 * tf.sqrt(sq_dist_xz + 1.0e-12)
        return t_alpha * (1.0 + sqrt_3_dist_xz) * tf.exp(- sqrt_3_dist_xz)

    def diag_func(t_X):
        t_N = tf.shape(t_X)[0]
        return t_alpha * tf.ones([t_N], dtype=dtype)

    return Kernel(descriptor='Matern 3/2',
                  covar_matrix_func=matrix_func,
                  covar_diag_func=diag_func,
                  kernels=None)


def create_squared_exp_kernel(t_alpha, t_gamma, dtype=tf.float64):
    def matrix_func(t_X, t_Z):
        xx = 0.5 * t_gamma * tf.reduce_sum(t_X * t_X, axis=1, keepdims=True)
        zz = 0.5 * t_gamma * tf.reduce_sum(t_Z * t_Z, axis=1, keepdims=True)
        sq_dist_xz = xx + tf.transpose(zz) - \
            t_gamma * tf.matmul(t_X, t_Z, transpose_b=True)
        return t_alpha * tf.exp(- sq_dist_xz)

    def diag_func(t_X):
        t_N = tf.shape(t_X)[0]
        return t_alpha * tf.ones([t_N], dtype=dtype)

    return Kernel(descriptor='Squared Exponential',
                  covar_matrix_func=matrix_func,
                  covar_diag_func=diag_func,
                  kernels=None)


def real_variable(inital_value, name=None, dtype=tf.float64):
    t_a = tf.Variable(inital_value, dtype=dtype, name=name)
    return t_a


def positive_variable(inital_value, name=None, dtype=tf.float64):
    t_a = tf.Variable(
        tf.math.log(tf.cast(inital_value, dtype=dtype)),
        dtype=dtype,
        name=name
    )
    return tf.exp(t_a)


def log_det_from_chol(L):
    return 2.0 * tf.reduce_sum(tf.log(tf.linalg.tensor_diag_part(L)))


def vec_to_tri(tri, N):
    """ map from vector to lower triangular matrix (adapted from gpflow) """
    indices = list(zip(*np.tril_indices(N)))
    indices = tf.constant([list(i) for i in indices], dtype=tf.int32)
    tri_part = tf.scatter_nd(indices=indices, shape=[N, N], updates=tri)
    return tri_part


def init_triangular(N, diag=None):
    """ Initialize lower triangular parametrization for
    covariance matrices (adapted from gpflow) """
    I_matrix = int(N*(N+1)/2)
    indices = list(zip(*np.tril_indices(N)))
    diag_indices = \
        np.array([idx for idx, (x, y) in enumerate(indices) if x == y])
    I_matrix = np.zeros(I_matrix)
    if diag is None:
        I_matrix[diag_indices] = 1
    else:
        I_matrix[diag_indices] = diag
    return I_matrix


class EulerMaruyama:
    def __init__(self, f, total_time, nsteps, jointly_gaussian=False):
        self.ts = np.linspace(0, total_time, nsteps)
        self.f = f
        self.jointly_gaussian = jointly_gaussian

    def forward(self, y0, save_intermediate=False):
        time_grid = tf.constant(self.ts, dtype=tf.float64, name='t')
        # y0 = tf.constant(y0, dtype=tf.float64, name='y0')
        time_delta_grid = time_grid[1:] - time_grid[:-1]
        time_grid = time_grid[1:]
        time_combined = \
            tf.concat([time_grid[:, None], time_delta_grid[:, None]], axis=1)
        scan_func = self._make_scan_func(self.f)

        if save_intermediate:
            y_grid = tf.scan(scan_func, time_combined, y0)
            y_s = tf.concat([[y0], y_grid], axis=0)
            y_t = y_s[-1, :, :]
            return y_t, y_s
        else:
            y_t = tf.foldl(scan_func, time_combined, y0)
            return y_t, None

    def _step_func(self, evol_func, t_and_dt, y):
        t = t_and_dt[0]
        dt = t_and_dt[1]
        mu, var = evol_func(y, t)
        if var.get_shape().ndims == 3:
            raise NotImplementedError
        dt_cast = tf.cast(dt, y.dtype)

        # Mean increment (without the stochastic component in the flow field)
        mean_increment = mu * dt_cast

        # Stochastic component of the increment in the flow field
        if self.jointly_gaussian:
            # Jointly Gaussian sampling of the incremets in the vector field
            # var is assumed to be the cholesky factor of the covariance matrix
            dy = mean_increment + \
                 tf.sqrt(dt_cast) * tf.matmul(
                     var, tf.random.normal(
                         shape=(tf.shape(mu)[0], 1),
                         dtype=y.dtype
                     )
                 )
        else: # The same (fixed) Winer trajectory for all inputs
            # Marginal variances for all points in the flow field
            cov_diag = tf.expand_dims(
                tf.linalg.tensor_diag_part(
                    tf.matmul(var, var, transpose_b=True)), -1)
            dy = mean_increment + \
                 tf.sqrt(dt_cast * cov_diag) * tf.random.normal(
                     shape=(1, 1), dtype=y.dtype)
        return dy

    def _make_scan_func(self, evol_func):
        def scan_func(y, t_and_dt):
            dy = self._step_func(evol_func, t_and_dt, y)
            dy = tf.cast(dy, dtype=y.dtype)
            return y + dy
        return scan_func


def mu_sigma_tilde(
    t_x_space,
    t_x_time,
    t_Z0,
    t_U1,
    t_Sigma1,
    t_kernel,
    dtype=tf.float64,
    jitter=1e-6):

    t_D = tf.shape(t_x_space)[0]
    t_M = tf.shape(t_Z0)[0]

    t_x_time = tf.tile(tf.reshape(t_x_time, (1, 1)), tf.shape(t_x_space))
    t_f0 = tf.concat([t_x_space, t_x_time], axis=1)
    t_K_ZZ0 = t_kernel.covar_matrix(t_Z0, t_Z0)

    t_L_Z0 = tf.linalg.cholesky(
        t_K_ZZ0 + jitter * tf.linalg.tensor_diag(tf.ones(t_M, dtype=dtype)))

    t_K_Zf0 = t_kernel.covar_matrix(t_Z0, t_f0)
    t_A = tf.transpose(tf.linalg.triangular_solve(t_L_Z0, t_K_Zf0), (1, 0))
    t_B = tf.linalg.triangular_solve(t_L_Z0, t_U1)
    t_mu1_tilde = tf.matmul(t_A, t_B)

    t_bracket = t_K_ZZ0 - t_Sigma1
    t_C = tf.linalg.triangular_solve(t_L_Z0, t_bracket)
    t_D = tf.matmul(t_A, t_C)
    t_E_T = tf.linalg.triangular_solve(t_L_Z0, tf.transpose(t_D, (1, 0)))
    t_E = tf.transpose(t_E_T, (1, 0))
    t_aba = tf.matmul(t_E, t_A, transpose_b=True)
    t_K_ff0_diag = t_kernel.covar_matrix(t_f0, t_f0)
    t_sigma1_tilda = t_K_ff0_diag - t_aba

    t_sigma1_tilda_chol = tf.linalg.cholesky(
        t_sigma1_tilda + \
        jitter * tf.eye(tf.shape(t_x_space)[0], dtype=dtype)
    )

    return t_mu1_tilde, t_sigma1_tilda_chol


def kl_divergence(t_L_Z, t_mu, t_Sigma, dtype=tf.float64):
    t_M = tf.shape(t_mu)[0]
    t_m = t_mu
    t_L_Z_inv = tf.linalg.triangular_solve(t_L_Z, tf.eye(t_M, dtype=dtype))
    t_LZ_inv_mu = tf.linalg.triangular_solve(t_L_Z, t_m)
    t_LZ_inv_Sigma = tf.linalg.triangular_solve(t_L_Z, t_Sigma)
    # K^-1 Sigma
    # (L L^T)^1 Sigma
    # L^-T L^-1 Sigma
    t_K_inv_Sigma = tf.matmul(t_L_Z_inv, t_LZ_inv_Sigma, transpose_a=True)

    t_Tr_K_inv_Sigma = tf.linalg.trace(t_K_inv_Sigma)

    # mu^T K^-1 mu
    # mu^T (L L^T)^-1 mu
    # mu^T L^-T L^-1 mu
    t_muT_L_invT = tf.matmul(t_L_Z_inv, t_LZ_inv_mu, transpose_a=True)
    t_muT_K_inv_mu = tf.matmul(t_m, t_muT_L_invT, transpose_a=True)

    # t_logdet_K = tf.linalg.logdet(t_K_ZZ)
    t_logdet_K = log_det_from_chol(t_L_Z)

    t_sum_sigma = tf.reduce_sum(
        tf.log(tf.linalg.tensor_diag_part(t_Sigma)), axis=0)

    kl = 0.5 * (
        t_Tr_K_inv_Sigma + \
        t_muT_K_inv_mu - \
        tf.cast(t_M, dtype=dtype) + \
        t_logdet_K - \
        t_sum_sigma
    )
    return kl
