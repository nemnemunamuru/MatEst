import numpy as np

def standardize(xy: np.ndarray, mean: np.ndarray=None, std: np.ndarray=None):
    if mean is None:
        mean = xy.mean(axis=0)
    if std is None:
        std = xy.std(axis=0)
    std[std==0] = 1.0
    z = (xy - mean) / std
    return z, mean, std

def build_design_matrix(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    terms = []
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            terms.append((i, j))
    cols = [(x**i) * (y**j) for (i, j) in terms]
    return np.column_stack(cols)

def solve_ridge(A: np.ndarray, z: np.ndarray, lam: float) -> np.ndarray:
    AtA = A.T @ A
    m = AtA.shape[0]
    return np.linalg.solve(AtA + lam * np.eye(m), A.T @ z)

def auto_degree(max_degree: int, n_points: int) -> int:
    for d in range(max_degree, -1, -1):
        n_terms = (d + 1) * (d + 2) // 2
        if n_terms <= n_points:
            return d
    return 0

def fit_predict_polynomial_norm(xy_m: np.ndarray, z: np.ndarray, xy_all: np.ndarray, degree: int, lam: float) -> np.ndarray:
    z_m, mu, sig = standardize(xy_m)
    z_all, _, _ = standardize(xy_all, mu, sig)
    A_m = build_design_matrix(z_m[:,0], z_m[:,1], degree)
    w = solve_ridge(A_m, z, lam)
    A_all = build_design_matrix(z_all[:,0], z_all[:,1], degree)
    return A_all @ w

def rbf_kernel(xy1: np.ndarray, xy2: np.ndarray, sigma: float) -> np.ndarray:
    x2 = np.sum(xy1**2, axis=1, keepdims=True)
    y2 = np.sum(xy2**2, axis=1, keepdims=True).T
    d2 = x2 + y2 - 2 * (xy1 @ xy2.T)
    return np.exp(-d2 / (2.0 * sigma * sigma))

def fit_predict_rbf(xy_m: np.ndarray, z: np.ndarray, xy_all: np.ndarray, sigma: float, lam: float) -> np.ndarray:
    K = rbf_kernel(xy_m, xy_m, sigma)
    alpha = np.linalg.solve(K + lam * np.eye(K.shape[0]), z)
    K_all = rbf_kernel(xy_all, xy_m, sigma)
    return K_all @ alpha

def fit_predict_radial(xy_m: np.ndarray, zx: np.ndarray, zy: np.ndarray, xy_all: np.ndarray, degree: int, lam: float):
    Xm = xy_m[:,0]; Ym = xy_m[:,1]
    Rm = np.sqrt(Xm*Xm + Ym*Ym)
    Rtarget = np.sqrt(zx*zx + zy*zy)
    if degree < 1:
        orders = [1]
    else:
        orders = [k for k in range(1, degree+1) if k % 2 == 1]
    if not orders:
        orders = [1]
    A_m = np.column_stack([Rm**k for k in orders]) if len(Rm)>0 else np.zeros((0,1))
    coef = np.linalg.solve(A_m.T @ A_m + lam*np.eye(len(orders)), A_m.T @ Rtarget)
    Xa = xy_all[:,0]; Ya = xy_all[:,1]
    Ra = np.sqrt(Xa*Xa + Ya*Ya)
    A_all = np.column_stack([Ra**k for k in orders])
    Rpred_all = A_all @ coef
    eps = 1e-12
    uXa = Xa / np.maximum(Ra, eps)
    uYa = Ya / np.maximum(Ra, eps)
    pred_x_all = Rpred_all * uXa
    pred_y_all = Rpred_all * uYa
    Rpred_m = A_m @ coef
    uXm = Xm / np.maximum(Rm, eps)
    uYm = Ym / np.maximum(Rm, eps)
    pred_x_meas = Rpred_m * uXm
    pred_y_meas = Rpred_m * uYm
    degree_used = max(orders)
    return pred_x_all, pred_y_all, pred_x_meas, pred_y_meas, degree_used
