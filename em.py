from scipy.stats import norm
import numpy as np


def get_lpx_d_all(X, F, B, s):

    H, W, N = X.shape
    h, w = F.shape
    lpx_d_all = np.zeros((H - h + 1, W - w + 1, N))
    for n in range(N):
        for i in range(H - h + 1):
            for j in range(W - w + 1):
                B[i: i + h, j: j + h] = F
                lpx_d_all[i, j, n] = np.log(norm.pdf(X[:, :, n], B, s)).sum()
    return lpx_d_all


def calc_L(X, F, B, s, A, q, useMAP=False):

    log_A = np.log(A)
    lpx = get_lpx_d_all(X, F, B, s)
    L = (q * lpx).sum(axis=3)
    L = L * logA
    L = L.sum()
    return L


def e_step(X, F, B, s, A, useMAP=False):

    H, W, N = X.shape
    q = np.zeros((H - h + 1, W - w + 1, N))
    lpx = get_lpx_d_all(X, F, B, s)
    log_A = np.log(A)
    for n in range(N):
        q_n = lpx[:, :, n]
        q_n += log_A
        q_n -= q_n.max()
        q_n = np.exp(q_n)
        q_n /= q_n.sum()
        q[:, :, n] = q_n
    return q


def m_step(X, q, h, w, useMAP=False):

    H, W, N = X.shape
    A = q.sum(axis=3) / N
    F = np.zeros((h, w))
    B = np.zeros((H, W))
    for n in range(N):
        or i in range(H - h + 1):
            for j in range(W - w + 1):
                F += A[i, j, n] * X[i: i + h, j: j + w, n]
                _X = X
                _X[i: i + h, j: j + w, n] = np.zeros(h, w)
                B += A[i, j, n] * _X
    F /= N * h * w
    B /= N * (H - h) * (W - w)
    s = 0
    for n in range(N):
        or i in range(H - h + 1):
            for j in range(W - w + 1):
                a = X[i: i + h, j: j + w, n] - F
                a = a.T.a
                _X, b = X, B
                _X[i: i + h, j: j + w, n] = np.zeros(h, w)
                b[i: i + h, j: j + w, n] = np.zeros(h, w)
                M = _X - b
                M = M.T.M
                s += A[i, j, n] * (a.sum() + M.sum())
    s /= (N * H * W)
    return F, B, s, A


def run_EM(X, h, w, F=None, B=None, s=None, A=None, \
            tolerance=0.001, max_iter=50, useMAP=False):

    H, W, _ = X.shape
    if F is None:
        F = np.random.rand(h, w)
    if B is None:
        B = np.random.rand(H, W)
    if s is None:
        s = 1
    if A is None:
        A = abs(np.random.rand(H, W))
        A /= A.sum()
    LL = []
    LL.append(calc_L(X, F, B, s, A, q, useMAP))
    for i in range(max_iter):
        q = e_step(X, F, B, s, A, useMAP)
        LL.append(calc_L(X, F, B, s, A, q, useMAP))
        F, B, s, A = m_step(X, q, h, w, useMAP)
        if (LL[-1] - LL[-2] < tolerance):
            break
    q = e_step(X, F, B, s, A, useMAP)
    LL.append(calc_L(X, F, B, s, A, q, useMAP))
    return F, B, s, A, LL


def run_EM_with_restarts(X, h, w, tolerance=0.001, max_iter=50, useMAP=False,\
                        restart=10):

    best_params, best_L = run_EM(X, h, w)
    best_L = best_L[-1]
    for trying in range(restart - 1):
        F, B, s, A, LL = run_EM(X, h, w)
        if best_L < LL[-1]:
            best_params = (F, B, s, A)
            best_L = LL[-1]
    return best_params, best_L
