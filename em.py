from scipy.stats import norm
import numpy as np
import sys


def get_lpx_d_all(X, F, B, s):
    H, W, N = X.shape
    h, w = F.shape
    lpx_d_all = np.zeros((H - h, W - w, N))
    for i in range(H - h):
        for j in range(W - w):
            BF = B
            BF[i: i + h, j: j + w] = F
            for n in range(N):
                lpx_d_all[i, j, n] = np.log(norm.pdf(X[:, :, n], BF, s)).sum()
    return lpx_d_all


def calc_L(X, F, B, s, A, q, useMAP=False):

    N = X.shape[2]
    log_A = np.log(A)
    lpx = get_lpx_d_all(X, F, B, s)
    L = 0
    for n in range(N):
        L += ((lpx[:, :, n] + log_A) * q[:, :, n]).sum()
    return L


def e_step(X, F, B, s, A, useMAP=False):

    H, W, N = X.shape
    h, w = F.shape
    q = np.zeros((H - h, W - w, N))
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
    A = q.sum(axis=2) / N
    F = np.zeros((h, w))
    B = np.zeros((H, W))
    for i in range(H - h):
        for j in range(W - w):
            _F = (X[i: i + h, j: j + w, :] * q[i, j, :]).sum(axis=2)
            _B = X
            _B[i:i + h, j:j + w, :] = np.zeros((h, w, N))
            _B = (_B * q[i, j, :]).sum(axis=2)
            F += _F
            B += _B
    F /= N
    B /= N
    s = 0
    for n in range(N):
        for i in range(H - h):
            for j in range(W - w):
                f = X[i: i + h, j: j + w, n] - F
                f = np.trace(f.T.dot(f))
                _X, _B = X[:, :, n], B
                _X[i: i + h, j: j + w] = np.zeros((h, w))
                _B[i: i + h, j: j + w] = np.zeros((h, w))
                b = _X - _B
                b = np.trace(b.T.dot(b))
                s += q[i, j, n] * (f + b)
    s /= (N * H * W)
    return F, B, s, A


def run_EM(X, h, w, F=None, B=None, s=None, A=None, \
            tolerance=0.001, max_iter=10, useMAP=False):

    H, W, _ = X.shape
    if F is None:
        F = np.random.random((h, w))
    if B is None:
        B = np.random.random((H, W))
    if s is None:
        s = .1
    if A is None:
        A = abs(np.random.rand(H - h, W - w))
        A /= A.sum()
    LL = []
    q = e_step(X, F, B, s, A, useMAP)
    F, B, s, A = m_step(X, q, h, w, useMAP)
    last_L = calc_L(X, F, B, s, A, q, useMAP)
    #print("log_liklihood:{0:%f}".format(last_L))
    print(last_L)
    LL.append(last_L)
    for i in range(max_iter - 1):
        cur_L = last_L
        q = e_step(X, F, B, s, A, useMAP)
        F, B, s, A = m_step(X, q, h, w, useMAP)
        last_L = calc_L(X, F, B, s, A, q, useMAP)
        #print("log_liklihood:{0%f}".format(last_L))
        print(last_L)
        LL.append(last_L)
        if (last_L - cur_L < tolerance):
            break
    return (F, B, s, A), LL


def run_EM_with_restarts(X, h, w, tolerance=0.001, max_iter=50, useMAP=False,\
                        restart=10):

    best_params, best_L = run_EM(X, h, w)
    for trying in range(restart - 1):
        params, LL = run_EM(X, h, w)
        if best_L < LL:
            best_params = params
            best_L = LL
    return best_params, best_L
