import numpy as np
import scipy
import math

def calculate_log_probability(X, F, B, s):
    """
    Calculates log p(X_k|d_k,F,B,s) for all images X_k in X and
    all possible displacements d_k.

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    F : array, shape (h, w)
        Estimate of villain's face.
    B : array, shape (H, W)
        Estimate of background.
    s : float
        Estimate of standard deviation of Gaussian noise.

    Returns
    -------
    ll : array, shape(H-h+1, W-w+1, K)
        ll[dh,dw,k] - log-likelihood of observing image X_k given
        that the villain's face F is located at displacement (dh, dw)
    """
    H, W, N = X.shape
    h, w = F.shape
    ll = np.zeros((H - h + 1, W - w + 1, N))

    for i in range(H - h + 1):
        for j in range(W - w + 1):
                m  = B.copy()
                m[i: i + h, j: j + w] = F.copy()
                m = -(X - m[:, :, np.newaxis]) ** 2 
                m /= 2 * s ** 2
                m += - 0.5 * np.log(2 * np.pi) - np.log(s)
                ll[i, j] = m.sum(axis=(0, 1))

    return ll


def calculate_lower_bound(X, F, B, s, A, q, use_MAP=False):
    """
    Calculates the lower bound L(q,F,B,s,A) for the marginal log likelihood.

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    F : array, shape (h, w)
        Estimate of villain's face.
    B : array, shape (H, W)
        Estimate of background.
    s : float
        Estimate of standard deviation of Gaussian noise.
    A : array, shape (H-h+1, W-w+1)
        Estimate of prior on displacement of face in any image.
    q : array
        If use_MAP = False: shape (H-h+1, W-w+1, K)
            q[dh,dw,k] - estimate of posterior of displacement (dh,dw)
            of villain's face given image Xk
        If use_MAP = True: shape (2, K)
            q[0,k] - MAP estimates of dh for X_k
            q[1,k] - MAP estimates of dw for X_k
    use_MAP : bool, optional
        If true then q is a MAP estimates of displacement (dh,dw) of
        villain's face given image Xk.

    Returns
    -------
    L : float
        The lower bound L(q,F,B,s,A) for the marginal log likelihood.
    """
    ll = calculate_log_probability(X, F, B, s)

    if not use_MAP:
        tmp = ll * q
        tmp[q == 0] = 0
        L = np.sum(tmp)

        tmp = np.log(A + 10 ** (-7))[:, :, np.newaxis] * q
        tmp[q == 0] = 0
        L += np.sum(tmp)

        tmp = np.log(q + 10 ** (-7)) * q
        tmp[q == 0] = 0
        L -= np.sum(tmp)
    else:
        logA = np.log(A + 10 ** (-7))
        L = 0
        for k, (l, m) in enumerate(q.T):
            L += ll[l, m, k] + logA[l, m]
    return L


def run_e_step(X, F, B, s, A, use_MAP=False):
    """
    Given the current esitmate of the parameters, for each image Xk
    esitmates the probability p(d_k|X_k,F,B,s,A).

    Parameters
    ----------
    X : array, shape(H, W, K)
        K images of size H x W.
    F  : array_like, shape(h, w)
        Estimate of villain's face.
    B : array shape(H, W)
        Estimate of background.
    s : scalar, shape(1, 1)
        Eestimate of standard deviation of Gaussian noise.
    A : array, shape(H-h+1, W-w+1)
        Estimate of prior on displacement of face in any image.
    use_MAP : bool, optional,
        If true then q is a MAP estimates of displacement (dh,dw) of
        villain's face given image Xk.

    Returns
    -------
    q : array
        If use_MAP = False: shape (H-h+1, W-w+1, K)
            q[dh,dw,k] - estimate of posterior of displacement (dh,dw)
            of villain's face given image Xk
        If use_MAP = True: shape (2, K)
            q[0,k] - MAP estimates of dh for X_k
            q[1,k] - MAP estimates of dw for X_k
    """
    l_q = calculate_log_probability(X, F, B, s) + np.log(A  + 10 ** (-7))[:,:, np.newaxis]
    if not use_MAP:
        l_q -= l_q.max(axis=(0, 1))
        q = np.exp(l_q)
        q /= q.sum(axis=(0, 1))
    else:
        max_idx = l_q.reshape(-1, l_q.shape[2]).argmax(0)
        q = np.column_stack(np.unravel_index(max_idx, l_q.shape[0:2]))
        q = q.T
    return q
    


def run_m_step(X, q, h, w, use_MAP=False):
    """
    Estimates F,B,s,A given esitmate of posteriors defined by q.

    Parameters
    ----------
    X : array, shape(H, W, K)
        K images of size H x W.
    q  :
        if use_MAP = False: array, shape (H-h+1, W-w+1, K)
           q[dh,dw,k] - estimate of posterior of displacement (dh,dw)
           of villain's face given image Xk
        if use_MAP = True: array, shape (2, K)
            q[0,k] - MAP estimates of dh for X_k
            q[1,k] - MAP estimates of dw for X_k
    h : int
        Face mask height.
    w : int
        Face mask width.
    use_MAP : bool, optional
        If true then q is a MAP estimates of displacement (dh,dw) of
        villain's face given image Xk.

    Returns
    -------
    F : array, shape (h, w)
        Estimate of villain's face.
    B : array, shape (H, W)
        Estimate of background.
    s : float
        Estimate of standard deviation of Gaussian noise.
    A : array, shape (H-h+1, W-w+1)
        Estimate of prior on displacement of face in any image.
    """
    H, W, N = X.shape

    if not use_MAP:
        A = q.mean(axis=2)

        B = np.zeros((H, W))
        F = np.zeros((h, w))
        Bnorm = np.zeros((H, W))
        indctr = np.ones((H, W))
        for l in range(H - h + 1):
            for m in range(W - w + 1):
                tmp = (X * q[l, m][np.newaxis, np.newaxis, :]).sum(axis=2)
                F += tmp[l:l + h, m:m + w]
                tmp[l:l + h, m:m + w] = 0
                B += tmp
                tmp = np.ones((H, W)) * np.sum(q[l, m, :])
                tmp[l:l + h, m:m + w] = 0
                Bnorm += tmp 
        F /= N
        Bnorm[Bnorm == 0] = 1
        B /= Bnorm

        s = 0
        for l in range(H - h + 1):
            for m in range(W - w + 1):
                mean = B.copy()
                mean[l:l + h, m:m + w] = F.copy()
                s += np.sum(q[l, m][np.newaxis, np.newaxis, :] * (X - mean[:, :, np.newaxis]) ** 2)

        s /= N * W * H
        s = np.sqrt(s)
    else:
        A = np.zeros((H - h + 1, W - w + 1))
        for l, m in q.T:
            A[l, m] += 1
        A /= A.sum()

        F = np.zeros((h, w))
        for k, (l, m) in enumerate(q.T):
            F += X[l:l + h, m:m + w, k]
        F /= N

        B = np.zeros((H, W))
        Bnorm = np.zeros((H, W))
        mask = np.ones((H, W))
        for k, (l, m) in enumerate(q.T):
            mask[mask == 0] = 1
            mask[l:l + h, m:m + w] = 0
            B += X[:, :, k] * mask
            Bnorm += mask
        Bnorm[Bnorm == 0] = 1 
        B /= Bnorm

        s = 0
        for k, (l, m) in enumerate(q.T):
            mean = B.copy()
            mean[l:l + h, m:m + w] = F.copy()
            s += np.sum((X[:, :, k] - mean) ** 2)
        s /= N * W * H
        s = np.sqrt(s)

    return (F, B, s, A)


def run_EM(X, h, w, F=None, B=None, s=None, A=None, tolerance=0.001,
           max_iter=10, use_MAP=False):
    """
    Runs EM loop until the likelihood of observing X given current
    estimate of parameters is idempotent as defined by a fixed
    tolerance.

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    h : int
        Face mask height.
    w : int
        Face mask width.
    F : array, shape (h, w), optional
        Initial estimate of villain's face.
    B : array, shape (H, W), optional
        Initial estimate of background.
    s : float, optional
        Initial estimate of standard deviation of Gaussian noise.
    A : array, shape (H-h+1, W-w+1), optional
        Initial estimate of prior on displacement of face in any image.
    tolerance : float, optional
        Parameter for stopping criterion.
    max_iter  : int, optional
        Maximum number of iterations.
    use_MAP : bool, optional
        If true then after E-step we take only MAP estimates of displacement
        (dh,dw) of villain's face given image Xk.

    Returns
    -------
    F, B, s, A : trained parameters.
    LL : array, shape(number_of_iters + 2,)
        L(q,F,B,s,A) at initial guess, after each EM iteration and after
        final estimate of posteriors;
        number_of_iters is actual number of iterations that was done.
    """
    H, W, N = X.shape
    if F is None:
        F = np.random.randint(0, 255, size=(h, w))
    if B is None:
        B = np.random.randint(0, 255, size=(H, W))
    if s is None:   
        s = np.random.randint(1, 100)
    if A is None:
        A = np.random.rand(H - h + 1, W - w + 1)
        A /= A.sum()

    q = run_e_step(X, F, B, s, A, use_MAP)
    LL = [calculate_lower_bound(X, F, B, s, A, q, use_MAP)]
    for i in range(max_iter):
        
        F, B, s, A = run_m_step(X, q, h, w, use_MAP)
        
        LL += [calculate_lower_bound(X, F, B, s, A, q, use_MAP)]
        print (i + 1, "step", LL[i])

        q = run_e_step(X, F, B, s, A, use_MAP)

        if abs(LL[i + 1] - LL[i]) < tolerance:
            break
    LL += [calculate_lower_bound(X, F, B, s, A, q, use_MAP)]
    return F, B, s, A, LL


def run_EM_with_restarts(X, h, w, tolerance=0.001, max_iter=50, use_MAP=False,
                         n_restarts=10):
    """
    Restarts EM several times from different random initializations
    and stores the best estimate of the parameters as measured by
    the L(q,F,B,s,A).

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    h : int
        Face mask height.
    w : int
        Face mask width.
    tolerance, max_iter, use_MAP : optional parameters for EM.
    n_restarts : int
        Number of EM runs.

    Returns
    -------
    F : array,  shape (h, w)
        The best estimate of villain's face.
    B : array, shape (H, W)
        The best estimate of background.
    s : float
        The best estimate of standard deviation of Gaussian noise.
    A : array, shape (H-h+1, W-w+1)
        The best estimate of prior on displacement of face in any image.
    L : float
        The best L(q,F,B,s,A).
    """
    bestL = -np.inf
    for i in range(n_restarts):
        np.random.seed(i)
        H, W, N = X.shape
        B_init = np.random.randint(0, 255, size=(H, W))
        F_init = np.random.randint(0, 255, size=(h, w))
        s_init = np.random.randint(1, 120)
        A_init = np.random.rand(H - h + 1, W - w + 1)
        A_init /= A_init.sum()

        F, B, s, A, LL = run_EM(X, h, w, F=F_init, B=B_init , s=s_init, A=A_init, \
                                    tolerance=tolerance, max_iter=max_iter, use_MAP=use_MAP)
        
        if i == 0 or bestL < LL[-1]:
            bestL = LL[-1]
            bestF = F
            bestB = B
            bests = s
            bestA = A
            bestLL = LL
    return bestF, bestB, bests, bestA, bestL

