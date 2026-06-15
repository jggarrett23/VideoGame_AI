import numpy as np
from scipy.integrate import quad


def optimal_SVHT_coef(beta, sigma_known):
    if sigma_known:
        return optimal_SVHT_coef_sigma_known(beta)
    else:
        return optimal_SVHT_coef_sigma_unknown(beta)


def optimal_SVHT_coef_sigma_known(beta):
    assert np.all(beta > 0)
    assert np.all(beta <= 1)
    assert len(beta.shape) == 1  # beta must be a vector

    w = (8 * beta) / (beta + 1 + np.sqrt(beta ** 2 + 14 * beta + 1))
    lambda_star = np.sqrt(2 * (beta + 1) + w)
    return lambda_star


def optimal_SVHT_coef_sigma_unknown(beta):
    assert np.all(beta > 0)
    assert np.all(beta <= 1)
    assert len(beta.shape) == 1  # beta must be a vector

    coef = optimal_SVHT_coef_sigma_known(beta)
    MPmedian = np.zeros_like(beta)

    for i in range(len(beta)):
        MPmedian[i] = MedianMarcenkoPastur(beta[i])

    omega = coef / np.sqrt(MPmedian)
    return omega


def MarcenkoPasturIntegral(x, beta):
    if beta <= 0 or beta > 1:
        raise ValueError('beta beyond')
    lobnd = (1 - np.sqrt(beta)) ** 2
    hibnd = (1 + np.sqrt(beta)) ** 2
    if (x < lobnd) or (x > hibnd):
        raise ValueError('x beyond')
    dens = lambda t: np.sqrt((hibnd - t) * (t - lobnd)) / (2 * np.pi * beta * t)
    I, _ = quad(dens, lobnd, x)
    print('x={:.3f}, beta={:.3f}, I={:.3f}'.format(x, beta, I))
    return I


def MedianMarcenkoPastur(beta):
    def MarPas(x):
        return 1 - incMarPas(x, beta, 0)

    lobnd = (1 - np.sqrt(beta)) ** 2
    hibnd = (1 + np.sqrt(beta)) ** 2
    change = True
    while change and (hibnd - lobnd > 0.001):
        change = False
        x = np.linspace(lobnd, hibnd, 5)
        y = np.array([MarPas(xi) for xi in x])
        if np.any(y < 0.5):
            lobnd = np.max(x[y < 0.5])
            change = True
        if np.any(y > 0.5):
            hibnd = np.min(x[y > 0.5])
            change = True
    med = (hibnd + lobnd) / 2
    return med


def incMarPas(x0, beta, gamma):
    if beta > 1:
        raise ValueError('betaBeyond')

    topSpec = (1 + np.sqrt(beta)) ** 2
    botSpec = (1 - np.sqrt(beta)) ** 2

    def MarPas(x):
        return np.where((topSpec - x) * (x - botSpec) > 0,
                        np.sqrt((topSpec - x) * (x - botSpec)) / (beta * x) / (2 * np.pi),
                        0)

    if gamma != 0:
        fun = lambda x: (x ** gamma * MarPas(x))
    else:
        fun = MarPas

    I, _ = quad(fun, x0, topSpec)
    return I
