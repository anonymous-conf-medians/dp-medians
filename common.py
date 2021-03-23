import numpy as np
import math

delta = 10**(-6)

def compute_t(n, epsilon, beta, sensitivity, scale=None, gaussian=True, cdp=True):
    if gaussian:
        if scale == None:
            scale = gaussian_scale(epsilon, sensitivity, cdp=cdp) 
        t = scale * math.sqrt(2.0 * np.log(1.0/beta)) / n
    else: # laplace
        if scale == None:
            scale = laplace_scale(epsilon, sensitivity) 
        t = scale * np.log(1.0/beta) / n
    return t

def noisy_count(true_count, epsilon, sensitivity, scale=None, gaussian=True):
    if gaussian:
        if scale == None:
            scale = gaussian_scale(epsilon, sensitivity) 
        noisy_count = np.random.normal(true_count, scale, size=1)[0]
    else: # laplace
        if scale == None:
            scale = laplace_scale(epsilon, sensitivity) 
        noisy_count = np.random.laplace(true_count, scale, size=1)[0]
    return noisy_count

def laplace_scale(epsilon, sensitivity):
    return float(sensitivity) / epsilon

def gaussian_scale(epsilon, sensitivity, cdp=True):
    if cdp:
        # Adding N(0, sigma^2) noise satisfies GS/(2sigma^2)-zCDP
        # so N(0, (1/eps)^2) and GS=1 satisfies eps^2/2-zDP = rho-zCDP
        return float(sensitivity) / epsilon
    else:
        return float(sensitivity) * math.sqrt(2.0*np.log(1.26/delta)) / epsilon

def eps_to_rho(eps):
    rho = (eps**2)/2.0
    return rho

def rho_to_eps(rho):
    eps = math.sqrt(2.0*rho)
    return eps

def add_eps_cdp(eps_1, eps_2):
    rho_1 = eps_to_rho(eps_1)
    rho_2 = eps_to_rho(eps_2)
    return rho_to_eps(rho_1 + rho_2)

def subtract_eps_cdp(eps_1, eps_2):
    assert eps_1 >= eps_2
    rho_1 = eps_to_rho(eps_1)
    rho_2 = eps_to_rho(eps_2)
    return rho_to_eps(rho_1 - rho_2)

def divide_eps_cdp(eps, divisor):
    return eps/math.sqrt(divisor)
