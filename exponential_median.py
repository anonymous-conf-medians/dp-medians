import numpy as np
import math
import publicCI as pub
import scipy

def computeFancyPrivQuantiles(n, alpha, lower_bound, upper_bound, epsilon, granularity):
    failure_probs = []
    data_range = upper_bound - lower_bound
    n = int(n)

    # Compute all the binomial/normal approx coefficients
    print("Computing coefficients")
    qm_probs = []
    for i in range(1, n+1):
        qm_prob = scipy.special.binom(n, i) * (0.5)**(i) * (0.5)**(n-i) if n < 800 else scipy.stats.norm.pdf(i, loc=n*0.5, scale=np.sqrt(n*0.25))
        qm_probs.append(qm_prob)
    print("Finished computing coefficients")

    print("n:", n)

    print("Searching for lower quantile")
    # LOWER
    for i in range(1, int(n/2.0)+1):
        if i % 500 == 0:
            print("i:", i)
        qtilde = float(i)/n
        first_terms = []
        for j in range(1, i+1):
            first_terms.append(qm_probs[j-1]*1.0)
        first_case = np.sum(first_terms)
        second_terms = []
        for j in range(i+1, n+1):
            qm = float(j)/n
            em_prob = min( (data_range - 2*granularity)/(2*granularity) * np.exp(-1*abs(qm-qtilde)*epsilon*n), 1.0)
            second_terms.append(qm_probs[j-1]*em_prob)
        second_case = np.sum(second_terms)
        res = first_case + second_case
        failure_probs.append(res)
    indices = np.where(np.array(failure_probs) <= alpha/2.0)[0]
    lower_priv_quantile = (max(indices)) /float(n) if len(indices) > 0 else 1.0/n

    print("Searching for upper quantile")
    # UPPER
    failure_probs = []
    for i in range(int(n/2.0) +1, n+1):
        if i % 500 == 0:
            print("i:", i)
        qtilde = float(i)/n
        first_terms = []
        for j in range(1, i+1):
            qm = float(j)/n
            em_prob = min( (data_range - 2*granularity)/(2*granularity) * np.exp(-1*abs(qtilde-qm)*epsilon*n), 1.0)
            first_terms.append(qm_probs[j-1]*em_prob)
        first_case = np.sum(first_terms)
        second_terms = []
        for j in range(i+1, n+1):
            second_terms.append(qm_probs[j-1]*1.0)
        second_case = np.sum(second_terms)
        res = first_case + second_case
        failure_probs.append(res)
    indices = np.where(np.array(failure_probs) <= alpha/2.0)[0]
    upper_priv_quantile = (int(n/2.0) + 1 + min(indices))/float(n) if len(indices) > 0 else float(n-1)/n

    return lower_priv_quantile, upper_priv_quantile


def computeFancyPrivQuantilesEfficient(n, alpha, lower_bound, upper_bound, epsilon, granularity, naive_lower=None, naive_upper=None):
    failure_probs = []
    data_range = upper_bound - lower_bound
    n = int(n)

    if naive_lower is None:
        naive_lower = 1
    else:
        naive_lower = int(np.floor(naive_lower))
    if naive_upper is None:
        naive_upper = n
    else:
        naive_upper = int(np.ceil(naive_upper))

    # Compute all the binomial/normal approx coefficients
    print("Computing coefficients")
    qm_probs = []
    qm_cum_probs = []
    for i in range(1, n+1):
        qm_prob = scipy.stats.binom.pmf(i, n, 0.5)
        qm_probs.append(qm_prob)
        qm_cum_prob = scipy.stats.binom.cdf(i, n, 0.5)
        qm_cum_probs.append(qm_cum_prob)

    print("Searching for lower quantile")
    # LOWER
    start_rank = max(1, naive_lower)
    for k in range(start_rank, int(n/2.0)+1):
        if k % 500 == 0:
            print("k:", k)
        first_case = qm_cum_probs[k-1]
        second_terms = []
        for m in range(k+1, n+1):
            em_prob = min( (data_range - 2*granularity)/(2*granularity) * np.exp(-1*abs(m-k)*epsilon), 1.0)
            second_terms.append(qm_probs[m-1]*em_prob)
        second_case = sum(second_terms)
        res = first_case + second_case
        failure_probs.append(res)

    indices = np.where(np.array(failure_probs) <= alpha/2.0)[0]
    lower_priv_quantile = ((start_rank - 1) + max(indices)) /float(n) if len(indices) > 0 else 1.0/n

    print("Searching for upper quantile")
    # UPPER
    end_rank = min(n, naive_upper) 
    failure_probs = []
    for k in range(int(n/2.0) +1, end_rank):
        if k % 500 == 0:
            print("k:", k)
        first_terms = []
        for m in range(1, k+1):
            em_prob = min( (data_range - 2*granularity)/(2*granularity) * np.exp(-1*abs(k-m)*epsilon), 1.0)
            first_terms.append(qm_probs[m-1]*em_prob)
        first_case = sum(first_terms)
        second_case = (1.0 - qm_cum_probs[k+1]) # may want to change to survival prob for better accuracy
        res = first_case + second_case
        failure_probs.append(res)
    indices = np.where(np.array(failure_probs) <= alpha/2.0)[0]
    upper_priv_quantile = (int(n/2.0) + 1 + min(indices))/float(n) if len(indices) > 0 else float(n-1)/n

    return lower_priv_quantile, upper_priv_quantile

# Computes 1/2 eps^2-CDP median for bounded data.
# This algorithm is the exponential mechanism with a "granularity" parameter around the true median.
def dpMedianExponential(x, lower_bound, upper_bound, epsilon, hyperparameters, num_trials, quantile=0.5):
    """
    :param x: List or numpy array of real numbers
    :param lower_bound: Lower bound on values in x.
    :param upper_bound: Upper bound on values in x.
    :param epsilon: Desired value of epsilon.
    :param hyperparameters: dictionary that contains:
        :param em_granularity: This parameter helps avoid pathological behavior when values are tightly concentrated (e.g all equal).
                        It is a lower bound on the algorithm's accuracy. Defaults to 0.
        :param quantile: Target quantile to estimate. Defaults to 0.5.
        :param beta: Value between 0 and 1 describing beta to meet for definition of (t, granularity)-goodness. May be None.
        :param add_t: Boolean. If True, will add t to the quantile. If False, will subtract. May be None.
    :param num_trials: Number of fresh DP median estimates to generate using dataset x.
    :return: List of 1/2 eps-DP estimates for the median of x.
    """
    # get hyperparameters and check validity
    min_granularity = 0.0
    granularity = hyperparameters['em_granularity'] if ('em_granularity' in hyperparameters) else min_granularity
    assert lower_bound <= upper_bound
    assert granularity > min_granularity
    assert 0.0 <= quantile <= 1.0

    z = np.concatenate([x, [lower_bound, upper_bound]])
    z.sort()
    n = len(z)

    # Now we clip the data to the interval specified by upper_bound and lower_bound,
    # and we put a buffer of width equal to 'granularity' around the quantile in the exponential mechanism.
    # We do this by moving all points granularity away from the true quantile.
    for i in range(min(n, math.floor(n*quantile)+1)):
        z[i] = max(lower_bound, z[i] - granularity)
    for i in range(math.floor(n*quantile)+1, n):
        z[i] = min(z[i] + granularity, upper_bound)

    # Iterate through z, assigning scores to each interval given by adjacent indices
    # currentMax and currentInt keep track of highest score and corresponding interval
    raw_scores = []
    scores = []
    for i in range(1, n):
        start = z[i-1]
        end = z[i]
        # Compute length of interval on logarithmic scale
        length = end-start
        if (length <= 0):
            loglength = -np.inf
        else:
            loglength = math.log(length)
        # The rungheight is the score of each individual point in the interval
        rungheight =  abs(i - n*quantile) 
        # The score has two components:
        # (1) Distance from index to median (closer -> higher score)
        # (2) Length of the interval on a logarithmic scale (larger -> higher score)
        # We include this since all values in the interval have the same score.
        score = -(epsilon/2) * rungheight + loglength
        scores.append(score)
        raw_scores.append(-(epsilon/2)*rungheight)
        
    results = []
    for j in range(num_trials):
        # Add Gumbel noise to sample from exponential mechanism
        # See https://pdfs.semanticscholar.org/2782/e47c5b0c8a2ce14eae8713b6f2db864f07c8.pdf
        noisy_score = np.array(scores) + np.random.gumbel(loc=0.0, scale=1.0, size=len(scores))
        max_score = np.argmax(noisy_score)
        results.append(np.random.uniform(low=z[max_score-1], high=z[max_score]))
    return results

def dpCIsExp(x, lower_bound, upper_bound, epsilon, hyperparameters, num_trials):
    min_granularity = 0.0
    granularity = hyperparameters['em_granularity'] if ('em_granularity' in hyperparameters) else min_granularity
    assert granularity > min_granularity
    alpha = hyperparameters['alpha'] if ('alpha' in hyperparameters) else None
    beta = hyperparameters['beta'] if ('beta' in hyperparameters) else None
    default_quantile = 0.5
    lower_quantile = hyperparameters['lower_quantile'] if ('lower_quantile' in hyperparameters) else None
    upper_quantile = hyperparameters['upper_quantile'] if ('upper_quantile' in hyperparameters) else None
    em_lower_quantile = hyperparameters['em_lower_quantile'] if ('em_lower_quantile' in hyperparameters) else None
    em_upper_quantile = hyperparameters['em_upper_quantile'] if ('em_upper_quantile' in hyperparameters) else None
    cdp = hyperparameters['cdp'] if ('cdp' in hyperparameters) else False
    naive = hyperparameters['naive'] if ('naive' in hyperparameters) else False
    assert lower_bound <= upper_bound
    n = len(x)

    # Calculate the epsilon that will be distributed by the exponential mechanism call
    if cdp:
        eps_per_run = epsilon/math.sqrt(2.0)
    else:
        eps_per_run = epsilon/2.0
    beta_per_run = beta/2.0

    # Calculate t aka how much to adjust the quantiles by.
    data_range = upper_bound - lower_bound
    t = 1.0/(n*eps_per_run) * np.log((data_range-2*granularity)/(2*granularity*beta_per_run))
    naive_lower_quantile = max(0.0, lower_quantile - t)
    naive_upper_quantile = min(1.0, upper_quantile + t)

    if (em_lower_quantile == None or em_upper_quantile == None) and naive == False:
        print("Computing fancy quantiles")
        naive_lower_rank = np.floor(n*naive_lower_quantile)
        naive_upper_rank = np.ceil(n*naive_upper_quantile)
        em_lower_quantile, em_upper_quantile = computeFancyPrivQuantilesEfficient(n, alpha, lower_bound, upper_bound, epsilon, granularity,
            naive_lower=naive_lower_rank, naive_upper=naive_upper_rank)

    if naive:
        target_lower_quantile = naive_lower_quantile
        target_upper_quantile = naive_upper_quantile 
    else:
        target_lower_quantile = em_lower_quantile
        target_upper_quantile = em_upper_quantile 

    lower_results = dpMedianExponential(x, lower_bound, upper_bound, eps_per_run, hyperparameters, num_trials, quantile=target_lower_quantile)
    upper_results = dpMedianExponential(x, lower_bound, upper_bound, eps_per_run, hyperparameters, num_trials, quantile=target_upper_quantile)
    results = [(lower_results[i]-granularity, upper_results[i]+granularity) for i in range(num_trials)]
    #print("Result: ", results[0])
    return results
