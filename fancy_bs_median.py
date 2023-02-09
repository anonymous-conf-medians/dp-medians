import numpy as np 
import math

def findSplit(weights, quantile):
    """
    Finds the index of the weights such that the sum of all previous weights is less than
    or equal to 0.5 and the sum including the subsequent weight is greater than 0.5.
    There's probably a fun clever way of doing this in a single line in python but this is not that.
    Update 7/15: generalized to arbitrary quantile.

    :param weights: an array of n weights between 0 and 1, which should sum to 1.
    :return: int, the index at which the weights to the left sum to <= 0.5 and if next index included sum
    to > 0.5.
    """
    i, total = 0,0
    while i < len(weights):
        newTotal = total + weights[i]
        if total <= quantile and newTotal > quantile:
            return i - 1
        else:
            total = newTotal
            i += 1 
    return i - 1

def findProbability(epsilon, noisy_count, num_points, quantile):
    """
    Probability of getting signal for left side of that is consistent w noisy count for the Laplace mechanism.
    Update 7/15: generalized to arbitrary quantile.
    :param epsilon: privacy parameter used for noisy count
    :param noisy_count: noisy count via Laplace mechanism
    :param num_points: number of points in the data set
    :return: float that should be between 0 and 1.
    """
    target_count = np.floor(quantile*num_points)
    p =1.0 - (1/2)*math.exp(-epsilon*abs(noisy_count - target_count)) # from cdf of Laplace distribution
    if noisy_count >= target_count: # want to weigh left side of mid higher here
        return p
    else:                             # otherwise want to weigh right side of mid
        return 1.0-p


def findNormalizations(weights, split_point, prob):
    """
    Normalization terms for updating the weights at each step of the algorithm.
    Update 7/15: generalized to arbitrary quantile.
    :param weights: weights of each segment of the search space
    :param split_point: point at which weights to the left sum to 0.5 and weights to left plus one to right are
    greater than 0.5.
    :param prob: maximal probability of the noisy count being greater than n/2 rather than less than n/2.
    :return: [float, float]; the normalization terms for both sides of the split_point. all updated weights should sum to one when using this as normalization parameter.
    """
    left_sum = sum(weights[0:split_point+1])
    right_sum = 1.0 - left_sum
    if left_sum != 0 and right_sum != 0:
        return [prob/left_sum, (1 - prob)/right_sum]
    elif left_sum == 0:
        return [0.0, 1.0]
    else:
        return [1.0,0.0]

def dpMedianFancyBS(x, lower_bound, upper_bound, epsilon, hyperparameters, num_trials):
    """
    Fancy DP Median Binary Search that updates weights, with wrapper matching wrapper_for_data.py.

    :param x: List or numpy array of real numbers
    :param lower_bound: Lower bound on values in x. 
    :param upper_bound: Upper bound on values in x.
    :param epsilon: Desired value of epsilon.
    :param hyperparameters: dictionary that contains:
        :param granularity: This parameter helps avoid pathological behavior when values are tightly contrated (e.g all equal). 
                        It is a lower bound on the algorithm's accuracy. Defaults to 0.001.
    :param num_trials: Number of fresh DP median estimates to generate using dataset x.
    :return: List of 1/2 eps-DP estimates for the median of x.
    """
    # get hyperparameters and check validity 
    min_granularity = 0.001 # to avoid divide by zero errors
    granularity = hyperparameters['granularity'] if ('granularity' in hyperparameters) else min_granularity
    default_quantile = 0.5
    quantile = hyperparameters['quantile'] if ('quantile' in hyperparameters) else default_quantile
    assert lower_bound <= upper_bound
    assert granularity >= min_granularity
    assert 0 <= quantile <= 1

    interval_range = upper_bound-lower_bound
    num_bins = int(interval_range//granularity)
    num_steps = math.floor(math.log2(num_bins)) # depth of tree
    eps_per_step = float(epsilon)/num_steps

    num_points = len(x)

    # Initialize weights as uniform
    weights = np.full(num_bins, 1.0/num_bins)

    results = []
    for j in range(num_trials):
        i = 0
        previous_count = 0
        while (i < num_steps):
            # Choose split point as the value at which the weights sum to 0.5
            split_point = findSplit(weights, quantile)
            mid = lower_bound + split_point * granularity
            # Compute true and noisy count of left half of interval
            true_count = sum(float(point) <= mid for point in x)
            # Sensitivity is 1 because we are only changing/releasing one count
            noisy_count = true_count + np.random.laplace(0, 1.0/eps_per_step, size=1)[0]
            # Calculate proxy of probability of getting that value
            p = findProbability(eps_per_step, noisy_count, num_points, quantile)
            # Calculate normalization factors
            [alpha, beta] = findNormalizations(weights, split_point, p)
            # Update weights to the left
            weights[0:split_point+1] = weights[0:split_point+1]*alpha
            # Update weights to the right
            weights[split_point+1:] = weights[split_point+1:]*beta
            i += 1
        # Return interval value at spot where equal to 0.5
        noisy_quantile = lower_bound + split_point * granularity
        results.append(noisy_quantile)

    return results

### cdf for CIs
def fancybsCI(noisy_counts, n, lower_bound, upper_bound, quantile, lower=False, upper=False):
    """
    Pull median or best estimate from tree
    :param tree: Optimized tree, formatted as a list of arrays where the contents of the ith array in the list is the ith level of the tree.
    :param cdf: CDF of tree as output by postProcessedCDF
    """
    vals = [val for (val, noisy_count) in noisy_counts]
    noisy_cdf = [noisy_count/float(n) for (val, noisy_count) in noisy_counts]
    distances = np.array([(x-quantile) for x in noisy_cdf])
    if lower:
        indices = np.where(distances >= 0)[0]
        if len(indices) > 0 and np.min(indices) > 0:
            i = np.min(indices) - 1
            val = vals[i]
        else:
            val = lower_bound
    else:
        indices = np.where(distances <= 0)[0]
        if len(indices) > 0 and np.max(indices) < len(vals) - 1:
            i = np.max(indices) + 1
            val = vals[i]
        else:
            val = upper_bound
    return val

def dpCIsFancyBS(x, lower_bound, upper_bound, epsilon, hyperparameters, num_trials):
    """
    Fancy DP Median Binary Search that updates weights, with wrapper matching wrapper_for_data.py.

    :param x: List or numpy array of real numbers
    :param lower_bound: Lower bound on values in x. 
    :param upper_bound: Upper bound on values in x.
    :param epsilon: Desired value of epsilon.
    :param hyperparameters: dictionary that contains:
        :param granularity: This parameter helps avoid pathological behavior when values are tightly contrated (e.g all equal). 
                        It is a lower bound on the algorithm's accuracy. Defaults to 0.001.
    :param num_trials: Number of fresh DP median estimates to generate using dataset x.
    :return: List of 1/2 eps-DP estimates for the median of x.
    """
    # get hyperparameters and check validity 
    min_granularity = 0.001 # to avoid divide by zero errors
    granularity = hyperparameters['granularity'] if ('granularity' in hyperparameters) else min_granularity
    assert lower_bound <= upper_bound
    assert granularity >= min_granularity

    default_quantile = 0.5
    lower_quantile = hyperparameters['lower_quantile'] if ('lower_quantile' in hyperparameters) else default_quantile
    upper_quantile = hyperparameters['upper_quantile'] if ('upper_quantile' in hyperparameters) else default_quantile
    beta = hyperparameters['beta'] if ('beta' in hyperparameters) else None
    separate_runs = hyperparameters['separate_runs'] if ('separate_runs' in hyperparameters) else False

    n = len(x)
    interval_range = upper_bound-lower_bound
    num_bins = int(interval_range//granularity)
    num_steps = math.floor(math.log2(num_bins)) # depth of tree
    eps_per_step = float(epsilon)/num_steps

    num_points = len(x)

    # Initialize weights as uniform
    weights = np.full(num_bins, 1.0/num_bins)

    eps_t = eps_per_step/2.0 if separate_runs else eps_per_step

    if n is not None and beta is not None:
        t = 1.0/(n*eps_t) * np.log(1/(beta))
    else:
        t = 0
    lower_quantile = lower_quantile-t
    upper_quantile = upper_quantile+t
    #print("lower: ", lower_quantile, "upper: ", upper_quantile)

    if separate_runs:
        eps_per_est = epsilon/2.0
        hyperparameters['quantile'] = lower_quantile
        lower_results = dpMedianFancyBS(x, lower_bound, upper_bound, eps_per_est, hyperparameters, num_trials)
        hyperparameters['quantile'] = upper_quantile
        upper_results = dpMedianFancyBS(x, lower_bound, upper_bound, eps_per_est, hyperparameters, num_trials)
        results = [(lower_results[i], upper_results[i]) for i in range(num_trials)]

    else:
        results = [None]*num_trials
        for j in range(num_trials):
            i = 0
            previous_count = 0
            noisy_counts = []
            while (i < num_steps):
                # Choose split point as the value at which the weights sum to 0.5
                split_point = findSplit(weights)
                mid = lower_bound + split_point * granularity
                # Compute true and noisy count of left half of interval
                true_count = sum(float(point) <= mid for point in x)
                # Sensitivity is 1 because we are only changing/releasing one count
                noisy_count = true_count + np.random.laplace(0, 1.0/eps_per_step, size=1)[0]
                noisy_counts.append((mid, noisy_count))

                # Calculate proxy of probability of getting that value
                p = findProbability(eps_per_step, noisy_count, num_points)
                # Calculate normalization factors
                [alpha, beta] = findNormalizations(weights, split_point, p)
                # Update weights to the left
                weights[0:split_point+1] = weights[0:split_point+1]*alpha
                # Update weights to the right
                weights[split_point+1:] = weights[split_point+1:]*beta
                i += 1
            # Compute lower and upper bound
            #print("Noisy counts: ", noisy_counts)
            noisy_counts.sort(key=lambda tup: tup[0])
            unique_noisy_counts = []
            for tup in noisy_counts:         
                if tup not in unique_noisy_counts:   
                    unique_noisy_counts.append(tup)
            #print("Unique noisy counts: ", unique_noisy_counts)
            lower_res = fancybsCI(unique_noisy_counts, n, lower_bound, upper_bound, lower_quantile, lower=True)
            upper_res = fancybsCI(unique_noisy_counts, n, lower_bound, upper_bound, upper_quantile, upper=True)
            results[j] = (lower_res, upper_res)
            #print("Results for trial", j, " : ", lower_res, upper_res)

    return results