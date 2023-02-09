import numpy as np
import math
import common

def computeEpsPerStep(eps_per_run, num_steps, step, cdp):
    # Scale epsilon linearly
    total_divisions = num_steps*(num_steps + 1.0)/2.0
    if cdp:
        eps_per_step = eps_per_run * math.sqrt(step / total_divisions)
    else:
        eps_per_step = eps_per_run * step / total_divisions
    return eps_per_step

def computeBetaPerStep(epsilon, current_eps, beta, cdp):
    # Compute share of beta based on share of epsilon
    if cdp:
        current_beta = (common.eps_to_rho(current_eps)/common.eps_to_rho(epsilon)) * beta
    else:
        current_beta = (current_eps/epsilon)*beta
    return current_beta

def nonAdaptiveBS(x, lower_bound, upper_bound, n, num_steps, epsilon, beta, target_quantile, noisy_start=False, reuse_queries=False, 
    prev_noisy_counts=None, gaussian=True, cdp=True):
    eps_per_step = float(epsilon)/math.sqrt(num_steps) if cdp else float(epsilon)/num_steps
    beta_per_step = float(beta)/num_steps
    y = np.clip(x,lower_bound,upper_bound) # note this should happen already in the wrapper function but just in case, adding a second clip.
    num_points = len(y)
    target_count = np.floor(target_quantile*num_points)
    # Initialize variables
    i = 0
    previous_count = 0
    interval_range = upper_bound-lower_bound
    interval = (lower_bound, upper_bound)
    noisy_counts = []
    reused_counts = []
    remaining_eps = epsilon
    remaining_beta = beta
    bs_sensitivity = 1.0 # Sensitivity is 1 because we only change/release one count
    while (i < num_steps and remaining_eps > eps_per_step and remaining_beta > beta_per_step):
    # while (i < num_steps):
        # Choose split point as middle of interval
        split_point = (interval[0] + interval[1]) / 2.0
        # Compute true and noisy count of left half of interval
        true_interval_count = sum(float(point) <= split_point for point in y)
        # Initialize
        reused_current = False
        # Reuse queries if possible
        if reuse_queries and prev_noisy_counts != None: 
            matches = [query for query in prev_noisy_counts if query[0] == split_point]
            if len(matches) > 0:
                noisy_count = matches[0][1]
                current_t = matches[0][2]
                reused_current = True
                reused_counts.append((split_point, noisy_count, current_t))
        # Otherwise, compute fresh noisy count
        if reused_current == False:
            # Compute current epsilon and beta
            if noisy_start: # Linearly increasing epsilon and beta
                current_eps = min(remaining_eps, computeEpsPerStep(epsilon, num_steps, i+1, cdp))
                current_beta = min(remaining_beta, computeBetaPerStep(epsilon, current_eps, beta, cdp))
            else: 
                num_steps_left = float(num_steps-i)
                current_eps = common.divide_eps_cdp(remaining_eps, num_steps_left) if cdp else remaining_eps/num_steps_left
                current_beta = remaining_beta / num_steps_left 
            # current_eps = eps_per_step
            # current_beta = beta_per_step
            # Compute current t
            current_t = common.compute_t(n, current_eps, current_beta, bs_sensitivity, gaussian=gaussian, cdp=cdp) 
            # Compute new count
            noisy_interval_count = common.noisy_count(true_interval_count, current_eps, bs_sensitivity, gaussian=gaussian)
            noisy_count = previous_count + noisy_interval_count
            # Subtract from remaining eps since we did not reuse a query
            remaining_eps = common.subtract_eps_cdp(remaining_eps, current_eps) if cdp else remaining_eps - current_eps
            remaining_beta = remaining_beta - current_beta
        noisy_counts.append((split_point, noisy_count, current_t))
        # Choose left or right half of interval based on noisy count 
        if noisy_count < target_count:
            # interval_range = interval[1]-split_point
            interval = (split_point, interval[1])
            previous_count += true_interval_count
        else:
            # interval_range = split_point-interval[0]
            interval = (interval[0], split_point)
        # Update for next iteration
        y = [point for point in y if (interval[0] <= point <= interval[1])]
        i += 1
    # print("remaining eps,", remaining_eps, "remaining beta", remaining_beta)
    # Save reused counts
    if reuse_queries:
        noisy_counts.append(reused_counts)
    # Return midpoint of resulting interval, or noisy counts
    res = (interval[0] + interval[1])/2.0
    return res, noisy_counts

def adaptiveBS(x, lower_bound, upper_bound, eps, n, num_nonadaptive_steps, beta, target_quantile, other_quantile, reuse_queries=False, prev_noisy_counts=None, 
    gaussian=True, cdp=True):
    # Currently only works for gaussian noise and CDP
    i = 0
    previous_count = 0
    noisy_counts = []
    reused_counts = []
    interval_range = upper_bound-lower_bound
    interval = (lower_bound, upper_bound, interval_range)
    y = np.clip(x,lower_bound,upper_bound) # note this should happen already in the wrapper function but just in case, adding a second clip.
    eps_used = 0.0
    beta_used = 0.0
    nonadaptive_eps_per_step = float(eps)/math.sqrt(num_nonadaptive_steps) if cdp else float(eps)/num_nonadaptive_steps
    initial_eps = nonadaptive_eps_per_step/10.0 # first try first est
    eps_first_try = initial_eps # first try per est
    step_eps = 0.0
    threshold_num_ests = 8
    # print("total eps for this function:", eps)
    bs_sensitivity = 1.0
    while (common.add_eps_cdp(eps_used, eps_first_try) < eps):
        # Choose split point as middle of interval
        split_point = interval[0] + (interval[2] / 2)
        # print("split point, i:", split_point, i)
        # Compute true and noisy count of left half of interval
        true_interval_count = sum(float(point) <= split_point for point in y)
        # Initialize
        eps_used_this_est = 0.0
        beta_used_this_est = 0.0
        reused_current = False
        # Reuse queries if possible
        if reuse_queries and prev_noisy_counts != None: 
            matches = [query for query in prev_noisy_counts if query[0] == split_point]
            if len(matches) > 0:
                noisy_count = matches[0][1]
                current_t = matches[0][2]
                reused_current = True
                reused_counts.append((split_point, noisy_count, current_t))
        # Otherwise, compute fresh noisy count
        if reused_current == False:
            # Keep getting estimates until they are good enough, we run out of epsilon, or we hit threshold number of queries at this point
            ests = []
            error_variances = []
            noisy_interval_count = 0.0
            current_t = 0.0
            current_eps = eps_first_try 
            current_beta = computeBetaPerStep(eps, current_eps, beta, cdp) # Set beta according to share of eps used
            est_good_enough = False
            while (est_good_enough == False and len(ests) < threshold_num_ests and common.add_eps_cdp(eps_used, current_eps) <= eps):
                # Compute new est and update eps and beta used this est
                ests.append(common.noisy_count(true_interval_count, current_eps, bs_sensitivity, gaussian=gaussian))
                error_variances.append((common.gaussian_scale(current_eps, bs_sensitivity, cdp=cdp))**2)
                eps_used_this_est = common.add_eps_cdp(eps_used_this_est, current_eps)
                beta_used_this_est += current_beta
                # Compute average of ests and corresponding error
                noisy_interval_count = np.mean(ests) 
                noisy_count = previous_count + noisy_interval_count
                avg_scale = (1.0/len(ests))*math.sqrt(np.sum(error_variances))
                # Compute t by passing in scale (no need to specify epsilon) and beta allocated to this est so far
                current_t = common.compute_t(n, None, beta_used_this_est, bs_sensitivity, scale=avg_scale, gaussian=gaussian, cdp=cdp)
                # print("split point", split_point, "est #", len(ests), "noisy q", noisy_count/n, "current t", current_t)
                # Count needs to be accurate enough for both quantiles; otherwise, can mess up the conservative post-processing
                if (((noisy_count - n*current_t > n*target_quantile) or (noisy_count + n*current_t < n*target_quantile)) and
                    ((noisy_count - n*current_t > n*other_quantile) or (noisy_count + n*current_t < n*other_quantile))):
                    est_good_enough = True
                # Update current eps and beta
                current_eps = common.add_eps_cdp(current_eps, step_eps) # in case we want to allocate more epsilon to later attempts at same query point
                current_beta = computeBetaPerStep(eps, current_eps, beta, cdp) # Set beta according to share of eps used
        # Add noisy count
        noisy_counts.append((split_point, noisy_count, current_t))
        # Choose left or right half of interval based on noisy count 
        if noisy_count < n*target_quantile:
            interval_range = interval[1]-split_point
            interval = (split_point, interval[1], interval_range)
            previous_count += true_interval_count
        else:
            interval_range = split_point-interval[0]
            interval = (interval[0], split_point, interval_range)
        # Update for next iteration
        y = [point for point in y if (interval[0] <= point <= interval[1])]
        i += 1
        eps_used = common.add_eps_cdp(eps_used, eps_used_this_est)
        beta_used += beta_used_this_est
        eps_first_try = initial_eps * (i+1) # allocate more epsilon to later query points

    # print("eps, eps_used:", eps, eps_used, "beta, beta_used:", beta, beta_used)

    # Return midpoint of resulting interval and noisy counts
    res = interval[0] + (interval[2]/2.0)
    # Save reused counts
    if reuse_queries:
        noisy_counts.append(reused_counts)
    return res, noisy_counts

# DP Median Binary Search point estimator with wrapper matching wrapper_for_data.py.
def dpMedianBinarySearch(x, lower_bound, upper_bound, epsilon, hyperparameters, num_trials):
    """
    :param x: List or numpy array of real numbers
    :param lower_bound: Lower bound on values in x. 
    :param upper_bound: Upper bound on values in x.
    :param epsilon: Desired value of epsilon.
    :param hyperparameters: dictionary that contains:
        :param granularity: This parameter helps avoid pathological behavior when values are tightly concentrated (e.g all equal). 
                        It is a lower bound on the algorithm's accuracy. Defaults to 0.001
    :param num_trials: Number of fresh DP median estimates to generate using dataset x.
    :return: List of 1/2 eps-DP estimates for the median of x.
    """
    # get hyperparameters and check validity 
    min_granularity = 0.001 # to avoid divide by zero errors
    granularity = hyperparameters['granularity'] if ('granularity' in hyperparameters) else min_granularity
    assert lower_bound <= upper_bound
    assert granularity >= min_granularity

    default_quantile = 0.5
    quantile = hyperparameters['quantile'] if ('quantile' in hyperparameters) else default_quantile
    assert 0 <= quantile <= 1 
    cdp = hyperparameters['cdp'] if ('cdp' in hyperparameters) else False
    noisy_start = hyperparameters['noisy_start'] if ('noisy_start' in hyperparameters) else False
    beta = hyperparameters['beta'] if ('beta' in hyperparameters) else None
    gaussian = hyperparameters['gaussian'] if ('gaussian' in hyperparameters) else True
    adaptive = hyperparameters['adaptive'] if ('adaptive' in hyperparameters) else False

    n = len(x)
    num_steps = math.floor(math.log2(float(interval_range)/granularity)) # depth of tree  
    results = []
    for j in range(num_trials):
        if adaptive:
            noisy_quantile, _ = adaptiveBS(x, lower_bound, upper_bound, epsilon, n, num_steps, beta, target_quantile=quantile, other_quantile=quantile,
                gaussian=gaussian, cdp=cdp)
        else:
            noisy_quantile, _ = nonAdaptiveBS(x, lower_bound, upper_bound, n, num_steps, epsilon, beta, target_quantile, noisy_start=noisy_start, 
                gaussian=gaussian, cdp=cdp)
        res = noisy_quantile  
        results.append(res)
    return results

def bsConsPostprocess(vals, noisy_cdf, n, lower_bound, upper_bound, quantile, lower=False, upper=False):
    """
    Conservative postprocessing of noisy counts; get lower and upper bounds of CI (that are below/above the threshold quantiles) 
    :param quantile: threshold quantile
    :param lower: True if we want the lower bound of the CI
    :param upper: True if we want the upper bound of the CI
    """
    distances = np.array([(x-quantile) for x in noisy_cdf])
    # print("distances:", distances)
    if lower: 
    # Return the val before the first val where noisy cdf is >= threshold quantile
        indices = np.where(distances >= 0)[0]
        # print("indices:", indices)
        if len(indices) > 0 and np.min(indices) > 0:
            i = np.min(indices) - 1
            val = vals[i]
        else:
            val = lower_bound
    else: 
    # Return the val after the last val where noisy cdf is <= threshold quantile
        indices = np.where(distances <= 0)[0]
        if len(indices) > 0 and np.max(indices) < len(vals) - 1:
            i = np.max(indices) + 1
            val = vals[i]
        else:
            val = upper_bound
    return val

def dpCIsBS(x, lower_bound, upper_bound, epsilon, hyperparameters, num_trials):
    """
    :param x: List or numpy array of real numbers
    :param lower_bound: Lower bound on values in x. 
    :param upper_bound: Upper bound on values in x.
    :param epsilon: Desired value of epsilon.
    :param hyperparameters: dictionary that contains:
        :param granularity: This parameter helps avoid pathological behavior when values are tightly concentrated (e.g all equal). 
                        It is a lower bound on the algorithm's accuracy. Defaults to 0.001
        :param cdp: Boolean, true if DP definition in use is zCDP.
        param monotonic: Boolean. If True, will force counts to be monotonic before calculating CI. Not included for now.
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
    save_path = hyperparameters['save_path'] if ('save_path' in hyperparameters) else None
    cdp = hyperparameters['cdp'] if ('cdp' in hyperparameters) else True
    gaussian = hyperparameters['gaussian'] if ('gaussian' in hyperparameters) else True
    noisy_start = hyperparameters['noisy_start'] if ('noisy_start' in hyperparameters) else False
    adaptive = hyperparameters['adaptive'] if ('adaptive' in hyperparameters) else False

    n = len(x)
    interval_range = upper_bound-lower_bound
    interval = (lower_bound, upper_bound, interval_range)
    num_steps = int(math.log2(float(interval_range)/granularity)) # depth of tree

    # Calculate the epsilon and beta that will be distributed by the binary search call
    if separate_runs:
        if cdp:
            eps_per_run = epsilon/math.sqrt(2.0)
        else:
            eps_per_run = epsilon/2.0
        beta_per_run = beta/2.0
    else:
        eps_per_run = epsilon
        beta_per_run = beta

    # Calculate the epsilon and beta that will be used per step of the binary search call
    if cdp:
        eps_per_step = eps_per_run/math.sqrt(num_steps)
    else:
        eps_per_step = eps_per_run/num_steps
    beta_per_step = beta_per_run/num_steps


    # Calculate t, aka how much to adjust the quantiles by. 
    assert n is not None and beta is not None
    bs_sensitivity = 1.0
    t = common.compute_t(n, eps_per_step, beta_per_step, bs_sensitivity, gaussian=gaussian, cdp=cdp) 
    # print("t:", t, "n:", n, "num steps:", num_steps, "eps_per_run:", eps_per_run, "eps_per_step:", eps_per_step, "beta_per_step:", beta_per_step)

    if separate_runs: # two-shot
        # Set save path (to visualize one run)
        index = save_path.find('sep')
        lower_save_path = save_path[:index] + 'lower_' + save_path[index:]
        upper_save_path = save_path[:index] + 'upper_' + save_path[index:]
        # Set shifted quantiles
        shifted_lower_quantile = lower_quantile-t
        shifted_upper_quantile = upper_quantile+t
        results = []
        for j in range(num_trials):
            # Search for lower and upper quantiles
            if adaptive:
                _, lower_noisy_counts = adaptiveBS(x, lower_bound, upper_bound, eps_per_run, n, num_steps, beta_per_run, 
                    target_quantile=shifted_lower_quantile, other_quantile=shifted_upper_quantile, gaussian=gaussian, cdp=cdp)
                _, upper_noisy_counts = adaptiveBS(x, lower_bound, upper_bound, eps_per_run, n, num_steps, beta_per_run, 
                    target_quantile=shifted_upper_quantile, other_quantile=shifted_lower_quantile, reuse_queries=True, prev_noisy_counts=lower_noisy_counts,
                    gaussian=gaussian, cdp=cdp)
            else:
                _, lower_noisy_counts = nonAdaptiveBS(x, lower_bound, upper_bound, n, num_steps, eps_per_run, beta_per_run, 
                    target_quantile=shifted_lower_quantile, noisy_start=noisy_start, gaussian=gaussian, cdp=cdp)
                _, upper_noisy_counts = nonAdaptiveBS(x, lower_bound, upper_bound, n, num_steps, eps_per_run, beta_per_run, 
                    target_quantile=shifted_upper_quantile, noisy_start=noisy_start, reuse_queries=True, prev_noisy_counts=lower_noisy_counts, 
                    gaussian=gaussian, cdp=cdp) 
            # Save reused counts TODO: Currently, must reuse queries - should allow for not reusing queries
            reused_counts = upper_noisy_counts[-1]
            upper_noisy_counts = upper_noisy_counts[:-1]
            # Apply conservative postprocessing
            noisy_counts = lower_noisy_counts.copy()
            noisy_counts.extend(upper_noisy_counts)
            noisy_counts.sort(key=lambda tup: tup[0])
            vals = [val for (val, noisy_count, t) in noisy_counts]
            noisy_cdf = [noisy_count/float(n) for (val, noisy_count, t) in noisy_counts]
            lower_cis = [(noisy_cdf[i] - noisy_counts[i][2]) for i in range(len(noisy_cdf))] 
            upper_cis = [(noisy_cdf[i] + noisy_counts[i][2]) for i in range(len(noisy_cdf))]
            lower_res = bsConsPostprocess(vals, upper_cis, n, lower_bound, upper_bound, quantile=lower_quantile, lower=True) # passing upper cis to lower and vice versa
            upper_res = bsConsPostprocess(vals, lower_cis, n, lower_bound, upper_bound, quantile=upper_quantile, upper=True)
            res = (lower_res, upper_res)
            results.append(res)

            # Save visualizations for trial 0
            if save_path != None and j == 0:
                num_lower_counts = len(lower_noisy_counts)
                num_upper_counts = len(upper_noisy_counts)
                lower_noisy_counts.append((num_lower_counts, t, shifted_lower_quantile, lower_quantile))
                lower_noisy_counts.append(lower_res)
                np.save(lower_save_path, np.array(lower_noisy_counts))
                upper_noisy_counts.append(reused_counts)
                upper_noisy_counts.append((num_upper_counts, t, shifted_upper_quantile, upper_quantile))
                upper_noisy_counts.append(upper_res)
                np.save(upper_save_path, np.array(upper_noisy_counts))
    else:
        # removed one-shot as we are not using it; see bs_unused.py
        results = []

    return results

# Testing:
# x = [0, 1, 2, 3, 4, 5, 6]
# lower_bound = 0
# upper_bound = 10
# epsilon = 1000
# hyperparameters = {'granularity': 0.1}
# num_trials = 1
# print(dpMedianBinarySearch(x, lower_bound, upper_bound, epsilon, hyperparameters, num_trials))


