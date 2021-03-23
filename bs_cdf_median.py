import numpy as np
import math 
import binary_search_median as bisearch_median
import cdf_median as cdf_median
import publicCI as pub
import common

data_path = r'data'

### Wrapper function for CI
def dpCIsAdaBSCDF(x, lower_bound, upper_bound, epsilon, hyperparameters, num_trials):
    """
    :param x: List or numpy array of real numbers
    :param lower_bound: Lower bound on values in x. 
    :param upper_bound: Upper bound on values in x.
    :param epsilon: Desired value of epsilon.
    :param hyperparameters: dictionary that contains:
        :param depth: depth of the tree desired.
    :param num_trials: Number of fresh DP median estimates to generate using dataset x.
    :return: List of 1/2 eps-DP estimates for the median of x.
    """
    granularity = hyperparameters['granularity'] if ('granularity' in hyperparameters) else None
    default_quantile = 0.5
    lower_quantile = hyperparameters['lower_quantile'] if ('lower_quantile' in hyperparameters) else default_quantile
    upper_quantile = hyperparameters['upper_quantile'] if ('upper_quantile' in hyperparameters) else default_quantile
    beta = hyperparameters['beta'] if ('beta' in hyperparameters) else None
    alpha = hyperparameters['alpha'] if ('alpha' in hyperparameters) else None
    save_path = hyperparameters['save_path'] if ('save_path' in hyperparameters) else None
    cdp = hyperparameters['cdp'] if ('cdp' in hyperparameters) else True
    gaussian = hyperparameters['gaussian'] if ('gaussian' in hyperparameters) else True
    naive = hyperparameters['naive'] if ('naive' in hyperparameters) else False
    assert lower_bound <= upper_bound
    interval_range = upper_bound - lower_bound
    num_nonadaptive_steps = int(math.log2(float(interval_range)/granularity)) 
    n = len(x)

    # allocate privacy budget for estimating IQR using adaptive binary search and finishing with CDF
    eps_share_for_range = 0.25
    beta_share_for_range = 0.9 
    eps_for_range = epsilon*math.sqrt(eps_share_for_range) if cdp else epsilon*(eps_share_for_range)
    eps_for_range_per_run = eps_for_range/math.sqrt(2.0) if cdp else eps_for_range/2.0
    eps_for_cdf = epsilon*math.sqrt(1.0-eps_share_for_range) if cdp else epsilon*(1.0-eps_share_for_range)
    # print("total eps, eps for range, eps for cdf", epsilon, eps_for_range, eps_for_cdf)
    beta_for_range = beta*(beta_share_for_range)
    beta_for_range_per_run = beta_for_range/2.0
    beta_for_cdf = beta*(1.0-beta_share_for_range)
    
    results = [None]*num_trials
    for j in range(num_trials):

        # Estimate range using adaptive binary search
        bs_lower_target_quantile = min(0.25, lower_quantile-0.1)
        bs_upper_target_quantile = max(0.75, upper_quantile+0.1)
        bs_sensitivity = 1.0
        bs_t = common.compute_t(n, eps_for_range_per_run, beta_for_range_per_run, bs_sensitivity, gaussian=gaussian, cdp=cdp)
        bs_lower_shifted_target_quantile = bs_lower_target_quantile - bs_t
        bs_upper_shifted_target_quantile = bs_upper_target_quantile + bs_t
        lower_bs_res, bs_lower_noisy_counts = bisearch_median.adaptiveBS(x, lower_bound, upper_bound, eps_for_range_per_run, n, num_nonadaptive_steps, beta_for_range_per_run, 
            target_quantile=bs_lower_shifted_target_quantile, other_quantile=bs_upper_shifted_target_quantile, gaussian=gaussian, cdp=cdp)
        upper_bs_res, bs_upper_noisy_counts = bisearch_median.adaptiveBS(x, lower_bound, upper_bound, eps_for_range_per_run, n, num_nonadaptive_steps, beta_for_range_per_run, 
            target_quantile=bs_upper_shifted_target_quantile, other_quantile=bs_lower_shifted_target_quantile, 
            reuse_queries=True, prev_noisy_counts=bs_lower_noisy_counts, gaussian=gaussian, cdp=cdp)
        # Remove reused queries
        bs_upper_noisy_counts = bs_upper_noisy_counts[:-1]

        # Sort noisy counts by the value they were queried at and generate cdf
        bs_lower_noisy_counts.extend(bs_upper_noisy_counts)
        bs_lower_noisy_counts.sort(key=lambda tup: tup[0])
        bs_vals = [val for (val, noisy_count, t) in bs_lower_noisy_counts]
        bs_noisy_cdf = [noisy_count/float(n) for (val, noisy_count, t) in bs_lower_noisy_counts]

        bs_lower_cis = [(bs_noisy_cdf[i] - bs_lower_noisy_counts[i][2]) for i in range(len(bs_noisy_cdf))] 
        bs_upper_cis = [(bs_noisy_cdf[i] + bs_lower_noisy_counts[i][2]) for i in range(len(bs_noisy_cdf))]
        lower_bs_bound = bisearch_median.bsConsPostprocess(bs_vals, bs_upper_cis, n, lower_bound, upper_bound, quantile=bs_lower_target_quantile, lower=True) # passing upper cis to lower and vice versa
        upper_bs_bound = bisearch_median.bsConsPostprocess(bs_vals, bs_lower_cis, n, lower_bound, upper_bound, quantile=bs_upper_target_quantile, upper=True)
        # print("bs_upper_target_quantile, lower bs bound, upper bs bound:", bs_upper_target_quantile, lower_bs_bound, upper_bs_bound)
        bs_noisy_counts = [(val, noisy_count, t*n) for (val, noisy_count, t) in bs_lower_noisy_counts] # We should switch t to counts at some point... 

        # Make bounds line up with granularity (from 0.0)
        # print("BS bounds:", lower_bs_bound, upper_bs_bound)
        # print("BS noisy counts:", bs_noisy_counts)
        # print("granularity:", granularity)
        lower_bs_bound = int(lower_bs_bound / granularity)*float(granularity) # BUG here! This is wrong if bound is like -0.5
        upper_bs_bound = (int(upper_bs_bound // granularity)+1.0)*float(granularity)
        # print("new BS bounds:", lower_bs_bound, upper_bs_bound)


        # Clip data to estimated range
        x = np.clip(x, lower_bs_bound, upper_bs_bound)
        n_bins = (upper_bs_bound-lower_bs_bound)//granularity + 1
        depth = int(np.log2(n_bins)//1 + 1)
        # print("depth:", depth)
        max_eps_per_val = eps_for_cdf / math.sqrt(depth) if cdp else eps_for_cdf / depth
        cdf_sensitivity = 2.0
        beta_for_cdf_per_val = beta_for_cdf # Postprocessing analysis applies to pointwise bounds
        t = common.compute_t(n, max_eps_per_val, beta_for_cdf_per_val, cdf_sensitivity, gaussian=gaussian, cdp=cdp)
        tt = cdf_median.dpTree(x, len(x), lower_bs_bound, upper_bs_bound, eps_for_cdf, depth, cdp)
        tOpt, wA, wB = cdf_median.optimalPostProcess(tt, eps_for_cdf)
        tOpt = list(tOpt)
        noisy_cdf, _ = cdf_median.postProcessedCDF(tOpt, eps_for_cdf, monotonic=False)

        # Computations for new post-processing analysis
        alpha_for_cdf = alpha - beta_for_range
        var_savePath = '%s/%s_%s_%s_%s.npy' % (data_path, 'AdaBSCDFmedian', str(depth), str(eps_for_cdf), 'variances')
        a_lower_savePath = '%s/%s_%s_%s_%s_%s_%s.npy' % (data_path, 'AdaBSCDFmedian', str(n), str(depth), str(eps_for_cdf), str(alpha_for_cdf), 'a_lower')
        a_upper_savePath = '%s/%s_%s_%s_%s_%s_%s.npy' % (data_path, 'AdaBSCDFmedian', str(n), str(depth), str(eps_for_cdf), str(alpha_for_cdf), 'a_upper')
        import os.path

        if os.path.isfile(var_savePath):
            # print("variance file exists")
            variances = np.load(var_savePath, allow_pickle=True).tolist()
        else:
            print("variance file doesn't exist; lb, ub, depth:", lower_bs_bound, upper_bs_bound, depth)
            variances = cdf_median.getVariances(tOpt, eps_for_cdf, cdp=cdp)
            np.save(var_savePath, np.array(variances))

        if os.path.isfile(a_lower_savePath) and os.path.isfile(a_lower_savePath):
            # print ("a file exists")
            a_list_lower = np.load(a_lower_savePath, allow_pickle=True).tolist()
            a_list_upper = np.load(a_upper_savePath, allow_pickle=True).tolist()
        else:
            print("a file doesn't exist; lb, ub, depth:", lower_bs_bound, upper_bs_bound, depth)
            a_list_lower, a_list_upper, _ = cdf_median.preProcessCDF(n, lower_bs_bound, upper_bs_bound, granularity, eps_for_cdf, 
                alpha_for_cdf, cdp=cdp, variances=variances)
            np.save(a_lower_savePath, np.array(a_list_lower))
            np.save(a_upper_savePath, np.array(a_list_upper))

        # print("a_list_lower:", a_list_lower)
        # print("a_list_upper:", a_list_upper)

        ts = [common.compute_t(n, max_eps_per_val, beta_for_cdf_per_val, cdf_sensitivity, scale=math.sqrt(variances[i]), gaussian=gaussian) for i in range(len(variances))] # might be wrong for Laplace noise
        #print(noisy_cdf)
        vals = tOpt[-1][1]
        noisy_counts = [(vals[i], n*noisy_cdf[i], n*ts[i]) for i in range(len(vals))]

        lower_cis = [(noisy_cdf[i] - ts[i]) for i in range(len(noisy_cdf))] 
        upper_cis = [(noisy_cdf[i] + ts[i]) for i in range(len(noisy_cdf))]
        lower_naive_res = cdf_median.cdfConsPostprocess(vals, upper_cis, quantile=lower_quantile, lower=True) # we are passing upper bounds to lower and vice versa
        upper_naive_res = cdf_median.cdfConsPostprocess(vals, lower_cis, quantile=upper_quantile, upper=True)
        lower_fancy_res, upper_fancy_res = cdf_median.cdfFancyPostProcess(vals, noisy_cdf, a_list_lower, a_list_upper)
        # print("naive:", lower_naive_res, upper_naive_res)
        # print("fancy:", lower_fancy_res, upper_fancy_res)
        if naive:
            lower_res = lower_naive_res
            upper_res = upper_naive_res
        else:
            lower_res = lower_fancy_res
            upper_res = upper_fancy_res

        results[j] = (lower_res, upper_res)

        if save_path != None and j == 0:
            # Save noisy counts from adaptive binary search and cdf; just for plotting purposes
            noisy_counts.extend(bs_noisy_counts)
            noisy_counts.sort(key=lambda tup: tup[0])
            num_counts = len(noisy_counts)
            noisy_counts.append((num_counts, t, lower_quantile, upper_quantile))
            noisy_counts.append((lower_res, upper_res))
            np.save(save_path, np.array(noisy_counts))
    return results

# Testing
# x = np.arange(0, 100)
# n = len(x)
# lower_bound = -200
# upper_bound = 200
# epsilon = 5.0
# num_trials = 1
# lower_quantile, upper_quantile = pub.getConfIntervalQuantiles(n, 1-0.95)
# hyperparameters = {'granularity': 0.1, 'beta': 0.01, 'cdp': True, 'lower_quantile': lower_quantile, 'upper_quantile': upper_quantile}
# dpCIsAdaBSCDF(x, lower_bound, upper_bound, epsilon, hyperparameters, num_trials)


