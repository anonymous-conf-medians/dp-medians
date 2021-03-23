import wrapper_for_data as wrap
import analysis_for_data as analysis
import exponential_median as exp_median
import noise_addition_for_smooth_sens as ss_median
import binary_search_median as bisearch_median
import fancy_bs_median as fancy_bs
import cdf_median as cdf_median
import grad_descent_median as gd_median
import bs_cdf_median
import publicCI as pub
import imp
import numpy as np
import scipy.stats as st
import common

data_path = r'data'
res_path = r'results'
analysis_path = r'analysis'

def runAnalyzeAlg(dataset_name, name, dir_path, res_path, analysis_path, num_datasets, num_trials, rho, alpha, beta, lower_bound, upper_bound, true_median, 
	function_name, hyperparameters, rerun_algs):
	confidence_str = str(int( (1.0-alpha) * 100))
	if rerun_algs == True:
		# run CIs
		print("computing %s%% CIs" % confidence_str)
		wrap.main('%s_%s' % (name, confidence_str), dataset_name, function_name, 
	          hyperparameters, num_trials,
	          rho, lower_bound, upper_bound, alpha=alpha, beta=beta, dir_path=dir_path, res_path=res_path, num_datasets=num_datasets)
	# run analysis
	print("running analysis")
	analysis.computeConfidenceIntervals('%s_%s' % (name, confidence_str),  
                                      dataset_name, 'sizes', name, hyperparameters, true_median, res_path=res_path, analysis_path=analysis_path)


def callRunAnalyzeAlgs(dataset_name_list, name, dir_path, num_datasets, num_trials, num_params, param_string, n_list, rho_list, range_center_list, range_scale_list, 
	true_median_list, alpha_list, beta_list, quantile_list, granularity_list, function_name, hyperparameters, rerun_algs, start_param, gen_preprocess):
	# Create new hyperparameters for each param

	for i in range(start_param, num_params):
			# Copy to new hyperparameters
			new_hyperparameters = hyperparameters.copy()
			print("param", i, "of", num_params)
			if num_params > 1:
				new_name = name + '_' + param_string + '_' + str(i)
			else:
				new_name = name
			dataset_name = dataset_name_list[i]
			rho = rho_list[i]
			lower_bound = range_center_list[i] - range_scale_list[i]
			upper_bound = range_center_list[i] + range_scale_list[i]
			true_median = true_median_list[i]
			alpha = alpha_list[i]
			beta = beta_list[i] 
			quantile = quantile_list[i]
			granularity = granularity_list[i]
			n = int(n_list[i])

			new_hyperparameters['granularity'] = granularity
			new_hyperparameters['beta'] = beta 
			new_hyperparameters['alpha'] = alpha
			cdp = True
			new_hyperparameters['cdp'] = cdp

			if name in ['Expmedian', 'Expmedian_naive'] and gen_preprocess:
				em_granularity = hyperparameters['em_granularity'] if 'em_granularity' in hyperparameters else granularity
				em_eps = common.rho_to_eps(rho/2.0)
				em_lower_quantile, em_upper_quantile = exp_median.computeFancyPrivQuantiles(n, alpha, lower_bound, upper_bound, em_eps, em_granularity)
				new_hyperparameters['em_lower_quantile'] = em_lower_quantile
				new_hyperparameters['em_upper_quantile'] = em_upper_quantile

			if name in ['CDFmedian', 'CDFmedian_naive', 'CDFBSmedian']:
				var_savePath = '%s/%s_%s_%s_%s_%s_%s_%s_%s.npy' % (data_path, 'CDFmedian', str(n), str(lower_bound), str(upper_bound), str(granularity), str(rho), str(alpha), 'variances')
				a_lower_savePath = '%s/%s_%s_%s_%s_%s_%s_%s_%s.npy' % (data_path, 'CDFmedian', str(n), str(lower_bound), str(upper_bound), str(granularity), str(rho), str(alpha), 'a_lower')
				a_upper_savePath = '%s/%s_%s_%s_%s_%s_%s_%s_%s.npy' % (data_path, 'CDFmedian', str(n), str(lower_bound), str(upper_bound), str(granularity), str(rho), str(alpha), 'a_upper')
				if gen_preprocess:
					print("preprocessing")
					print("n", n, "lb, ub", lower_bound, upper_bound, "gran", granularity, "alpha", alpha, "rho", rho)
					cdf_eps = common.rho_to_eps(rho)
					a_list_lower, a_list_upper, variances = cdf_median.preProcessCDF(n, lower_bound, upper_bound, granularity, cdf_eps, alpha, cdp=cdp)
					# print("a_list_lower:", a_list_lower)

					print("done preprocessing")
					np.save(var_savePath, np.array(variances))
					np.save(a_lower_savePath, np.array(a_list_lower))
					np.save(a_upper_savePath, np.array(a_list_upper))
				else:
					variances = np.load(var_savePath, allow_pickle=True).tolist()
					a_list_lower = np.load(a_lower_savePath, allow_pickle=True).tolist()
					a_list_upper = np.load(a_upper_savePath, allow_pickle=True).tolist()
				new_hyperparameters['cdf_variances'] = variances
				new_hyperparameters['a_list_lower'] = a_list_lower
				new_hyperparameters['a_list_upper'] = a_list_upper

			print("n", n, "lb, ub", lower_bound, upper_bound, "gran", granularity, "alpha", alpha, "rho", rho)
			print("name, hyperparameters", new_name, new_hyperparameters)
			runAnalyzeAlg(dataset_name, new_name, dir_path, res_path, analysis_path, num_datasets, num_trials, rho, alpha, beta, lower_bound, upper_bound, true_median, 
				function_name, new_hyperparameters, rerun_algs)	

def isListOrArray(var):
	t = type(var)
	if (type(var) is list) or (type(var) is np.ndarray):
		return True
	else:
		return False

def gen_datasets(dataset_name_list, num_datasets, num_params, n_list, data_distribution, true_median_list, data_center_list, 
	data_scale_list, data_skew_list, dir_path):
	for i in range(num_params):
		dataset_name = dataset_name_list[i]
		n = int(n_list[i])
		data_center = data_center_list[i]
		data_scale = data_scale_list[i]
		data_skew = data_skew_list[i]

		for j in range(num_datasets):
			if data_distribution == 'normal':
				dataset = np.array(st.norm.rvs(loc=data_center, scale=data_scale, size=n))
				true_median_list.append(data_center)
			elif data_distribution == 'skewnormal':
				dataset = np.array(st.skewnorm.rvs(loc=data_center, scale=data_scale, a=data_skew, size=n))
				true_median_list.append(st.skewnorm.median(data_skew, loc=data_center, scale=data_scale))
			else: # default is 'lognormal'
				dataset = np.array(st.lognorm.rvs(data_scale, loc=data_center, size=n))
				true_median_list.append(st.lognorm.median(data_scale, loc=data_center))
			save_path = '%s/%s_%s.npy' % (dir_path, dataset_name, str(j))
			np.save(save_path, dataset)

def runCIAlgs(dataset_name, num_trials=2, num_params=1, param_string='', rho=1, range_center=5, range_scale=10, alpha=0.05, beta=0.01, quantile=0.5,
	em_granularity=0.01, granularity=0.05, alg_list=['Expmedian', 'CDFmedian', 'CDFBSmedian', 'AdaBSCDFmedian', 'BSmedian_sep'], dir_path=data_path,
	n=300, true_median=0.0, data_distribution='lognormal', data_center=0, data_scale=1, data_skew=0, num_datasets=100, start_param=0, gen_data=False, gen_preprocess=False,
	rerun_algs=True):
	# Convert all relevant (hyper)parameters to lists
	rho_list = rho if isListOrArray(rho) else [rho]*num_params
	range_center_list = range_center if isListOrArray(range_center) else [range_center]*num_params
	range_scale_list = range_scale if isListOrArray(range_scale) else [range_scale]*num_params
	alpha_list = alpha if isListOrArray(alpha) else [alpha]*num_params
	beta_list = beta if isListOrArray(beta) else [beta]*num_params
	quantile_list = quantile if isListOrArray(quantile) else [quantile]*num_params
	granularity_list = granularity if isListOrArray(granularity) else [granularity]*num_params
	dataset_name_list = [dataset_name]*num_params
	true_median_list = true_median if isListOrArray(true_median) else [true_median]*num_params
	n_list = n if isListOrArray(n) else [n]*num_params


	# Generate datasets
	if gen_data:
		print("generating datasets")
		dataset_name_list = [dataset_name+ '_' + param_string + '_' + str(i) for i in range(num_params)]
		data_center_list = data_center if isListOrArray(data_center) else [data_center]*num_params
		data_scale_list = data_scale if isListOrArray(data_scale) else [data_scale]*num_params
		data_skew_list = data_skew if isListOrArray(data_skew) else [data_skew]*num_params
		true_median_list = []
		gen_datasets(dataset_name_list, num_datasets, num_params, n_list, data_distribution, true_median_list, 
			data_center_list, data_scale_list, data_skew_list, dir_path)

	# # Exp Mech
	name = 'Expmedian'
	if name in alg_list:
		print("starting EM")
		hyperparameters = {'em_granularity': em_granularity} # Include extra hyperparameters beyond common ones
		function_name = exp_median.dpCIsExp
		callRunAnalyzeAlgs(dataset_name_list, name, dir_path, num_datasets, num_trials, num_params, param_string, n_list, rho_list, range_center_list, range_scale_list, 
			true_median_list, alpha_list, beta_list, quantile_list, granularity_list, function_name, hyperparameters, rerun_algs, start_param, gen_preprocess)

	# # Exp Mech
	name = 'Expmedian_naive'
	if name in alg_list:
		print("starting EM")
		hyperparameters = {'em_granularity': em_granularity, 'naive': True} # Include extra hyperparameters beyond common ones
		function_name = exp_median.dpCIsExp
		callRunAnalyzeAlgs(dataset_name_list, name, dir_path, num_datasets, num_trials, num_params, param_string, n_list, rho_list, range_center_list, range_scale_list, 
			true_median_list, alpha_list, beta_list, quantile_list, granularity_list, function_name, hyperparameters, rerun_algs, start_param, gen_preprocess)

	# # CDF
	name = 'CDFmedian'
	if name in alg_list:
		print("starting CDF")
		hyperparameters = {} # Include extra hyperparameters beyond common ones
		function_name = cdf_median.dpCIsCDF
		callRunAnalyzeAlgs(dataset_name_list, name, dir_path, num_datasets, num_trials, num_params, param_string, n_list, rho_list, range_center_list, range_scale_list, 
			true_median_list, alpha_list, beta_list, quantile_list, granularity_list, function_name, hyperparameters, rerun_algs, start_param, gen_preprocess)

	# # CDF
	name = 'CDFmedian_naive'
	if name in alg_list:
		print("starting CDF naive")
		hyperparameters = {'naive': True} # Include extra hyperparameters beyond common ones
		function_name = cdf_median.dpCIsCDF
		callRunAnalyzeAlgs(dataset_name_list, name, dir_path, num_datasets, num_trials, num_params, param_string, n_list, rho_list, range_center_list, range_scale_list, 
			true_median_list, alpha_list, beta_list, quantile_list, granularity_list, function_name, hyperparameters, rerun_algs, start_param, gen_preprocess)

	# # CDF with BS
	name = 'CDFBSmedian'
	if name in alg_list:
		print("starting CDF with BS")
		hyperparameters = {'bs':True} # Include extra hyperparameters beyond common ones
		function_name = cdf_median.dpCIsCDF
		callRunAnalyzeAlgs(dataset_name_list, name, dir_path, num_datasets, num_trials, num_params, param_string, n_list, rho_list, range_center_list, range_scale_list, 
			true_median_list, alpha_list, beta_list, quantile_list, granularity_list, function_name, hyperparameters, rerun_algs, start_param, gen_preprocess)
	# # BS two-shot
	name = 'BSmedian_sep'
	if name in alg_list:
		print("starting BS two-shot")
		hyperparameters = {'separate_runs':True} # Include extra hyperparameters beyond common ones
		function_name = bisearch_median.dpCIsBS
		callRunAnalyzeAlgs(dataset_name_list, name, dir_path, num_datasets, num_trials, num_params, param_string, n_list, rho_list, range_center_list, range_scale_list, 
			true_median_list, alpha_list, beta_list, quantile_list, granularity_list, function_name, hyperparameters, rerun_algs, start_param, gen_preprocess)

	# # # BS two-shot reuse queries
	name = 'BSmedian_sep_reuse_queries'
	if name in alg_list:
		print("starting BS two-shot reuse queries")
		hyperparameters = {'separate_runs':True, 'reuse_queries':True} # Include extra hyperparameters beyond common ones
		function_name = bisearch_median.dpCIsBS
		callRunAnalyzeAlgs(dataset_name_list, name, dir_path, num_datasets, num_trials, num_params, param_string, n_list, rho_list, range_center_list, range_scale_list, 
			true_median_list, alpha_list, beta_list, quantile_list, granularity_list, function_name, hyperparameters, rerun_algs, start_param, gen_preprocess)

	# # # Adaptive BS two-shot
	name = 'AdaBSmedian_sep'
	if name in alg_list:
		print("starting Adaptive BS two-shot")
		hyperparameters = {'separate_runs':True, 'adaptive':True} # Include extra hyperparameters beyond common ones
		function_name = bisearch_median.dpCIsBS
		callRunAnalyzeAlgs(dataset_name_list, name, dir_path, num_datasets, num_trials, num_params, param_string, n_list, rho_list, range_center_list, range_scale_list, 
			true_median_list, alpha_list, beta_list, quantile_list, granularity_list, function_name, hyperparameters, rerun_algs, start_param, gen_preprocess)

	# # # Adaptive BS CDF 
	name = 'AdaBSCDFmedian_naive'
	if name in alg_list:
		print("starting Adaptive BS CDF naive")
		hyperparameters = {"naive": True} # Include extra hyperparameters beyond common ones
		function_name = bs_cdf_median.dpCIsAdaBSCDF
		callRunAnalyzeAlgs(dataset_name_list, name, dir_path, num_datasets, num_trials, num_params, param_string, n_list, rho_list, range_center_list, range_scale_list, 
			true_median_list, alpha_list, beta_list, quantile_list, granularity_list, function_name, hyperparameters, rerun_algs, start_param, gen_preprocess)

	# # # Adaptive BS CDF 
	name = 'AdaBSCDFmedian'
	if name in alg_list:
		print("starting Adaptive BS CDF")
		hyperparameters = {} # Include extra hyperparameters beyond common ones
		function_name = bs_cdf_median.dpCIsAdaBSCDF
		callRunAnalyzeAlgs(dataset_name_list, name, dir_path, num_datasets, num_trials, num_params, param_string, n_list, rho_list, range_center_list, range_scale_list, 
			true_median_list, alpha_list, beta_list, quantile_list, granularity_list, function_name, hyperparameters, rerun_algs, start_param, gen_preprocess)

	print("finished!")

