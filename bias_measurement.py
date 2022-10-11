import numpy as np
from scipy.stats import skewnorm, lognorm
import matplotlib.pyplot as plt
import analysis_for_data as analysis


# Pull confidence interval values from the data analysis and calculate the median
def median_from_CI(dataset_name, param_string, algorithm, i):
	private_medians = []
	file_suffix = 'sizes'
	path = 'analysis/%s_%s_%s_sizes_%s_%s_%s.npy' % (dataset_name, param_string, str(i), algorithm, param_string, str(i)) 

	# save_path = '%s/%s_%s_%s%s.npy' % (analysis_path, dataset_name, file_suffix, name, compare_str)
	# analysis/lognormal_bias_data_scale_0_sizes_Expmedian_data_scale_0.npy

	data = np.load(path, allow_pickle=True)
	private_vals = data[3]

	for i in range(len(private_vals)):
		for j in range(len(private_vals[i])):
			private_medians.append((private_vals[i][j][0] + private_vals[i][j][1])/2.0)

	nonprivate_vals = data[6]
	nonprivate_medians = []
	for i in range(len(nonprivate_vals)):
		nonprivate_medians.append((nonprivate_vals[i][0] + nonprivate_vals[i][1])/2.0)
	return private_medians, nonprivate_medians

# Calculate the true median of the distribution:
def population_median(skew, location, data_scale): #location is skewed norm dist param
	# true_median = skewnorm.median(a=skew, loc=location, scale=data_scale)
	true_median = lognorm.median(data_scale, loc=location)
	return true_median

def calculate_bias(dataset_name, algorithms, param_string, num_params, skews, locations, data_scales): #location is skewed norm dist param
	bias_for_mechanisms = []
	for j in range(len(algorithms)): #these are the different mechanisms
		alg = algorithms[j]
		priv_bias_for_mechanism = []
		nonpriv_bias_for_mechanism = []
		for i in range(num_params): # these are the different skew levels
			true_median = population_median(skew=skews[i], location=locations[i], data_scale=data_scales[i])
			private_medians, nonprivate_medians = median_from_CI(dataset_name, param_string, alg, i)
			priv_bias = np.mean([m - true_median for m in private_medians])
			priv_bias_for_mechanism.append(priv_bias)

			nonpriv_bias = np.mean([m - true_median for m in nonprivate_medians])
			nonpriv_bias_for_mechanism.append(nonpriv_bias)
		bias_for_mechanisms.append(priv_bias_for_mechanism)
		if j == len(algorithms)-1:
			bias_for_mechanisms.append(nonpriv_bias_for_mechanism)
	return bias_for_mechanisms

def plot_bias(dataset_name, algorithms, algorithm_names, param_string, num_params, skews, locations, data_scales, colors, styles, save=True):
	bias_for_mechanisms = calculate_bias(dataset_name, algorithms, param_string, num_params, skews, locations, data_scales)
	algorithm_names.append("Nonprivate nonparametric")

	xs = [data_scales for i in range(len(algorithm_names))]
	results = bias_for_mechanisms
	confidence_str = '95'
	title = 'Bias of CI algorithms as Median Estimates for Log-Normal Data'
	labels = algorithm_names
	xlabel = '' # 'Data scale parameter of Log-Normal Data'
	ylabel = '' # 'Average Bias of Median Estimated As Midpoint of CI'
	log = True
	xlog = False
	legend_out = True
	xlim = []
	ylim = []
	show_title = False
	save_path = 'plots/bias-plot-revised.pdf'
	line_plot = True
	alpha_lines = []
	box_plot = False
	blackwhite = True
	legend = True

	analysis.plot(xs, results, confidence_str, dataset_name, title, labels, xlabel, ylabel, log, xlog, legend_out, xlim, ylim, save, show_title=show_title, 
            save_path=save_path, line_plot=line_plot, alpha_lines=alpha_lines, box_plot=box_plot, marker_size=10, colors=colors, styles=styles, 
            blackwhite=blackwhite, legend=legend)



	# fig = plt.gcf()
	# fig.set_size_inches(10, 5)
	# for i in range(len(bias_for_mechanisms)):
	#  	plt.plot(data_scales, bias_for_mechanisms[i], label = algorithm_names[i])
	# plt.legend()
	# plt.xlabel('Data scale parameter of Log-Normal Data')
	# plt.ylabel('Average Bias of Median Estimated As Midpoint of CI')
	# plt.title('Bias of CI algorithms as Median Estimates for Log-Normal Data')
	# plt.savefig('plots/bias-plot.png')
	# plt.show()

