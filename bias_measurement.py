import numpy as np
from scipy.stats import skewnorm
import matplotlib.pyplot as plt


# Pull confidence interval values from the data analysis and calculate the median
def median_from_CI(run_name, algorithm_file_name, dataset_name):
	private_medians = []
	path = 'analysis/%s_%s_sizes_%s_%s.npy' % (run_name, dataset_name, algorithm_file_name, dataset_name) #this is dumb but it's how they were run so hardcoding it in for now
	data = np.load(path, allow_pickle=True)
	private_vals = data[3]

	for i in range(0, len(private_vals)):
		for j in range(0, len(private_vals[i])):
			private_medians.append(private_vals[i][j][0] + (private_vals[i][j][1] - private_vals[i][j][0])/2.0)
	return private_medians

# Calculate the true median of the distribution:
def population_median(skew, location, data_scale): #location is skewed norm dist param
	true_median = skewnorm.median(a=skew, loc=location, scale=data_scale)
	return true_median

def calculate_bias(run_name, algorithm_file_names, dataset_names, location, skews, data_scale): #location is skewed norm dist param
	bias_for_mechanisms = []
	for alg in algorithm_file_names: #these are the different mechanisms
		bias_for_mechanism = []
		i = 0
		for dataset_name in dataset_names: # these are the different skew levels
			true_median = population_median(skews[i], location, data_scale=data_scale)
			private_medians = median_from_CI(run_name, alg, dataset_name)
			bias = np.mean([m - true_median for m in private_medians])
			bias_for_mechanism.append(bias)
			i += 1
		bias_for_mechanisms.append(bias_for_mechanism)
	return bias_for_mechanisms

def plot_bias(run_name, algorithm_names, algorithm_file_names, dataset_names, skews, location=0.5, data_scale=1.0):
	bias_for_mechanisms = calculate_bias(run_name, algorithm_file_names, dataset_names, location, skews, data_scale)
	fig = plt.gcf()
	fig.set_size_inches(10, 5)
	for i in range(len(bias_for_mechanisms)):
	 	plt.plot(skews, bias_for_mechanisms[i], label = algorithm_names[i])
	plt.legend()
	plt.xlabel('Skew Parameter of Log-Normal Data')
	plt.ylabel('Average Bias of Median Estimated As Midpoint of CI')
	plt.title('Bias of CI algorithms as Median Estimates for Log-Normal Data')
	plt.savefig('plots/bias-plot.png')
	plt.show()

