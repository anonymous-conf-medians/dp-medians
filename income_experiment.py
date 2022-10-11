import pandas as pd
import numpy as np
import math
import exponential_median as exp_median
import cdf_median 
import bs_cdf_median
import binary_search_median as binary_search_median
import wrapper_for_data as wrap
import common
import analysis_for_data as analysis_for_data
import publicCI as pub
import run_CI_algs
import matplotlib.pyplot as plt

dir_path = "../../07_income_data" 
res_path = "../../07_income_results" 
analysis_path = "../../07_income_analysis" 
figs_path = "../../figs/income-experiments"

jitter_dir_path = "../../jitter_income_data" 
jitter_res_path = "../../jitter_income_results" 
jitter_analysis_path = "../../jitter_income_analysis" 
jitter_figs_path = "../../figs/income-experiments"

# Categories and codes

# HHTYPE: Household type 
# Family (codes 1-married, 2-male, 3-female)
# Non-family (codes 4/5-male, 6/7-female)
hhtype_codes = [['1', '2', '3'], ['4', '5', '6','7']]
hhtype_detailed_codes = [['1'], ['2'], ['3'], ['4', '5'], ['6','7']]
hhtype_index =  4 
hhtype_descriptions = ['Family \n households', 'Non-family \n households']
hhtype_cat = ['hhtype', hhtype_index, hhtype_codes, hhtype_descriptions]

# RACE: 
# codes 1-white, 2-Black, 3-AIAN, 4/5/6-Asian
race_codes = [['1'], ['2'], ['3'], ['4','5','6']]
race_index = 16
race_descriptions = []
race_cat = ['race', race_index, race_codes]

# NATIVITY: 
# codes 1/2/3/4-native, 5-foreign
nativity_codes = [['0'], ['1', '2', '3', '4'], ['5']]
nativity_index = 20
nativity_descriptions = []
nativity_cat = ['nativity', nativity_index, nativity_codes, nativity_descriptions]

# REGION: 
# codes - 11/12/13-northeast, 21/22/23-midwest, 31/32/33/34-south, 41/42/43-west
region_codes = [['11', '12', '13'], ['21','22', '23'], ['31','32','33','34'], ['41','42','43']]
region_start_index = 5 
region_end_index = 7 
region_index = (region_start_index, region_end_index)
region_descriptions = []
region_cat = ['region', region_index, region_codes, region_descriptions]

# METRO
# codes - 1-not in metro area, 2/3/4-in metro area
metro_codes = [['1'], ['2', '3', '4']]
metro_index = 7
metro_descriptions = ['Not in \n metro area', 'In \n metro area']
metro_cat = ['metro', metro_index, metro_codes, metro_descriptions]

# AGE:
age_codes = [[str(i).zfill(3) for i in range(65)], [str(i).zfill(3) for i in range(65, 136)]]
age_start_index = 12 
age_end_index = 15 
age_index = (age_start_index, age_end_index)  
age_descriptions = ['Age < 65', r'Age $\geq$ 65']
age_cat = ['age', age_index, age_codes, age_descriptions]

# MARITAL STATUS:
# codes - 1/2-married, 3/4/5-separated,widowed,divorced, 6-single
mar_codes = [['1', '2'], ['3','4', '5'], ['6']]
mar_index = 15
mar_descriptions = []
mar_cat = ['marital', mar_index, mar_codes, mar_descriptions]

# EDUC:
# codes - 00:NA or no schooling, 01-05:less than high school, 06:high school, 07-09:some college, 10-11:bachelors and beyond
educ_codes = [['00'], ['01', '02', '03', '04', '05'], ['06'], ['07', '08', '09'], ['10', '11']]
educ_start_index = 21
educ_end_index =  23
educ_index = (educ_start_index, educ_end_index) 
educ_descriptions = [] 
educ_cat = ['educ', educ_index, educ_codes, educ_descriptions]

# EDUCD:
# codes -  999/000/001/002-Missing/N/A or no schooling, 010-026-less than 9th grade, 030/040/050/060/061-grades 9-12 no diploma, 062/063/064-high school graduate or GED, 
#          065/070/071/080/081/082/083/090/100-some college or associate's degree, 101/110/111/112/113/114/115/116-bachelor's degree or more 
educd_codes = [['999', '000', '001', '002'], ['010', '011', '012', '013', '014', '015', '016', '017', '020', '021', '022', '023', '024', '025', '026'], ['030', '040', '050', '060', '061'], ['062', '063', '064'], ['065', '070', '071', '080', '081', '082', '083', '090', '100'], ['101', '110', '111', '112', '113', '114', '115', '116']]
educd_start_index =  23
educd_end_index = 26
educd_index = (educd_start_index, educd_end_index)  
educd_descriptions = []
educd_cat = ['educd', educd_index, educd_codes, educd_descriptions]

# INCWAGE: 
# Topcoded at $5,001, remove records with missing values 999999, 999998
inc_start_index = 26 
inc_end_index = 32
inc_topcode = 5001

person_id_start_index = 8
person_id_end_index = 12 
head_id_start_index = 32
head_id_end_index = 36

target_region_code = '41' # mountain region

# Categories
categories = [hhtype_cat, metro_cat, age_cat]
param_string = 'cat'
print("Categories:", categories)

# Dataset name
downloaded_data_path = "../../usa_00007.dat" 
full_dataset_name = '1940'
region_dataset_name = '1940_region%s' % (target_region_code)
sample_str = 'sample'
jitter_str = 'jitter'

def format_full_data():
	# Import full data
	file_path = downloaded_data_path
	print(file_path)
	# above .data file is comma delimited
	full_income_data = pd.read_csv(file_path, delimiter=",")

	# Remove person records who are not heads of household and records with N/A incomes
	# Person index should match person Head index
	# Incomes should be within [0, 5001]
	household_income_data = []
	print("Len full income data", len(full_income_data))
	for i in range(len(full_income_data)):
		record = str(full_income_data.iloc[i,0])
		income = record[inc_start_index:inc_end_index].strip()
		if (i % 500000 == 0):
			print(i, income)
		if income != '' and record[person_id_start_index:person_id_end_index] == record[head_id_start_index:head_id_end_index] and int(income) <= inc_topcode:
			household_income_data.append(record)
			if (i % 500000 == 0):
			   print("Saving record", i, income)
	print("Length: full-", len(full_income_data), "households", len(household_income_data))
	# Save full household income data
	save_path = "%s/%s.npy" % (dir_path, full_dataset_name)
	np.save(save_path, np.array(household_income_data))

def restrict_to_target_region():
	# Load full household income data
	print("Loading household income data")
	save_path = "%s/%s.npy" % (dir_path, full_dataset_name)
	household_income_data = np.load(save_path, allow_pickle=True)
	num_households = len(household_income_data)
	print("Num households:", num_households)

	# Select households in target region
	region_household_data = []
	for i in range(num_households):
		record = household_income_data[i]
		region_code = record[region_start_index:region_end_index]
		if i % 20000 == 0:
			print(i, region_code, target_region_code)
		if region_code == target_region_code:
			region_household_data.append(record) 
			if i % 20000 == 0:
				print("Adding record", i, region_code)
	print("Num households in target region:", len(region_household_data))
	save_path = "%s/%s.npy" % (dir_path, region_dataset_name)
	print("Save path:", save_path)
	np.save(save_path, region_household_data)

# def jitter_data(use_full_household_data=False):
# 	if use_full_household_data:
# 		print("Loading full household income data")
# 		dataset_name = full_dataset_name
# 	else:
# 		print("Loading target region household income data")
# 		dataset_name = region_dataset_name
		
# 	# Load household income data
# 	save_path = "%s/%s.npy" % (dir_path, dataset_name)
# 	household_data = np.load(save_path, allow_pickle=True)

# 	# Add jitters
# 	print("Creating jittered data")
# 	jitter_household_data = [int(household_data[i]) + np.random.random() for i in range(len(household_data))]
# 	print("Saving jittered data")
# 	jitter_dataset_name = dataset_name + '_' + jitter_str
# 	save_path = "%s/%s.npy" % (jitter_dir_path, jitter_dataset_name)
# 	np.save(save_path, jitter_household_data)

def sample_from_data(start_index, num_samples, use_full_household_data=False, use_jitter_data=False, 
	dir_path=dir_path, res_path=res_path, analysis_path=analysis_path):
	if use_full_household_data:
		print("Loading full household income data")
		dataset_name = full_dataset_name
	else:
		print("Loading target region household income data")
		dataset_name = region_dataset_name	

	# Load household income data
	save_path = "%s/%s.npy" % (dir_path, dataset_name)
	household_income_data = np.load(save_path, allow_pickle=True)

	# Get jitters, set save paths
	if use_jitter_data:
		dir_path = jitter_dir_path
		save_path = "%s/%s_%s.npy" % (dir_path, dataset_name, 'all_jitters')
		jitters = np.load(save_path, allow_pickle=True)
		dataset_name = dataset_name + '_' + jitter_str
	sample_dataset_name = dataset_name + '_' + sample_str

	# Set sample size to 1% of total
	num_households = len(household_income_data)
	num_sample_households = int(0.01*num_households)
	all_sample_indices = np.random.choice(num_households, num_sample_households*num_samples)
	for i in range(start_index, num_samples):
		print("Starting to create sample", i)
		# Sample 1 percent of indices randomly
		sample_indices = all_sample_indices[i*num_sample_households:(i+1)*num_sample_households] # np.random.choice(num_households, num_sample_households)
		print("Len sample:", len(sample_indices))
		# Break down and save incomes by code
		sample_incomes_by_code = [[[] for code in categories[c][2]] for c in range(len(categories))]
		for s in range(len(sample_indices)):
			sample_index = sample_indices[s]
			record = household_income_data[sample_index]
			record_income = int(record[inc_start_index:inc_end_index])
			if use_jitter_data:
				record_income = record_income + jitters[sample_index]
			for c in range(len(categories)):
				cat = categories[c][0]
				code_index = categories[c][1]
				codes = categories[c][2]
				if type(code_index) is tuple:
					record_code = record[code_index[0]:code_index[1]]
				else:
					record_code = record[code_index]
				for j in range(len(codes)):
					if str(record_code) in codes[j]:
						sample_incomes_by_code[c][j].append(record_income)
						# if s % 100000 == 0:
						# 	print("sample index number:", s, "record_income:", record_income, "category:", cat, "record code", record_code, "in", codes[j])
						break
		print("Finished creating sample", i)
		# Save sample incomes by code
		k = 0
		for c in range(len(categories)):
			cat = categories[c][0]
			codes = categories[c][2]
			print("Saving sample", i, "for category", cat)
			for j in range(len(codes)):
				sample_incomes_for_code = sample_incomes_by_code[c][j]
				print("Len category", cat, "code", j, "sample incomes", len(sample_incomes_for_code))
				save_path = "%s/%s_%s_%s_%s.npy" % (dir_path, sample_dataset_name, param_string, str(k), str(i))
				print("Save path for category", cat, "code", j, ":", save_path)
				np.save(save_path, sample_incomes_for_code)
				k += 1

def save_population_incomes_by_code(use_full_household_data=False, use_jitter_data=False, dir_path=dir_path, res_path=res_path, analysis_path=analysis_path):
	if use_full_household_data:
		print("Loading full household income data")
		dataset_name = full_dataset_name
	else:
		print("Loading target region household income data")
		dataset_name = region_dataset_name
		
	# Load household income data
	save_path = "%s/%s.npy" % (dir_path, dataset_name)
	household_income_data = np.load(save_path, allow_pickle=True)
	print("Len household income data", len(household_income_data))

	if use_jitter_data:
		jitters = np.random.normal(loc=0.0, scale=0.1, size=len(household_income_data))

	# Break down and save incomes by code
	pop_incomes_by_code = [[[] for code in categories[c][2]] for c in range(len(categories))]
	for i in range(len(household_income_data)):
		record = household_income_data[i]
		record_income = int(record[inc_start_index:inc_end_index]) 
		# Add noise to make incomes distinct
		if use_jitter_data:
			record_income = float(record_income) + jitters[i]
		for c in range(len(categories)):
			cat = categories[c][0]
			code_index = categories[c][1]
			codes = categories[c][2]
			if type(code_index) is tuple:
				record_code = record[code_index[0]:code_index[1]]
			else:
				record_code = record[code_index]
			for j in range(len(codes)):
				if i % 200000 == 0:
					print("index number:", i, "category:", cat, "record code", record_code)
				for j in range(len(codes)):
					if record_code in codes[j]:
						pop_incomes_by_code[c][j].append(record_income)
						if i % 200000 == 0:
							print("index number:", i, "record_income:", record_income, "category:", cat, "record code", record_code, "in", codes[j])
						break

	# Save jitters, set save paths
	if use_jitter_data:
		dir_path = jitter_dir_path
		save_path = "%s/%s_%s.npy" % (dir_path, dataset_name, 'all_jitters')
		np.save(save_path, jitters)
		dataset_name = dataset_name + '_' + jitter_str

	# Save population incomes by code
	for c in range(len(categories)):
		cat = categories[c][0]
		codes = categories[c][2]
		print("Saving category", cat)
		for j in range(len(codes)):
			pop_incomes_for_code = pop_incomes_by_code[c][j]
			print("Len category", cat, "code", j, "pop incomes", len(pop_incomes_for_code))
			save_path = "%s/%s_%s_%s.npy" % (dir_path, dataset_name, cat, str(j))
			print("Save path for category", cat, "code", j, ":", save_path)
			np.save(save_path, pop_incomes_for_code)

def analyze_population_incomes_by_code(use_full_household_data=False, use_jitter_data=False, dir_path=dir_path, res_path=res_path, analysis_path=analysis_path):
	if use_full_household_data:
		dataset_name = full_dataset_name
	else:
		dataset_name = region_dataset_name

	if use_jitter_data:
		dir_path = jitter_dir_path
		res_path = jitter_res_path
		dataset_name = dataset_name + '_' + jitter_str

	# Collect population income by type
	nonprivate_pop_medians_by_code = [[] for c in range(len(categories))]
	for c in range(len(categories)):
		cat = categories[c][0]
		codes = categories[c][2]
		for j in range(len(codes)):
			# Load population incomes for jth type
			save_path = "%s/%s_%s_%s.npy" % (dir_path, dataset_name, str(cat), str(j))
			incomes = np.load(save_path, allow_pickle=True) 
			# Save population median
			med = np.median(incomes)
			print("Category", cat, "code", j, "population median:", med)
			nonprivate_pop_medians_by_code[c].append(med)
	print(nonprivate_pop_medians_by_code)
	save_path = "%s/%s_all_nonpriv_meds.npy" % (res_path, dataset_name)
	print("Save path for nonprivate medians:", save_path)
	np.save(save_path, np.array(nonprivate_pop_medians_by_code))


def visualize_population_incomes_by_code(use_full_household_data=False, use_jitter_data=False, dir_path=dir_path, res_path=res_path, analysis_path=analysis_path):
	if use_full_household_data:
		dataset_name = full_dataset_name
	else:
		dataset_name = region_dataset_name

	if use_jitter_data:
		dir_path = jitter_dir_path
		dataset_name = dataset_name + '_' + jitter_str

	for c in range(len(categories)):
		cat = categories[c][0]
		codes = categories[c][2]
		for j in range(len(codes)):
			save_path = "%s/%s_%s_%s.npy" % (dir_path, dataset_name, cat, str(j))
			incomes_by_code = np.load(save_path, allow_pickle=True)
			print("Category:", cat, str(j), "min, max, median, mean", min(incomes_by_code), max(incomes_by_code), np.median(incomes_by_code), np.mean(incomes_by_code))
			plt.hist(incomes_by_code, bins=5000)
			plt.title(cat+'_'+str(j))
			fig_save_path = '%s/%s_%s_incomes_hist.pdf' % (dir_path, cat, str(j))
			plt.savefig(fig_save_path, bbox_inches='tight')

def flatten_cats():
	flat_cats = []
	for c in range(len(categories)):
		codes = categories[c][2]
		for j in range(len(codes)):
			cat_desc = categories[c][3][j]
			flat_cats.append((cat_desc, codes[j]))
	return flat_cats

def flatten_nonpriv_medians(nonprivate_pop_medians_by_code):
	flat_nonpriv_pop_medians = []
	for c in range(len(categories)):
		codes = categories[c][2]
		for j in range(len(codes)):
			flat_nonpriv_pop_medians.append(nonprivate_pop_medians_by_code[c][j])
	return flat_nonpriv_pop_medians

def run_DPmed_on_sample_incomes(name, num_trials, num_datasets, alpha, beta, rho, lower_bound, upper_bound, granularity, em_granularity, cdp, 
	rerun_algs, start_param=0, end_param=6, use_full_household_data=False, use_jitter_data=False, dir_path=dir_path, res_path=res_path, 
	analysis_path=analysis_path, compare_to_nonparametric=True):

	if use_full_household_data:
		dataset_name = full_dataset_name
	else:
		dataset_name = region_dataset_name

	if use_jitter_data:
		dir_path = jitter_dir_path
		res_path = jitter_res_path
		analysis_path = jitter_analysis_path
		dataset_name = dataset_name + '_' + jitter_str

	nonpriv_results_path = "%s/%s_all_nonpriv_meds.npy" % (res_path, dataset_name)
	nonprivate_pop_medians_by_code = np.load(nonpriv_results_path, allow_pickle=True)

	hyperparameters = {}
	hyperparameters['em_granularity'] = em_granularity
	hyperparameters['granularity'] = granularity
	hyperparameters['beta'] = beta 
	hyperparameters['alpha'] = alpha
	hyperparameters['cdp'] = cdp

	# Choose function
	if name == 'Expmedian':
		function_name = exp_median.dpCIsExp
	elif name == 'Expmedian_naive':
		function_name = exp_median.dpCIsExp
		hyperparameters['naive'] = True
	elif name == 'CDFmedian':
		function_name = cdf_median.dpCIsCDF
	elif name == 'test':
		function_name = exp_median.test

	print("name", name, "lb, ub", lower_bound, upper_bound, "gran", granularity, "em_gran", em_granularity, "alpha", alpha, "rho", rho)

	flat_cats = flatten_cats()
	flat_nonpriv_pop_medians = flatten_nonpriv_medians(nonprivate_pop_medians_by_code)
			
	for k in range(max(start_param, 0), min(end_param, len(flat_cats))):
		cat = flat_cats[k][0]
		print("Starting category", cat)

		sample_dataset_name = '%s_%s_%s_%s' % (dataset_name, sample_str, param_string, str(k))
		true_median = flat_nonpriv_pop_medians[k]
		new_name = name+'_'+param_string+'_'+str(k)

		print("Category", cat, "true median", true_median, "sample dataset name", sample_dataset_name)

		run_CI_algs.runAnalyzeAlg(sample_dataset_name, new_name, dir_path, res_path, analysis_path, num_datasets, num_trials, rho, alpha, beta, lower_bound, upper_bound, true_median, 
			function_name, hyperparameters, rerun_algs, compare_to_nonparametric)

def plot_DPmed_for_sample_incomes(name, num_trials, num_datasets, alpha, rho, granularity, lower_bound, upper_bound, 
	use_full_household_data=False, use_jitter_data=False, dir_path=dir_path, res_path=res_path, analysis_path=analysis_path, 
	compare_to_nonparametric=True, blackwhite=True, plot_width=True, plot_coverage=False):
	if use_full_household_data:
		dataset_name = full_dataset_name
	else:
		dataset_name = region_dataset_name

	if use_jitter_data:
		dir_path = jitter_dir_path
		res_path = jitter_res_path
		analysis_path = jitter_analysis_path
		dataset_name = dataset_name + '_' + jitter_str

	sample_dataset_name = dataset_name + '_' + sample_str
	compare_str = '' if compare_to_nonparametric else '_parametric'
	bw_str = 'blackwhite' if blackwhite else ''


	flat_cats = flatten_cats()
	names = [name]
	num_params = len(flat_cats)
	x_set = [cat[0] for cat in flat_cats]
	xlabel = "" # 'Characteristics'
	labels = ['ExpMech', 'Nonprivate']
	colors = ['indianred', 'darkslategray']
	hatches = ['///', '\\\\']
	styles = [{'color': colors[i], 'linestyle': '-', 'marker':'.', 'hatch': hatches[i]} for i in range(len(colors))]

	if plot_width:
		title =  "" #r"ExpMech vs. Non-private Confidence Intervals ($\alpha$=%.2f, $\rho$=%.2f, $\theta$=%.2f, $\mathcal{R}$=[%s, %s])" % (alpha, rho, granularity, lower_bound, upper_bound)
		save_path = '%s/width-boxplots-90-%s%s.pdf' % (figs_path, bw_str, compare_str)
		analysis_for_data.plotCIs(names, sample_dataset_name, title, x_set, xlabel, labels, alpha=alpha, file_suffix='sizes', param_string=param_string, 
			dataset_param=True, log=False, legend_out=False, xlim=None, ylim=None, sort='byParam', save=True, save_path=save_path, ratio=False, 
			line_plot=False, box_plot=True, analysis_path=analysis_path, colors=colors, styles=styles, blackwhite=blackwhite, compare_to_nonparametric=compare_to_nonparametric)

	if plot_coverage:
		title = "" # r"Coverage of ExpMech vs. Non-private Confidence Intervals ($\alpha$=%.2f, $\rho$=%.2f, $\theta$=%.2f, $\mathcal{R}$=[%s, %s])" % (alpha, rho, granularity, lower_bound, upper_bound)
		save_path = '%s/coverage-90-%s%s.pdf' % (figs_path, bw_str, compare_str)
		analysis_for_data.plotCIs(names, sample_dataset_name, title, x_set, xlabel, labels, alpha=alpha, file_suffix='sizes', param_string=param_string, 
			dataset_param=True, log=False, legend_out=False, xlim=None, ylim=None, sort='byParam', coverage=True, T=num_trials*num_datasets, save=True, 
			save_path=save_path, ratio=False, line_plot=True, box_plot=False, analysis_path=analysis_path, colors=colors, styles=styles, blackwhite=blackwhite,
			compare_to_nonparametric=compare_to_nonparametric)

# =========== Run experiment =========== 

# Parameters for sample
start_index = 0
num_samples = 1000
# Parameters for DP median
name = 'Expmedian'
num_trials = 20
num_datasets = num_samples
alpha = 0.1
beta = 0.01
num_categories = len(categories)
rho = 0.5/num_categories
lower_bound = 0.0
upper_bound = 5001.0
granularity = 5.0
em_granularity = 5.0
cdp = True

# format_full_data()
# restrict_to_target_region()

# save_population_incomes_by_code()
# analyze_population_incomes_by_code()
# visualize_population_incomes_by_code()

# sample_from_data(start_index, num_samples) # NEED TO RERUN NON-JITTERED!

# run_DPmed_on_sample_incomes(name, num_trials, num_datasets, alpha, beta, rho, lower_bound, upper_bound, granularity, em_granularity, 
# 	cdp, start_param=0, end_param=6, rerun_algs=True)
# plot_DPmed_for_sample_incomes(name, num_trials, num_datasets, alpha, rho, granularity, lower_bound, upper_bound)


# JITTER DATA
# save_population_incomes_by_code(use_jitter_data=True)
# analyze_population_incomes_by_code(use_jitter_data=True)
# visualize_population_incomes_by_code(use_jitter_data=True)

# sample_from_data(start_index, num_samples, use_jitter_data=True)

name = 'Expmedian'
num_datasets = 50

def run_DPmed_incomes():
	run_DPmed_on_sample_incomes(name, num_trials, num_datasets, alpha, beta, rho, lower_bound, upper_bound, granularity, em_granularity, 
		cdp, start_param=0, end_param=6, rerun_algs=False, use_jitter_data=True, compare_to_nonparametric=True)

def plot_DPmed_incomes():
	plot_DPmed_for_sample_incomes(name, num_trials, num_datasets, alpha, rho, granularity, lower_bound, upper_bound, 
		use_jitter_data=True, compare_to_nonparametric=True, blackwhite=True, plot_width=True, plot_coverage=False)


