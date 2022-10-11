import os
import matplotlib.pyplot as plt
import matplotlib.patches as pltpatch
import numpy as np
import seaborn as sns
import math
from publicCI import *
import scipy.stats as st
from matplotlib.patches import Patch

sns.set(style="whitegrid", color_codes=True)
default_colors = ['goldenrod', 'indianred']
colors = sns.color_palette("hls", 10)
colors_by_label = {
'Binary Search': colors[0],
'Weighted Binary Search': colors[1],
'CDF': colors[2],
'Exponential Mechanism': colors[3],
'Gradient Descent': colors[4],
'Smooth Sensitivity': colors[5],
'Non-Private': colors[6],
}

dir_path = r'data'
res_path = r'results'
analysis_path = r'analysis'

def computeIntervalSize(ests, coverage, true_param):
    """
    Calculates empirical confidence interval with respect to the non-noisy sample median.
    :param stats: List of computed statistics
    :param coverage: Coverage probability (ie. .95 for 95% CI)
    :param trials: Number of statistics computed
    :return: Size of smallest interval
    """
    num_ests = len(ests)
    # Compute absolute value of differences of statistic from true value
    diffs = [abs(ests[i]-true_param) for i in range(num_ests)]
    # Sort differences
    diffs.sort()
    index = int(num_ests * coverage) - 1
    return diffs[index]

def getPrivateConfidenceIntervals(data, numTracts, true_med):
    """
    :param data: Input data from .npy file, as decompressed by analyzeData. Each row of the array is of the form [[results], n, true median, public_ci].
    :param numTracts: Number of data tracts in the .npy file, as passed by analyzeData
    :return:(1) numTracts length array of num_trial length arrays of private confidence interval sizes,
            (2) num_tracts length array of num_trial length arrays private confidence interval (lower, upper) tuples,
            (3) coverage of true median over numTracts and num_trials.
    """
    cis_sizes_by_tracts = []
    cis_vals_by_tracts = []
    coverage_sum = 0.0
    num_trials = 1
    for i in range(0, numTracts):
        cis = []
        ci_vals = []
        trials = data[i][0]
        num_trials = len(trials)
        for j in range(0, num_trials):
            cis.append(trials[j][1] - trials[j][0])
            ci_vals.append(trials[j])
            covers_true_med = true_med >= trials[j][0] and true_med <= trials[j][1]
            coverage_sum += covers_true_med
            # if j == 0:
            #     print("Priv CI", trials[j][0], trials[j][1])
        cis_sizes_by_tracts.append(cis)
        cis_vals_by_tracts.append(ci_vals)
    cis_coverage = coverage_sum / (numTracts*num_trials)

    return cis_sizes_by_tracts, cis_vals_by_tracts, cis_coverage

def getPublicConfidenceIntervals(data, numTracts, true_med):
    """
    :param data: Input data from .npy file, as decompressed by analyzeData.
    :param numTracts: Number of data tracts in the .npy file, as passed by analyzeData
    :return:(1) numTracts length array of non-private confidence interval sizes,
            (2) num_tracts length array of non-private confidence interval (lower, upper) tuples,
            (3) coverage of true median over numTracts.
    """
    ciDiffs = []
    ciVals = []
    cov_sum = 0.0

    for i in range(0, numTracts):
        ciDiffs.append(data[i][3][1] - data[i][3][0])
        ciVals.append(data[i][3])
        covers_true_med = true_med >= data[i][3][0] and true_med <= data[i][3][1]
        # print("Nonpriv CI", data[i][3][0], data[i][3][1])
        cov_sum += covers_true_med
    ciCov = cov_sum / numTracts

    return ciDiffs, ciVals, ciCov

def getMeansAndVariances(data, numTracts):
    """
    :param data: Input data from .npy file, as decompressed by analyzeData.
    :param numTracts: Number of data tracts in the .npy file, as passed by analyzeData
    :return: Array of two arrays, the first of which is a numTracts length array of sample means,
            and the second which is num_tracts length array of variances).
    """
    means = []
    variances = []
    for i in range(0, numTracts):
        trials = data[i][0]
        means.append(np.mean(trials))
        variances.append(np.var(trials))
    return [means, variances]
    
def computeConfidenceIntervals(name_ci, dataset_name, file_suffix, name, hyperparameters, true_med, res_path=res_path, analysis_path=analysis_path, compare_to_nonparametric=True):
    compare_str = ''
    if compare_to_nonparametric is False:
        compare_str = '_parametric'

    path = '%s/%s_%s%s.npy' % (res_path, dataset_name, name_ci, compare_str)
    fileContents = np.load(path, allow_pickle=True)
    header = fileContents[0]
    data = fileContents[1]
    numTracts = int(len(data))

    save_path = '%s/%s_%s_%s%s.npy' % (analysis_path, dataset_name, file_suffix, name, compare_str) # Change to name_ci instead of name here? Would need to change plotCIs.
    ns = [x[1] for x in data]
    nonpriv_medians = [x[2] for x in data]
    if len(nonpriv_medians) > 0:
        print("nonpriv_median:", nonpriv_medians[0])

    cis_sizes_by_tracts, cis_vals_by_tracts, cis_coverage = getPrivateConfidenceIntervals(data, numTracts, true_med)
    ciPub_size, ciPub_vals, ciPub_coverage = getPublicConfidenceIntervals(data, numTracts, true_med)
    print("Coverage - nonpriv:", ciPub_coverage, "priv:", cis_coverage)

    print("Save path:", save_path)

    np.save(save_path, np.array([header, ns,  
                            cis_sizes_by_tracts, cis_vals_by_tracts, cis_coverage, 
                            ciPub_size, ciPub_vals, ciPub_coverage]))

def get_nonprivate_cis(dataset_name, name, other_param=None, file_suffix='', dataset_param=False, analysis_path=analysis_path, compare_str=''):

    nonprivate_sizes = []
    nonprivate_vals = []
    nonprivate_cov = []

    def _pull_data(path, list1, list2, list3):
        # note that append in python alters list outside of function scope.
        analysis = np.load(path, allow_pickle=True)
        list1.append(analysis[-3])  # nonprivate CI sizes
        list2.append(analysis[-2])  # nonprivate CI vals
        list3.append(analysis[-1])  # nonprivate CI coverage

    if other_param is None:
        path = '%s/%s_%s_%s%s.npy' % (analysis_path, dataset_name, file_suffix, name, compare_str)
        _pull_data(path, nonprivate_sizes, nonprivate_vals, nonprivate_cov)

    else:
        res_sizes = [] 
        res_vals = []
        res_cov = []
        for i in range(len(other_param)):
            if dataset_param:
                # dataset name should include param, ie. 'lognormal_v1_n'
                # name should include param, ie. 'CDFmedian_n'
                # file_suffix is related to analysis, ie. 'sizes'
                path = '%s/%s_%s_%s_%s_%s%s.npy' % (analysis_path, dataset_name, str(i), file_suffix, name, str(i), compare_str) 
            else:
                # name should include param, ie. CDFmedian_gran
                path = '%s/%s_%s_%s_%s%s.npy' % (analysis_path, dataset_name, file_suffix, name, str(i), compare_str) 
                print("Nonpriv path:", path)
            _pull_data(path, res_sizes, res_vals, res_cov)
        nonprivate_sizes.append(res_sizes)
        # nonprivate_vals.append(res_vals) # Doesn't make sense to plot this over many params
        nonprivate_cov.append(res_cov)

    return [nonprivate_sizes, nonprivate_vals, nonprivate_cov]


def get_private_cis_and_ns(dataset_name, names, other_param=None, file_suffix='', dataset_param=False, analysis_path=analysis_path, compare_str=''):
    ns = []
    private_sizes = []
    private_vals = []
    private_cov = []

    def _pull_data(path, list1, list2, list3, list4):
        # note that append in python alters list outside of function scope.
        analysis = np.load(path, allow_pickle=True)
        list1.append(analysis[1])  # n's
        list2.append(analysis[2])  # CI sizes
        list3.append(analysis[3])  # CI vals
        list4.append(analysis[4])  # CI coverage

    for name in names:
        if other_param is None:
            path = '%s/%s_%s_%s%s.npy' % (analysis_path, dataset_name, file_suffix, name, compare_str)
            _pull_data(path, ns, private_sizes, private_vals, private_cov)

        else:
            res_sizes = [] 
            res_vals = []
            res_cov = []
            for i in range(len(other_param)):
                if dataset_param:
                    # dataset name should include param, ie. 'lognormal_v1_n'
                    # name should include param, ie. 'CDFmedian_n'
                    # file_suffix is related to analysis, ie. 'sizes'
                    path = '%s/%s_%s_%s_%s_%s%s.npy' % (analysis_path, dataset_name, str(i), file_suffix, name, str(i), compare_str)
                else:
                    # name should include param, ie. CDFmedian_gran
                    path = '%s/%s_%s_%s_%s%s.npy' % (analysis_path, dataset_name, file_suffix, name, str(i), compare_str) 
                _pull_data(path, ns, res_sizes, res_vals, res_cov)
            private_sizes.append(res_sizes)
            # private_vals.append(res_vals) # Doesn't make sense to plot this over many params
            private_cov.append(res_cov)

    return [ns, private_sizes, private_vals, private_cov]

def datasets_sorted_by_n(private_results, nonprivate_results, ns):
    private_results = np.array(private_results)
    nonprivate_results = np.array(nonprivate_results)
    out = []

    #print("private results:", private_results)
    for i in range(len(private_results)):
        #print("i:", i)
        ns_i = np.array(ns[i])
        #print("ns i:", ns_i)
        index = np.argsort(ns_i)
        #print("index:", index)
        out.append(private_results[i][index])

    #print("nonprivate_results:", nonprivate_results)
    #print("final index:", index)
    out.append(nonprivate_results[index])      # ToDo: fix this
    return out

def datasets_sorted_by_vals(private_results, nonprivate_results, single_dataset=False):
    priv_lower = []
    priv_upper = []
    priv_x_set = []
    if single_dataset:
        for i in range(len(private_results)): # algorithm (should only be one)
            j = 0 # dataset
            for k in range(len(private_results[i][j])): # trial on jth dataset
                priv_lower.append(private_results[i][j][k][0])
                priv_upper.append(private_results[i][j][k][1])
                priv_x_set.append(k)
        nonpriv_x_set = priv_x_set
        nonpriv_lower = [nonprivate_results[j][0] for k in range(len(priv_x_set))]
        nonpriv_upper = [nonprivate_results[j][1] for k in range(len(priv_x_set))]
    else:
        for i in range(len(private_results)): # algorithm (should only be one)
            for j in range(len(private_results[i])): # dataset
                for k in range(len(private_results[i][j])): # trial
                    priv_lower.append(private_results[i][j][k][0])
                    priv_upper.append(private_results[i][j][k][1])
                    priv_x_set.append(j)
        nonpriv_x_set = [i for i in range(len(nonprivate_results))]
        nonpriv_lower = [nonprivate_results[i][0] for i in range(len(nonprivate_results))]
        nonpriv_upper = [nonprivate_results[i][1] for i in range(len(nonprivate_results))]

    return ([priv_x_set, priv_x_set, nonpriv_x_set, nonpriv_x_set], [priv_lower, priv_upper, nonpriv_lower, nonpriv_upper])


def datasets_sorted_by_error_quantile_size(private_results, nonprivate_results, ratio=False):
    if ratio:
        for i in range(len(private_results)):
            private_results[i] = [(private_results[i][j] / nonprivate_results[j]) for j in range(len(private_results[i]))]
            private_results[i].sort()
        nonprivate_results = [ (nonprivate_results[j] / nonprivate_results[j]) for j in range(len(nonprivate_results))]
    else:
        for i in range(len(private_results)):
            private_results[i].sort()
        nonprivate_results.sort()
    private_results.append(nonprivate_results)
    return private_results

def style_box_plot(bp, style, labels, show_medians=False, median_color='cyan', blackwhite=False):
    # plt.setlabels(labels)

    color = style['color']
    linestyle = style['linestyle']
    hatch = style['hatch']

    fill_color = 'white' if blackwhite else color
    outline_color = color
    
    plt.setp(bp['caps'], color=outline_color, linewidth=5.0, linestyle='-')

    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians']:
        plt.setp(bp[element], color=outline_color, linewidth=3.0)

    if show_medians:
        for element in ['medians']:
            plt.setp(bp[element], color=median_color, linewidth=3.5)

    for patch in bp['boxes']:
        patch.set(facecolor=fill_color)  
        patch.set(hatch=hatch, edgecolor=outline_color)

def boxplot_positions(i, j, length):
    # change coefficient for i to change spacing between boxes within a single x-tick
    # change coefficient for j to change spacing between x-ticks and/or width of boxes
    return 0.55*i + 0.5 + 0.8*j*length

def plot(xs, ys, confidence_str, dataset_name, title, labels, xlabel, ylabel, log, xlog, legend_out, xlim, ylim, save, show_title=True,
    colors=colors, styles=[], blackwhite=False, line_plot=False, box_plot=False, small_first_marker=False, first_marker_size=8, marker_size=15, 
    error_bars=[], hlines=[], nonpriv_quantiles=[], vlines=[], nonpriv_ci=[], boxes=[], ellipses=[], 
    true_result=None, extra_line_plots=[], alpha_lines=[], save_path=None, legend=True):

    # Title and size
    plt.figure(figsize=[20,10])

    if show_title:
        plt.title(title, fontsize=25)

    # Set default styles
    if len(styles) == 0:
        styles = [{'color': colors[i], 'linestyle': '-', 'marker':'.', 'hatch':''} for i in range(len(labels))]

    # Line, scatter, or box plots
    box_plots = []
    for i in range(len(ys)):
        print(i, labels[i])
        # print("len xs:", len(xs))
        # print("len ys:", len(ys[i]))
        print("ys:", ys[i])
        line_style = styles[i]['linestyle'] if line_plot else '' #'-' if line_plot else ''
        line_width = 5.0
        marker = styles[i]['marker'] if blackwhite else '.'
        outline_color = styles[i]['color']

        if box_plot and isinstance(xs, (list, np.ndarray)) and len(xs) > 0:
            # positions = [(0.7)*i + 1 + (j*(len(ys)+3)) for j in range(len(ys[i]))]
            positions = [boxplot_positions(i, j, len(ys)) for j in range(len(ys[i]))]
            # positions = [(0.55)*i + 0.1 + (0.6*j*(len(ys))) for j in range(len(ys[i]))]
            # print(positions)
            print([np.median(ys[i][j]) for j in range(len(ys[i]))])
            bp = plt.boxplot(ys[i], patch_artist=True, showfliers=False, labels=xs[i], positions=positions)
            plt.xlim([0, max(positions)])
            style_box_plot(bp, styles[i], xs[i], blackwhite=blackwhite)
            box_plots.append(bp)
        elif isinstance(xs, (list, np.ndarray)) and len(xs) > 0:
            curr_marker_size = first_marker_size if i==0 and small_first_marker else marker_size
            plt.plot(xs[i], ys[i], marker=marker, label=labels[i], linestyle=line_style, linewidth=line_width, markersize=curr_marker_size, color=outline_color)
        else:
            plt.plot(ys[i], marker=marker, label=labels[i], linestyle=line_style, linewidth=line_width, markersize=marker_size, color=outline_color)

    # Add-ons
    for (i, errors) in error_bars:
        # print(i, errors)
        plt.errorbar(xs[i],ys[i],yerr=errors, linestyle="None", linewidth=line_width)
    for yval in hlines:
        plt.axhline(y=yval, color='gray', linestyle='-', linewidth=line_width)
    for yval in nonpriv_quantiles:
        plt.axhline(y=yval, color='gray', linestyle='--', linewidth=line_width)
    for xval in vlines:
        plt.axvline(x=xval, color='indigo', linestyle='dotted', linewidth=line_width)
    for xval in nonpriv_ci:
        plt.axvline(x=xval, color='green', linestyle='-.', linewidth=line_width)
    for (pos, width, height) in boxes:
        rectangle = plt.Rectangle(pos, width, height, fill=False, ec="cyan", linewidth=line_width)
        plt.gca().add_patch(rectangle)
    for (pos, width, height) in ellipses:
        ellipse = pltpatch.Ellipse(pos, width, height, fill=False, ec='blue', linewidth=line_width)
        plt.gca().add_patch(ellipse)
    if true_result:
        plt.axvline(x=true_result, linewidth=line_width, color='cyan')
    for i in range(len(extra_line_plots)): 
        xs = extra_line_plots[0]
        if len(extra_line_plots) > 1:
            for ys in extra_line_plots[1]:
                plt.plot(xs, ys, marker='s', linestyle='', markersize=first_marker_size, color='pink')
    print("alpha lines:", alpha_lines)
    if len(alpha_lines) > 1:
        xs = alpha_lines[0]
        ys = alpha_lines[1]
        plt.plot(xs, ys, linestyle='--', linewidth=line_width, color='gray')

    # Legend
    if box_plot and legend:
        if legend_out:
            plt.legend([bp["boxes"][0] for bp in box_plots], labels, bbox_to_anchor=(1.04,1), loc="upper left", fontsize='10', numpoints=1)
        else:
            plt.legend([bp["boxes"][0] for bp in box_plots], labels, fontsize=20, markerscale=3.0, numpoints=1)
    elif legend:
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        if legend_out:
            #plt.legend(bbox_to_anchor=(1.04,1), loc="upper left", fontsize='10', numpoints=1)
            plt.legend(fontsize=20,loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=2)
        else:
            plt.legend(fontsize=20, markerscale=3.0, numpoints=1)

    # Axes
    if log:
        plt.yscale('log')
    if xlog:
        plt.xscale('log')
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    if box_plot:
        ticks_locs = [boxplot_positions(len(ys)/2.0-1.0, j, len(ys)) for j in range(len(ys[0]))] 
        plt.xticks(fontsize=35, ticks=ticks_locs)
    else:
        plt.xticks(fontsize=35)
    plt.yticks(fontsize=35)
    plt.xlabel(xlabel, fontsize=40)
    plt.ylabel(ylabel, fontsize=40)

    # Save and show
    if save:
        if save_path == None:
            save_path = 'plots/' + dataset_name + confidence_str +'CI.pdf'
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close()

def plotCIs(names, dataset_name, title, x_set, xlabel, labels, alpha=0.05, file_suffix='sizes', param_string='', dataset_param=False, 
    log=False, xlog=False, legend=True, legend_out=False, xlim=None, ylim=None, sort=None, vals=False, single_dataset=False, coverage=False, T=None, n='', 
    show_title=True, colors=colors, styles=[], blackwhite=False, save=False, save_path=None, widened_nonpriv_vals=None, ratio=False, line_plot=True, box_plot=False, 
    analysis_path=analysis_path, compare_to_nonparametric=True, true_result=None):

    if param_string != '':
        names = [names[i]+'_'+param_string for i in range(len(names))]
    if dataset_param:
        dataset_name = dataset_name+'_'+param_string
    confidence_str = '' if alpha == None else str(int((1.-alpha)*100))
    compare_str = '' if compare_to_nonparametric else '_parametric'

    [nonprivate_sizes, nonprivate_vals, nonprivate_cov] = get_nonprivate_cis(dataset_name, names[-1], other_param=x_set, file_suffix=file_suffix, 
        dataset_param=dataset_param, analysis_path=analysis_path, compare_str=compare_str)  # Returns single nested list
    [ns, private_sizes, private_vals, private_cov] = get_private_cis_and_ns(dataset_name, names, other_param=x_set, file_suffix=file_suffix, 
        dataset_param=dataset_param, analysis_path=analysis_path, compare_str=compare_str)
    ylabel = "Ratio of widths for %s%% confidence interval" % confidence_str if ratio else "Width of %s%% Confidence Interval" % confidence_str

    if vals == True and sort == None: # If vals is true, sort should be None
    # This will plot a single algorithm's CI values on single/many datasets
        ylabel = "%s%% Confidence Interval Values" % confidence_str
        xlabel = "Datasets"
        x_set, results = datasets_sorted_by_vals(private_vals, nonprivate_vals[0], single_dataset=single_dataset)

        hlines = widened_nonpriv_vals if widened_nonpriv_vals != None else [] # This is hacky but okay for now
        hlines = [true_result] if true_result !=None else [] # This is hacky but okay for now
        plot(x_set, results, confidence_str, dataset_name, title, labels, xlabel, ylabel, log, xlog, legend_out, xlim, ylim, save, show_title=show_title, 
            save_path=save_path, hlines=hlines, colors=colors, styles=styles, blackwhite=blackwhite, legend=legend)

    if sort == 'byParam':
        assert isinstance(x_set, (list, np.ndarray))
        if coverage and T:
            ylabel = r"cov$_{%d, %s}$" % (T, str(n)) if type(n) is int else r"cov$_{%d, n}$" % (T)
            # ylabel = "Ratio of coverage for %s%% confidence interval" % confidence_str if ratio else "Coverage of %s%% Confidence Interval" % confidence_str
            results = private_cov
            nonprivate_results = nonprivate_cov[0]
            results.append(nonprivate_results)
            if alpha == None:
                alpha_lines = [x_set, [1.0-x_set[i] for i in range(len(x_set))]]
            else:
                alpha_lines = [x_set, [1.0-alpha for i in range(len(x_set))]]
        else:
            ylabel = r"rel-width$^{%.2f}$" % alpha if ratio else r"width$^{%.2f}$" % alpha
            alpha_lines = []
            if box_plot:
                results = [[[np.mean(dataset) for dataset in param] for param in alglist] for alglist in private_sizes]
                nonprivate_results = [param for param in nonprivate_sizes[0]]
                results.append(nonprivate_results)
                if ratio: 
                    for i in range(len(results)): # algorithm i
                        for j in range(len(results[i])): # param j
                            for k in range(len(results[i][j])): # dataset k
                                results[i][j][k] = results[i][j][k] / nonprivate_results[j][k]
            else: # mean line plot
                results = [[np.mean([np.mean(dataset) for dataset in param]) for param in alglist] for alglist in private_sizes]
                nonprivate_results = [np.mean(param) for param in nonprivate_sizes[0]]
                results.append(nonprivate_results)
                if ratio:
                    results = [[(results[i][j] / nonprivate_results[j]) for j in range(len(results[i]))] for i in range(len(results))]
        xs = [x_set for i in range(len(results))]
        plot(xs, results, confidence_str, dataset_name, title, labels, xlabel, ylabel, log, xlog, legend_out, xlim, ylim, save, show_title=show_title, 
            save_path=save_path, line_plot=line_plot, alpha_lines=alpha_lines, box_plot=box_plot, marker_size=10, colors=colors, styles=styles, 
            blackwhite=blackwhite, legend=legend)

    if sort == 'byErrorQuantile':
        xlabel = "Datasets sorted by ratio of confidence intervals" if ratio else "Datasets sorted by width of confidence interval"
        results = [[np.mean(dataset) for dataset in alglist] for alglist in private_sizes]
        results = datasets_sorted_by_error_quantile_size(results, nonprivate_sizes[0], ratio=ratio)
        plot(None, results, confidence_str, dataset_name, title, labels, xlabel, ylabel, log, xlog, legend_out, xlim, ylim, save, 
            show_title=show_title, save_path=save_path, colors=colors, styles=styles, blackwhite=blackwhite, legend=legend)

    if sort == 'byN':
        xlabel = "Datasets sorted by size of dataset"
        print("ns:", ns)
        results = datasets_sorted_by_n(private_sizes, nonprivate_sizes[0], ns)
        ns = np.sort(ns[0])
        xs = [ns for i in range(len(results))]
        plot(xs, results, confidence_str, dataset_name, title, labels, xlabel, ylabel, log, xlog, legend_out, xlim, ylim, save, show_title=show_title,
            save_path=save_path, colors=colors, styles=styles, blackwhite=blackwhite, legend=legend)

def makeAvgBoxes(boxes, vals, noisy_cdf, t, avg_window):
    boxes = []
    num_vals = len(vals)
    num_windows = int(np.ceil(num_vals / avg_window))
    avged_t = t / math.sqrt(avg_window)
    avged_cdf = [np.mean(noisy_cdf[i*avg_window:min(num_vals, (i+1)*avg_window)]) for i in range(num_windows)]
    for i in range(num_windows):
        pos = (vals[i*avg_window], avged_cdf[i] - avged_t)
        width = vals[min(num_vals-1, (i+1)*avg_window-1)] - vals[i*avg_window]
        height = 2*avged_t
        boxes.append((pos, width, height))
    return boxes

def plotCDFs(path, path2, dataset_name, num_datasets, title, xlabel, ylabel, labels, log=False, xlog=False, legend_out=False, xlim=None, ylim=None, 
    sort=None, vals=False, colors=colors, avg_window=1, bs=False, true_result=None, extra_line_plots=[], show_title=False,
    reuse_queries=False, plot_nonpriv_quantiles=True, plot_expanded_nonpriv_quantiles=False, alpha=0.05, beta=0.01, save=False, save_path=None, 
    dir_path=dir_path, res_path=res_path):

    confidence_str = str(1.0 -alpha)
    dataset_path = '%s/%s.npy' % (dir_path, dataset_name)
    dataset = np.load(dataset_path, allow_pickle=True)
    dataset.sort()
    # print("dataset", dataset)
    num_points = len(dataset)
    dataset_cdf = [i/num_points for i in range(num_points)]
    # print("dataset cdf", dataset_cdf)
    np_ci = getConfInterval(dataset, num_points, alpha) 
    np_q = getConfIntervalQuantiles(num_points, alpha)
    nonpriv_ci = [np_ci[0], np_ci[1]]
    print("nonpriv_ci", nonpriv_ci)
    nonpriv_quants = [np_q[0], np_q[1]]

    if path2 == None: # one-shot
        noisy_path = '%s/%s.npy' % (res_path, path)
        noisy_counts = np.load(noisy_path, allow_pickle=True)
        (num_counts, t, lower_quantile, upper_quantile) = noisy_counts[-2]
        (lower_res, upper_res) = noisy_counts[-1]
        # print("noisy counts", noisy_counts)
        vals = [noisy_counts[i][0] for i in range(num_counts)]
        # print("noisy vals", noisy_vals)
        noisy_cdf = [(noisy_counts[i][1])/num_points for i in range(num_counts)]
        # print("noisy cdf", noisy_cdf)
        error_bars = [(1, [noisy_counts[i][2]/num_points for i in range(num_counts)])]
        # print("t:", t)

        boxes = []
        if avg_window > 1:
            boxes.append(makeAvgBoxes(vals, noisy_cdf, t, int(avg_window)))
        if bs:
            intervals = noisy_counts[-3]
            print("intervals", intervals)
            for (lower, upper) in intervals:
                boxes.append(((lower, 0), upper-lower, 0.1)) # change cdf code to pass in vals not indices
        if len(extra_line_plots) > 0:
            extra_line_plots = [vals, extra_line_plots]
        hlines = []
        vlines = [lower_res, upper_res]
        if show_title:
            title = title+': (%s, %s)' % (str(lower_res), str(upper_res))
        nonpriv_quantiles = []
        if plot_nonpriv_quantiles:
            nonpriv_quantiles = nonpriv_quants
        if plot_expanded_nonpriv_quantiles:
            extra_line_plots = [vals, [[lower_quantile]*len(vals), [upper_quantile]*len(vals)]]
            print("nonpriv reg, expanded quantiles:", nonpriv_quantiles, (lower_quantile, upper_quantile))
        plot([dataset, vals], [dataset_cdf, noisy_cdf], confidence_str, dataset_name, title, labels, xlabel, ylabel, log, xlog, legend_out, xlim, ylim, save, 
            colors=colors, error_bars=error_bars, hlines=hlines, nonpriv_quantiles=nonpriv_quantiles, vlines=vlines, true_result=true_result, legend=False,
            extra_line_plots=extra_line_plots, nonpriv_ci=nonpriv_ci, boxes=boxes, small_first_marker=True, save_path=save_path)
    else: # two-shot
        noisy_path = '%s/%s.npy' % (res_path, path)
        noisy_counts = np.load(noisy_path, allow_pickle=True)
        (num_counts, t, quantile, nonpriv_quantile) = noisy_counts[-2]
        # print("t:", t)
        res = noisy_counts[-1]
        print("lower noisy counts", noisy_counts)
        vals = [noisy_counts[i][0] for i in range(num_counts)]
        # print("vals", vals)
        noisy_cdf = [(noisy_counts[i][1])/num_points for i in range(num_counts)]

        noisy_path2 = 'results/%s.npy' % (path2)
        noisy_counts2 = np.load(noisy_path2, allow_pickle=True)
        (num_counts2, t2, quantile2, nonpriv_quantile2) = noisy_counts2[-2]
        res2 = noisy_counts2[-1]
        vals2 = [noisy_counts2[i][0] for i in range(num_counts2)]
        # print("vals", vals)
        noisy_cdf2 = [(noisy_counts2[i][1])/num_points for i in range(num_counts2)]
        print("upper noisy counts", noisy_counts2)

        error_bars = [(0, [noisy_counts[i][2] for i in range(num_counts)]), (1, [noisy_counts2[i][2] for i in range(num_counts2)])]
        hlines = [nonpriv_quantile, nonpriv_quantile2]
        vlines = [res, res2]
        # print("vlines:", vlines)
        if avg_window > 1:
            boxes = makeAvgBoxes(vals, noisy_cdf, t, avg_window) + makeAvgBoxes(vals2, noisy_cdf2, t, avg_window)
        else:
            boxes = []
        if len(extra_line_plots):
            extra_line_plots = [vals, extra_line_plots]
        reused_counts = noisy_counts2[-3]
        ellipses = []
        if reuse_queries and len(reused_counts) > 0:
            ellipses = [((reused_counts[i][0], reused_counts[i][1]/num_points), 0.1, 0.05) for i in range(len(reused_counts))]
            print("ellipses:", ellipses)
        if show_title:
            title = title+': (%s, %s)' % (str(res), str(res2))
        if plot_nonpriv_quantiles:
            nonpriv_quantiles = nonpriv_quants
        plot([vals, vals2, dataset], [noisy_cdf, noisy_cdf2, dataset_cdf], confidence_str, dataset_name, title, labels, xlabel, ylabel, log, xlog, legend_out, xlim, ylim, save, 
            colors=colors, error_bars=error_bars, hlines=hlines, nonpriv_quantiles=nonpriv_quantiles, vlines=vlines, true_result=true_result, legend=False,
            extra_line_plots=extra_line_plots, nonpriv_ci=nonpriv_ci, boxes=boxes, ellipses=ellipses, small_first_marker=True, save_path=save_path)

def export_legend(legend, save_path):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    print("Saving legend to ", save_path)
    fig.savefig(save_path, dpi="figure", bbox_inches=bbox)

