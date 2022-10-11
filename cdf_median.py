import numpy as np
import math 
import common
import publicCI as pub
import scipy

from monotonic_proj import monotonic_proj

### DP Functions ###
def dpHistogram(x, n, lower_bound, upper_bound, epsilon, bins):
    # true histogram
    true_hist = np.histogram(x, bins=bins, range=(lower_bound, upper_bound))
    # calculate laplace noise
    # noise = np.random.laplace(0, 2.0/epsilon, size=len(true_hist[0])) #histogram queries have sensitivity 2
    # caculate gaussian noise
    cdf_sensitivity = 2.0
    noise = np.random.normal(0, common.gaussian_scale(epsilon, cdf_sensitivity, cdp=True), size=len(true_hist[0]))
    # add noise to bins
    out = ([i+j for (i,j) in zip(true_hist[0], noise)], true_hist[1])
    return out

def dpTree(x, n, lower_bound, upper_bound, epsilon, depth, cdp):
    # initialize list of number of bins at each level of tree
    bins = [2**i for i in range(1,depth)]
    # divide up epsilon
    if cdp:
        eps = epsilon/math.sqrt(depth)
    else:
        eps = epsilon/depth
    # build noisy histogram at each level of tree
    tree = [([n],(lower_bound, upper_bound))] + [dpHistogram(x, n, lower_bound, upper_bound, eps, bins[i]) for i in range(depth-1)]
    return tree


### Post-processed tree optimization ###

def inverseVariance(epsilon):
    # histogram has sensitivity 2
    b = epsilon/2
    return 2*(b**2)

# Create an array of elements that are adjacent e.g. adjacentElements([1,2,3,4]) returns [2,1,4,3]
def adjacentElements(ls):
    adj = [None]*len(ls)
    i = 0
    while i < len(ls):
        if i%2 == 1:
            adj[i] = ls[i-1]
        else:
            adj[i] = ls[i+1]
        i +=1 
    return adj


def wBelow(tree):
    """
    Recursive weight estimate from below. This assumes that the variance is the same for every node at in the tree.
    :param tree: Tree, formatted as a list of arrays where the contents of the ith array in the list is the ith level of the tree.  
    :return: Single weight for each level of the tree.
    """
    weights = [None]*len(tree) # initialize with one weight per level of the tree
    i = len(tree) - 1
    while i >= 0:
        if i == len(tree) - 1:
            weights[i] = 1
        else:
            prev = weights[i+1]
            weights[i] = (2.0*prev)/(2*prev+1) #coerce to floats with 2.0
        i -= 1
    return(weights)

def countBelow(tree, wBelows):
    """
    Recursively compute counts from below
    :param tree: Tree, formatted as a list of arrays where the contents of the ith array in the list is the ith level of the tree.
    :param wBelows: Array of weights of same length as tree, where the ith weight corresponds to the ith weight calculated from below.
    :return: List of counts for each node of the tree.
    """
    counts = [None]*len(tree)
    i = len(tree) - 1
    while i >= 0:
        if i == len(tree) - 1:
            counts[i] = tree[i][0]
        else:
            w = wBelows[i]
            child = tree[i+1][0]
            childSum = [sum(x) for x in zip(child[0:len(child)-1],child[1:len(child)])] # sum all pairs of children counts
            childSum = childSum[0:len(childSum):2] # pick out the sums that are children with the same parents
            weightedT = [w*x for x in tree[i][0]] #weigh parent counts
            weightedChild = [(1-w)*x for x in childSum] #weigh child counts
            counts[i] = [sum(x) for x in zip(weightedT, weightedChild)] #sum parent and child counts
        i -= 1
    return counts

def wAbove(tree, wBelows):
    """
    Recursive weight estimation from above

    :param tree: Tree, formatted as a list of arrays where the contents of the ith array in the list is the ith level of the tree.
    :param wBelows: Array of weights of same length as tree, where the ith weight corresponds to the ith weight calculated from below.
    :return: Single weight for each level of the tree.
    """
    weights = [None]*len(tree)
    i = 0
    while i < len(tree):
        if i == 0:
            weights[i] = 1
        else:
            prevAbove = weights[i-1]
            prevBelow = wBelows[i]
            weights[i] = 1.0/(1.0 + (prevAbove + prevBelow)**(-1))
        i +=1 
    return weights

def countAbove(tree, countsBelow, wAboves):
    """
    Recursively compute counts from above
    :param tree: Tree, formatted as a list of arrays where the contents of the ith array in the list is the ith level of the tree.
    :param countsBelow: Array of counts of same length as tree, as calculated by countBelow function
    :param wAboves: Weights computed from above, assuming each node in tree has same variance.
    :return: List of counts for each node of the tree.
    """
    counts = [None]*len(tree)
    i = 0
    while i < len(tree):
        if i == 0:
            counts[i] = tree[i][0]
        else:
            w = wAboves[i]
            parents = [val for val in counts[i-1] for _ in (0, 1)] #replicating parent counts so dimensions match
            adjacents = adjacentElements(tree[i][0])
            parentAdjDiff = [(lambda x: x[0]-x[1])(x) for x in zip(parents, adjacents)] # get difference between parent and adjacent node
            weightedT = [w*x for x in tree[i][0]] #weight current node count
            weightedPA = [(1-w)*x for x in parentAdjDiff] #weighted parent - adjacent count
            counts[i] = [sum(x) for x in zip(weightedT, weightedPA)]
        i += 1
    return counts

def optimalCount(tree, wA, cA, cB):
    """
    Optimal counts for nodes of tree
    :param tree: Tree, formatted as a list of arrays where the contents of the ith array in the list is the ith level of the tree.
    :param wA: Array of weights calculated from above with wAbove
    :param cA: Array of counts calculated from below with countBelow 
    :param cB: Array of counts calculated from above with countAbove
    :return: Optimized tree
    """
    counts = [None]*len(tree)
    i = 0
    while i < len(tree):
        if i == 0:
            counts[i] = tree[i][0]
        else:
            w = wA[i]
            parents = [val for val in cA[i-1] for _ in (0, 1)] #replicating parent counts so dimensions match
            adjacents = adjacentElements(cB[i])
            parentAdjDiff = [(lambda x: x[0]-x[1])(x) for x in zip(parents, adjacents)] # get difference between parent and adjacent node
            weightedT = [w*x for x in cB[i]] #weight current node count
            weightedPA = [(1-w)*x for x in parentAdjDiff] #weighted parent - adjacent count
            counts[i] = [sum(x) for x in zip(weightedT, weightedPA)]
        i += 1
    return zip(counts, [tree[i][1] for i in range(len(tree))])

def optimalSigma(wA, wB, epsilon):
    # sB = inverseVariance(epsilon)*wB #histogram has sensitivity 2
    # sOpt = sB * math.sqrt(wA)
    cdf_sensitivity = 2.0
    sigma = common.gaussian_scale(epsilon, cdf_sensitivity, cdp=True)
    sOpt = sigma * math.sqrt(wB) * math.sqrt(wA)
    return sOpt

def optimalPostProcess(tree, epsilon):
    """
     Optimal Post Processing
 
     Wrapper function that generates optimal tree from noisy tree generated with the Laplace mechanism. The general idea is that
     you can leverage the fact that the child counts at each node should sum to the parent count, and a node's count should be
     equal to its parent count minus the count of the adjacent child node to generate less noisy counts at every node of the tree.
     
     You can think of the leveraging of child node's information to get the parent node's counts as a recursive process that
     begins at the leaves of a tree, which here we refer to with the tag "Below" in the helper function names. Similarly, leveraging
     a parent node and the adjacent child nodes can be thought of as a recursive process that begins with the root node, which
     is referred to here with the tage "Above" in the helper function names. A new count at every node in the tree can then be calculated
     using the counts that are generated in this way, which each contribute an amount according to some weight, which are calculated
     by the wBelow and wAbove functions respectively.
     
     The theory behind this is explored in detail in the Honaker paper, whose implementation here is described in extra_docs/tree-post-processing. 
     The implementation here assumes that the variance of the noise added is equal at every level of the tree, which also means that the weights
     wBelow and wAbove are the same for nodes at the same level of the tree. Honaker provides a more general implementation in his work.
     
     reference: Honaker, James. "Efficient Use of Differentially Private Binary Trees." (2015).

    :param tree: Differentially private tree generated from dpTree$release method
    :param epsilon: The epsilon value used for the noise addition at each node (note this is not the same as the global epsilon value.)
    """
    wB = wBelow(tree) # calculate the weights of counts from below
    wA = wAbove(tree, wB) # calculate the weights of counts from above
    cB = countBelow(tree, wB) # calculate counts from below
    cA = countAbove(tree, cB, wA) # calculate counts from above
  
    return optimalCount(tree, wA, cA, cB), wA, wB

### Quantify effects of nodes on each other ###

def empty_tree(depth):
    nodes_per_level = [2**i for i in range(0,depth)]    
    tree = []
    # build a list of list of zeroes corresponding to all nodes in the tree
    [tree.append([0]*n) for n in nodes_per_level]
    return tree


def single_value_tree(depth, one_loc):
    # Creates a balanced tree of depth "depth" with 0s at all locations except a single one at the level and index to right indicated by one_loc.
    # Have to make the formatting here kind of silly to match the way the DP histogram tree tracks both the couunts and ranges for each level.

    # figure out how many nodes there are at each level of the tree
    nodes_per_level = [2**i for i in range(0,depth)]    
    tree = []
    # build a list of list of zeroes corresponding to all nodes in the tree
    [tree.append(([0]*n, [])) for n in nodes_per_level] # silly formatting is to match DP Tree formatting.
    # insert a single 1 at the relevant location
    tree[one_loc[0]][0][one_loc[1]] = 1

    return tree


def sum_trees(tree_1, tree_2):
    depth = len(tree_1)
    sum_tree = empty_tree(depth)
    for i in range(depth):
        for j in range(len(tree_1[i])):
            sum_tree[i][j] = tree_1[i][j] + tree_2[i][j]
    return sum_tree


def flip_level(level, index, node_vals):
    index_bin_string = bin(index)[2:].zfill(level)
    start = 0
    end = len(node_vals)
    for c in index_bin_string:
        mid = int((start + end)/2)
        if c == '1':
            flipped_vals = node_vals[mid:end]
            flipped_vals.extend(node_vals[start:mid])
            node_vals = [flipped_vals[i - start] if (i >= start and i < end) else node_vals[i] for i in range(len(node_vals))]
            start = mid
        else:
            end = mid
    return node_vals

def extrapolate_node_effect_tree(level, index, depth, node_effect_tree):
    index_bin_string = bin(index)[2:].zfill(level)
    first_one = index_bin_string.find('1')
    new_tree = node_effect_tree.copy()
    for i in range(first_one+1, depth):
        new_node_vals = flip_level(level, index, new_tree[i][0])
        # print("tuple:", node_effect_tree[level])
        new_tree[i]= (new_node_vals, [])
        # print("new tuple:", node_effect_tree[level])
    return new_tree

def node_effects_efficient(depth, epsilon):
    # returns a list of trees that describe the effect that a node at each level of the tree effects the measurements at any other node.
    # efficient version: only compute one post-processing per level, and extrapolate to other nodes in level

    # figure out how many nodes there are at each level of the tree
    nodes_per_level = [2**i for i in range(0,depth)] 

    node_effect_trees = []
    for i in range(0, depth):
        node_effect_trees_at_level = []
        tOpt, _, _ = optimalPostProcess(single_value_tree(depth, [i, 0]), epsilon)
        node_effect_trees_at_level.append(list(tOpt))
        for j in range(1, nodes_per_level[i]):
            new_tree = extrapolate_node_effect_tree(i, j, depth, node_effect_trees_at_level[0])
            node_effect_trees_at_level.append(new_tree)
        node_effect_trees.append(node_effect_trees_at_level)

    return node_effect_trees

def node_effects(depth, epsilon):
    # returns a list of trees that describe the effect that a node at each level of the tree effects the measurements at any other node.
    # figure out how many nodes there are at each level of the tree
    nodes_per_level = [2**i for i in range(0,depth)] 

    # build all the trees of 0's with single 1s
    all_input_trees = []
    for i in range(0,depth):
        input_trees_at_level = []
        for j in range(nodes_per_level[i]):
            input_trees_at_level.append(single_value_tree(depth, [i,j]))
        all_input_trees.append(input_trees_at_level)

    node_effect_trees = []
    for level in all_input_trees:
        node_effect_trees_at_level = []
        for input_tree in level:
            tOpt, _, _ = optimalPostProcess(input_tree, epsilon)
            node_effect_trees_at_level.append(list(tOpt))
        node_effect_trees.append(node_effect_trees_at_level)

    return node_effect_trees

# Testing efficient node effects
def test_node_effects(depth):
    res = node_effects(depth, epsilon=1)
    new_res = node_effects_efficient(depth, epsilon=1)
    print("NODE EFFECT TREES")
    for i in range(depth):
        for j in range(2**i):
            print("depth:", i, "index:", j)
            print(res[i][j])
            print(new_res[i][j])
    return res


def effects_on_node(depth, node_location, node_effect_trees):
    # tree that displays how much a particular node is affected by all other node's counts
    # node location is entered as [depth, location inside level from the left]

    # figure out how many nodes there are at each level of the tree
    nodes_per_level = [2**i for i in range(0,depth)] 
    # figure out how many nodes there are at each level of the tree   
    effects_on_node = []
    # build an empty tree
    [effects_on_node.append([0]*n) for n in nodes_per_level]
    
    # go through all the node_effect_trees and read off the effects
    # 
    for i in range(0, depth):
        for j in range(nodes_per_level[i]):
            t = node_effect_trees[i][j]
            effect = t[node_location[0]][0][node_location[1]]
            effects_on_node[i][j] = effect
    return effects_on_node
        
### Post-processed CDF
def postProcessedCDF(tree, epsilon, monotonic=False):
    # Commenting out node effects code because we can pre-process that
    # smallest granularity cdf possible uses leaf buckets
    
    # tree1 = list(tree)
    tree1 = tree
    vals = tree1[-1][1]
    counts, variances = [], []
    i = 0

    # generate trees of node effects
    depth = len(tree1)
    # node_effect_trees = node_effects_efficient(depth, epsilon)

    while i < len(vals):
        # build tree to keep track of the effects of each node on the calculation
        # effects_tree = empty_tree(len(tree1))
        # start out with min and max values at top of tree
        [m,M] = tree1[0][1]
        # initialize count for i
        count = 0
        # iterate through layers of tree
        index = 0
        j = 0
        while j<len(tree1):
            # determine if should traverse tree to left or right
            mid = m + (M-m)/2
            # if looking at leftmost node in the tree, we know empirical cdf should evaluate
            # to 0 not to bin size.
            if i == 0:
                break
            # if you don't need higher granularity, stop traversal
            if vals[i] == M:
                count += tree1[j][0][index]
                # effects = effects_on_node(depth, [j, index], node_effect_trees)
                # effects_tree = sum_trees(effects_tree, effects)
                break
            # if at leaves of tree, record the count there
            if j == len(tree1)-1:
                count += tree1[j][0][index]
                # effects = effects_on_node(depth, [j, index], node_effect_trees)
                # effects_tree = sum_trees(effects_tree, effects)
            # if traversing left
            elif vals[i] <= mid :
                # reset max value to the mid, don't add to the count
                M = mid
                # set next index of node to look at in next layer
                index = index*2
                # if at end of tree, record count at that node
            #if traversing right, 
            else: 
                # reset min value to the mid
                m = mid
                # set to next index of node to look at in next layer
                index = index*2 + 1
                count += tree1[j+1][0][index - 1] # add the node's left child to the count
                # effects = effects_on_node(depth, [j+1, index-1], node_effect_trees)
                # effects_tree = sum_trees(effects_tree, effects)
            
            j += 1       
        counts.append(count)
        
        ## Calculate variance of the count
        # variance = variance_from_effects(effects_tree, epsilon)
        i += 1
        n = tree1[0][0][0] # pull public n from root of tree
        percents = [c/n for c in counts] # normalize counts by n
        # variances.append(variance/n**2) # scaling by the normalization
        # variances.append(variance)

    if monotonic:
        percents = monotonic_proj(vals, percents)

    return percents, variances

def variance_from_effects(effects_tree, epsilon):
    # calculate the variance of the noise added originally
    # variance = 2.0*(2.0/epsilon)**2 # why times 2.0?
    cdf_sensitivity = 2.0
    variance = (common.gaussian_scale(epsilon, cdf_sensitivity, cdp=True))**2

    # square and sum all the values in the effects_tree
    total_effect = 0
    for i in range(len(effects_tree)):
        for j in range(len(effects_tree[i])):
            total_effect += effects_tree[i][j]**2
    net_variance = total_effect * variance
    return net_variance

### Post-processed median
def cdfMedian(tree, cdf, quantile=0.5):
    """
    Pull median or best estimate from tree
    :param tree: Optimized tree, formatted as a list of arrays where the contents of the ith array in the list is the ith level of the tree.
    :param cdf: CDF of tree as output by postProcessedCDF
    """
    tree1 = tree
    if quantile in cdf:
        i = cdf.index(quantile)
        val = tree1[-1][1][i]
    else:
        distances = [abs(x-quantile) for x in cdf]
        i = distances.index(min(distances))
        val = tree1[-1][1][i+1]
    return val

### Wrapper function for everything
def dpMedianCDF(x, lower_bound, upper_bound, epsilon, hyperparameters, num_trials):
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
    quantile = hyperparameters['quantile'] if ('quantile' in hyperparameters) else default_quantile
    cdp = hyperparameters['cdp'] if ('cdp' in hyperparameters) else False
    assert lower_bound <= upper_bound
    n_bins = (upper_bound-lower_bound)//granularity + 1
    depth = int(np.log2(n_bins)//1 + 1)
    
    results = [None]*num_trials
    for i in range(num_trials):
        t = dpTree(x, len(x), lower_bound, upper_bound, epsilon, depth, cdp)
        tOpt, _, _ = optimalPostProcess(t, epsilon)
        tOpt = list(tOpt)
        #print('a', list(tOpt))
        cdf, _ = postProcessedCDF(tOpt, epsilon)
        #print('b', list(tOpt))
        results[i] = cdfMedian(tOpt, cdf, quantile=quantile)
    return results

# print(dpMedianCDF(np.random.normal(size=100), -1, 1, 1, {'granularity': 0.1}, 5))

### conservative postprocessing for CDF CIs
def cdfConsPostprocess(vals, cdf, quantile=0.5, lower=False, upper=False):
    """
    Conservative postprocessing
    :param vals: Values of CDF queries
    :param cdf: CDF counts with one-sided error 
    """
    distances = np.array([(x-quantile) for x in cdf])
    if lower:
        indices = np.where(distances >= 0)[0]
        if len(indices) > 0 and np.min(indices) > 0:
            i = np.min(indices) - 1
        else:
            i = 0
        #print("lower:", i)
    else:
        indices = np.where(distances <= 0)[0]
        if len(indices) > 0 and np.max(indices) < len(vals) - 1:
            i = np.max(indices) + 1
        else:
            i = len(vals)-1
        #print("upper:", i)
    val = vals[i]
    return val

def avgNoisyCounts(vals, noisy_cdf, t, window_size):
    num_windows = int(len(noisy_cdf)/ window_size) # change to ceil? to capture last few
    # avged_noisy_counts = [np.mean(noisy_cdf[i*window_size : (i+1)*window_size]) for i in range(num_windows)]
    lower_vals = [vals[i*window_size] for i in range(num_windows)]
    upper_vals = [vals[(i+1)*window_size - 1] for i in range(num_windows)]
    lower_cis = [noisy_cdf[i*window_size]-t for i in range(num_windows)]
    upper_cis = [noisy_cdf[(i+1)*window_size - 1]+t for i in range(num_windows)]
    return lower_vals, upper_vals, lower_cis, upper_cis

def traverseTree(n, eps, lower_quantile, upper_quantile, num_steps, vals, noisy_cdf, ts, step, interval):
    if step >= num_steps-1:
        # print(step, num_steps, interval)
        return [interval]
    else:
        step = step+1
        mid_index = int(interval[0] + (interval[1]-interval[0])/2)
        noisy_est = noisy_cdf[mid_index]
        ci_est = ts[mid_index]

        left_interval = (interval[0], mid_index)
        right_interval = (mid_index+1, interval[1])

        if noisy_est - ci_est > upper_quantile:
            return traverseTree(n, eps, lower_quantile, upper_quantile, num_steps, vals, noisy_cdf, ts, step, left_interval)
        elif noisy_est + ci_est < lower_quantile:
            return traverseTree(n, eps, lower_quantile, upper_quantile, num_steps, vals, noisy_cdf, ts, step, right_interval)
        else:
            left_res = traverseTree(n, eps, lower_quantile, upper_quantile, num_steps, vals, noisy_cdf, ts, step, left_interval)
            right_res = traverseTree(n, eps, lower_quantile, upper_quantile, num_steps, vals, noisy_cdf, ts, step, right_interval)
            left_res.extend(right_res)
            return left_res


### Wrapper function for CI
def dpCIsCDF(x, lower_bound, upper_bound, epsilon, hyperparameters, num_trials):
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
    bs = hyperparameters['bs'] if ('bs' in hyperparameters) else False
    naive = hyperparameters['naive'] if ('naive' in hyperparameters) else False
    avg_window = int(hyperparameters['avg_window']) if ('avg_window' in hyperparameters) else 1
    monotonic = hyperparameters['monotonic'] if ('monotonic' in hyperparameters) else False
    a_list_lower = hyperparameters['a_list_lower'] if ('a_list_lower' in hyperparameters) else None
    a_list_upper = hyperparameters['a_list_upper'] if ('a_list_upper' in hyperparameters) else None
    variances = hyperparameters['cdf_variances'] if ('cdf_variances' in hyperparameters) else None
    assert lower_bound <= upper_bound

    n = len(x)
    n_bins = (upper_bound-lower_bound)//granularity + 1
    depth = int(np.log2(n_bins)//1 + 1)
    max_eps_per_val = epsilon / math.sqrt(depth) if cdp else epsilon / depth
    cdf_sensitivity = 2.0

    # if beta is not None:
    #     t = compute_t(n, max_eps_per_val, beta, gaussian=True, cdp=True)
    # else:
    #     t = 0
    
    results = [None]*num_trials
    for j in range(num_trials):
        tt = dpTree(x, len(x), lower_bound, upper_bound, epsilon, depth, cdp)
        tOpt, wA, wB = optimalPostProcess(tt, epsilon)
        tOpt = list(tOpt)
        noisy_cdf, _ = postProcessedCDF(tOpt, epsilon, monotonic) # cdf is in percents, variance is for counts
        vals = tOpt[-1][1]
        ts = []
        beta_per_val = beta # Postprocessing analysis applies to pointwise bounds
        for i in range(len(variances)):
            scale = np.sqrt(variances[i])
            t = common.compute_t(n, max_eps_per_val, beta_per_val, cdf_sensitivity, scale, gaussian) # in percents
            ts.append(t) 
        noisy_counts = [(vals[i], n*noisy_cdf[i], n*ts[i]) for i in range(len(vals))]

        # Choose desired post-processing mechanism
        if bs: 
            intervals = traverseTree(n, max_eps_per_val, lower_quantile, upper_quantile, depth, vals, noisy_cdf, ts, 1, (0, len(vals)-1))
            # print("orig intervals", intervals)
            lower_res = vals[min([lower for (lower, upper) in intervals])]
            upper_res = vals[max([upper for (lower, upper) in intervals])]
            noisy_counts.append([(vals[i], vals[j]) for (i,j) in intervals])
        else:
            lower_cis = [(noisy_cdf[i] - ts[i]) for i in range(len(noisy_cdf))] 
            upper_cis = [(noisy_cdf[i] + ts[i]) for i in range(len(noisy_cdf))]
            lower_naive_res = cdfConsPostprocess(vals, upper_cis, quantile=lower_quantile, lower=True) # we are passing upper bounds to lower and vice versa
            upper_naive_res = cdfConsPostprocess(vals, lower_cis, quantile=upper_quantile, upper=True)
            lower_fancy_res, upper_fancy_res = cdfFancyPostProcess(vals, noisy_cdf, a_list_lower, a_list_upper)
            # print("naive:", lower_naive_res, upper_naive_res)
            # print("fancy:", lower_fancy_res, upper_fancy_res)
            if naive:
                lower_res = lower_naive_res
                upper_res = upper_naive_res
            else:
                lower_res = lower_fancy_res
                upper_res = upper_fancy_res
        
        results[j] = (lower_res, upper_res)
        #print("Results for trial", j, " : ", lower_res, upper_res)
        if save_path != None and j == 0:
            noisy_counts.append((len(vals), t, lower_quantile, upper_quantile))
            noisy_counts.append((lower_res, upper_res))
            np.save(save_path, np.array(noisy_counts))
    return results


## Pre-process Post-processed variances (not data-dependent)
def getVariances(tree, epsilon, cdp=True):
    tree1 = tree
    vals = tree1[-1][1]
    variances = []
    i = 0

    # generate trees of node effects
    depth = len(tree1)
    node_effect_trees = node_effects_efficient(depth, epsilon)
    max_eps_per_val = epsilon / math.sqrt(depth) if cdp else epsilon / depth

    while i < len(vals):
        [m,M] = tree1[0][1]
        # build tree to keep track of the effects of each node on the calculation
        effects_tree = empty_tree(depth)
        # iterate through layers of tree
        index = 0
        j = 0
        while j < depth:
            # determine if should traverse tree to left or right
            mid = m + (M-m)/2.0
            # if looking at leftmost node in the tree, we know empirical cdf should evaluate
            # to 0 not to bin size.
            if i == 0:
                break
            # if you don't need higher granularity, stop traversal
            if vals[i] == M:
                effects = effects_on_node(depth, [j, index], node_effect_trees)
                effects_tree = sum_trees(effects_tree, effects)
                break
            # if at leaves of tree, record the count there
            if j == depth - 1:
                effects = effects_on_node(depth, [j, index], node_effect_trees)
                effects_tree = sum_trees(effects_tree, effects)
            # if traversing left
            elif vals[i] <= mid :
                # reset max value to the mid, don't add to the count
                M = mid
                # set next index of node to look at in next layer
                index = index*2
                # if at end of tree, record count at that node
            #if traversing right, 
            else: 
                # reset min value to the mid
                m = mid
                # set to next index of node to look at in next layer
                index = index*2 + 1
                effects = effects_on_node(depth, [j+1, index-1], node_effect_trees)
                effects_tree = sum_trees(effects_tree, effects)
            
            j += 1       

        ## Calculate variance of the count and std of the quantile
        variance = variance_from_effects(effects_tree, max_eps_per_val)
        variances.append(variance)

        i += 1

    return variances

def a_binsearch(n, qm_probs, noise_scale, a_lower, a_upper, alpha, epsilon):
    if noise_scale == 0:
        noise_scale = 0.0001
    a_gran = alpha/10.0
    num_iters = int(np.log2((a_upper - a_lower)/a_gran))
    # Very rough heuristic for scaling up iterations based on epsilon
    if epsilon < 1.0:
        num_iters = num_iters
    elif epsilon < 3.0:
        num_iters = num_iters + 1
    elif epsilon < 5.0:
        num_iters = num_iters + 2
    else:
        num_iters = num_iters + 3
    ret = a_upper
    # print("num_iters:", num_iters)
    for i in range(num_iters):
        a_mid = (a_lower + a_upper)/2.0
        terms = []
        for j in range(1, n+1):
            qm = float(j)/n
            qm_prob = qm_probs[j-1]
            normal_prob = 1.0 - scipy.stats.norm.cdf(a_mid, loc=qm, scale=noise_scale) 
            terms.append(qm_prob*normal_prob)
        res_prob = sum(terms)
        # print("a_lower, a_mid, a_upper, res_prob:", a_lower, a_mid, a_upper, res_prob)
        if res_prob < alpha:
            a_upper = a_mid
            ret = a_mid
        else:
            a_lower = a_mid
            ret = a_upper 
    return ret

def cdfFancyPostProcess(vals, cdf, a_list_lower, a_list_upper):
    # for lower interval, look at upper cis
    distances = np.array([(cdf[i]-a_list_lower[i]) for i in range(len(vals))])
    indices = np.where(distances >= 0)[0]
    if len(indices) > 0 and np.min(indices) > 0:
        i = np.min(indices) -1
    else:
        i = 0
    lower_val = vals[i]

    # vice versa
    distances = np.array([(cdf[i]-a_list_upper[i]) for i in range(len(vals))])
    indices = np.where(distances <= 0)[0]
    if len(indices) > 0 and np.max(indices) < len(vals)-1:
        i = np.max(indices) +1
    else:
        i = len(vals)-1
    # print("upper measurement", cdf[i], "corresponding a", a_list_upper[i])
    upper_val = vals[i]
    cdf_for_indices = [cdf[i] for i in indices]
    a_for_indices = [a_list_upper[i] for i in indices]
    dist_for_indices = [distances[i] for i in indices]

    return lower_val, upper_val

def preProcessCDF(n, lower_bound, upper_bound, granularity, epsilon, alpha, cdp=True, variances=None):
    print("Starting CDF preprocessing. This will take a few minutes...")
    if variances == None:
        x = list(np.linspace(lower_bound, upper_bound, num=n))
        n_bins = (upper_bound-lower_bound)//granularity + 1
        depth = int(np.log2(n_bins)//1 + 1)
        tt = dpTree(x, len(x), lower_bound, upper_bound, epsilon, depth, cdp=cdp)
        tOpt, wA, wB = optimalPostProcess(tt, epsilon)
        tOpt = list(tOpt)
        variances = getVariances(tOpt, epsilon, cdp=cdp)
    # Compute all the binomial/normal approx coefficients
    qm_probs = []
    for i in range(1, n+1):
        qm_prob = scipy.special.binom(n, i) * (0.5)**(i) * (0.5)**(n-i) if n < 800 else scipy.stats.norm.pdf(i, loc=n*0.5, scale=np.sqrt(n*0.25))
        qm_probs.append(qm_prob)
    # Compute the a_is 
    a_lower = 0.0
    a_upper = 1.0
    a_list_lower = []
    a_list_upper = []
    for i in range(len(variances)):
        noise_scale = np.sqrt(variances[i])/float(n) # variances are for counts, we want std for percents
        # print("noise_scale:", noise_scale)
        a_i_upper = a_binsearch(n, qm_probs, noise_scale, a_lower, a_upper, alpha/2.0, epsilon)
        a_list_upper.append(a_i_upper)
        a_list_lower.append(1.0-a_i_upper)
    print("CDF preprocessing finished!")
    return a_list_lower, a_list_upper, variances

# Testing:
# x = list(np.arange(0, 100, 1))
# n = len(x)
# lower_bound = 0
# upper_bound = 100
# granularity = 5
# epsilon = 0.5
# alpha = 0.05
# num_trials = 1
# hyperparameters = {'granularity': granularity, 'quantile': 0.5, 'alpha': alpha, 'cdp': True}

# a_list_lower, a_list_upper, variances = preProcessCDF(n, lower_bound, upper_bound, granularity, epsilon, alpha)
# hyperparameters['cdf_variances'] = variances
# hyperparameters['a_list_lower'] = a_list_lower
# hyperparameters['a_list_upper'] = a_list_upper
# # print("a_lower:", a_list_lower)
# # print("a_upper:", a_list_upper)

# beta = 0.01
# nonpriv_beta = (alpha-beta)/(1.-beta) # compute nonprivate failure probability
# lower_quantile, upper_quantile = pub.getConfIntervalQuantiles(n, nonpriv_beta) # private algs take in non-private quantiles
# hyperparameters['lower_quantile'] = lower_quantile
# hyperparameters['upper_quantile'] = upper_quantile
# hyperparameters['beta'] = beta

# lower_quantile, upper_quantile = pub.getConfIntervalQuantiles(n, alpha) 
# print("nonpriv quantiles:", x[int(n*lower_quantile)], x[int(n*upper_quantile)])
# dpCIsCDF(x, lower_bound, upper_bound, epsilon, hyperparameters, num_trials)

# vals = np.arange(0, 10, 1)
# lenvals = len(vals)
# print("lenvals", lenvals, "vals", vals)
# cdf = [5]*lenvals
# a_list_lower = [4]*lenvals
# a_list_upper = [6]*lenvals
# print(cdfFancyPostProcess(vals, cdf, a_list_lower, a_list_upper))



