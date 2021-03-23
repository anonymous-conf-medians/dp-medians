import numpy as np

def grad(x, clip_range, current):
    '''Returns the non-noisy clipped gradient.
    x: array of x values in [0,1]
    clip_range: individual contributions to the gradient
       are clipped to a box [-a,a] where a = clip_range
    current: current prediction

    \frac{d}{dx} \sum_{i=1}^n |x_i - x| = \sum_{i=1}^n sgn(x - x_i) 
    '''
    gradients = np.sign(current*np.ones(len(x))-x)   # sign of (current - x_1, ..., current - x_n)
    clipped_gradients = np.clip(gradients, - clip_range, clip_range)
    total_clipped_gradient = np.sum(clipped_gradients, axis =0) #axis just defines axis of array to sum over

    return total_clipped_gradient

def straight_GD(x, lower_bound, upper_bound, T = 5, eta = None, clip_range=None):
    '''Straight GD with no noise and constant eta'''
    n = len(x)
    if eta==None:
        eta = 10/(T*n)
    if clip_range == None:
        clip_range = 10 #big enough not to clip
    ests = np.array((lower_bound+upper_bound)/2)
    for t in range(T):
        this_grad = grad(x, clip_range, ests)
        this_eta = 10/((t+1)*n)
        ests += - this_eta*this_grad
        #print(t, ests, eta, this_grad)
    return ests

def noisyGrad(x, clip_range, current, rho):
    '''Returns the noisy clipped gradient of the OLS loss.
    The result is rho-CDP (think rho=epsilon^2/2)
    x: array of x values in [0,1]
    clip_range: individual contributions to the gradient
       are clipped to a box [-a,a] where a = clip_range
    current: current prediction 
    rho: CDP parameter (for one gradient computation)
    '''
    sensitivity = 2*clip_range

    # Now add noise
    # Recall that rho = (l2_sens^2 / 2 sigma^2).
    # So we set sigma = l2_sens / sqrt(2*rho).
    noisy_total_clipped_gradient = \
                grad(x, clip_range, current) \
                + np.random.normal(loc=0.0, scale = sensitivity / np.sqrt(2*rho))

    return noisy_total_clipped_gradient

def straight_NGD(x, rho,
                     lower_bound,
                     upper_bound,
                     T = 5,
                     clip_range = None,
                     eta = None):
    '''Straightforward noisy GD with Gaussian noise and constant eta.
    Divide rho evenly between executions.
    Assumes n is public
    Return two concatenated pairs of estimates
    '''
    n = len(x) 
    if clip_range == None:
        clip_range = 4
    if eta!=None:
        this_eta = eta
    ests = np.array((lower_bound+upper_bound)/2)
    iterates = np.zeros(T)
    # Divide budget evenly between iterations
    this_rho = rho/T 
    for t in range(T):
        # Set eta unless user provided it. 
        if eta==None:
            this_eta = (upper_bound-lower_bound)/(n*(t+1))  ### Assume that n is public! (Maybe not needed)
        this_grad = noisyGrad(x, clip_range, ests, this_rho)
        ests += - this_eta * this_grad
        if ests > upper_bound:
            ests = upper_bound
        elif ests < lower_bound:
            ests = lower_bound
        iterates[t] = ests
    return np.average(iterates[int(np.floor(T/2)):], axis=0)
                            #also average over some window of iterates

def graddescent(x, lower_bound, upper_bound, eps, hyperparameters, num_trials):
    '''This wrapper matches the signature needed to run tests on OI data.
    x, y: numpy arrays containing data
    xm, ym are ignored
    eps is used to set the rho for CDP
    '''
    clip_range = hyperparameters['clip_range'] if ('clip_range' in hyperparameters) else None
    T = hyperparameters['T'] if ('T' in hyperparameters) else 80
    eta = hyperparameters['eta'] if ('eta' in hyperparameters) else None
    
    x = np.array(x)
    
    my_rho = eps**2 / 2 #Convert xnew to rho value for CDP
    
    results = []
    
    for i in range(num_trials):
        results.append(straight_NGD(x, my_rho, lower_bound, upper_bound, T, clip_range, eta))
    return results
