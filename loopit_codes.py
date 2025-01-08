def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    They weights are in effect first normalized so that they
    sum to 1 (and so they must not all be 0).

    values, weights -- NumPy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights, axis = 1)
    diffsq_arr = np.zeros(values.shape)
    num_samples = values.shape[1]
    for i in range(num_samples):
      diffsq_arr[:,i] = values[:,i] - average
    variance = np.average(diffsq_arr**2, weights=weights, axis = 1)
    return (average, np.sqrt(variance))

def loopit(x_obs, y_mean, covariance, chains):
  """
  Run LOOPIT algorithm give some data, covariance, and MCMC chains

  x_obs: n dimension array
  y_mean: n dimension array
  covariance: n x n array
  chains: flattened chains from emcee work here
  """
  num_data = len(y_mean)
  num_samples = chains.shape[0]
  ppc_samples = np.zeros((num_data, num_samples))

  ###
  ###

  #here we assume the data is multivariate gaussian with the est covariance and just sample to get the unweighted PPC
  for i in tqdm(range(num_samples), position = 0, leave = True): 
    theta = chains[i]
    ppc_sample = model(theta, x_obs)
    ppc_samples[:,i] = np.random.multivariate_normal(ppc_sample, covariance)

  ###
  ###

  #now we calculate weights
  log_weights = np.zeros(ppc_samples.shape)
  print("Calculating unsmoothed weights")
  post_log_lik = np.zeros(num_samples)

  for s in tqdm(range(num_samples), position = 0, leave = True):
    theta = chains[s]
    post_log_lik[s] = log_probability(theta, x_obs, y_mean, covariance)

  for i in tqdm(range(num_data), position = 0, leave = True):
    """
    This is the slowest step, but I've made it about 30% faster than the version for the paper
    I'm sure some thinking could make it much faster, but it works correctly here
    """
    loo_y_mean = np.delete(y_mean, i)
    loo_x_obs = np.delete(x_obs, i)
    loo_covariance = np.delete(np.delete(covariance, i , 1), i, 0)
    log_weights[i,:] = np.array([log_probability(chains[s], loo_x_obs, loo_y_mean, loo_covariance) for s in range(num_samples)]) - post_log_lik
  
  ###
  ###

  #now apply the PSIS smoothing algo
  print("Applying PSIS on {} data points".format(num_data))
  for i in tqdm(range(num_data), position = 0, leave = True):
    temp_log_psis, k = az.psislw(log_weights[i,:])
    log_weights[i,:] = temp_log_psis

    if k > 0.7:
      print("Data Index {} returned Pareto shape k = {}. Please check!".format(i, k))
  
  ###
  ###

  #Here we assume the PPCs are themselves gaussian and do the PIT
  loos = []
  pp_mean, pp_stdv = weighted_avg_and_std(ppc_samples, np.exp(log_weights))
  for i in range(num_data):
    uni_draw = scipy.stats.norm.cdf(y_mean[i], loc = pp_mean[i], scale = pp_stdv[i])
    loos.append(uni_draw)
  az.plot_dist(loos, plot_kwargs = {'lw': 2})

  for i in range(100):
    uniform = np.random.rand(num_data)
    az.plot_dist(uniform, plot_kwargs = {'alpha': 0.15})
  loos = np.array(loos)
  
  ###
  ###

  #you can save the loos array for your own plotting or KS testing etc.
