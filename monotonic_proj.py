import numpy as np
from sklearn.datasets import make_regression
from sklearn.isotonic import IsotonicRegression

def monotonic_proj(xs, counts, y_max=1.0):
	"""
	Performs an isotonic regression to enforce monotonicity between the counts. 

	qs: ordered numpy array of the quantiles queried
	counts: a numpy array of noisy counts, where counts[i] corresponds to the quantile qs[i]
	"""
	# create the regression model
	iso_reg = IsotonicRegression(y_min=0.0, y_max=y_max).fit(xs, counts)
	# output array of predictions where monotonic_counts[i] is the model's prediction for qs[i]. 
	monotonic_counts = iso_reg.predict(xs)		   
	return monotonic_counts

