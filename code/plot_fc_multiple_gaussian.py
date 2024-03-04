import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.integrate import quad

# Measured X value
x_0 = 2.5

# Alpha values to plot
alpha_list = [0.6827, 0.9545, 0.9973, 0.9999]

# Measurement and signal values
x_vals = np.arange(-10,15,0.01)
mu_vals = np.arange(0,8,0.01)

# Nonnegative mu maximizing P(x,mu)
def mu_best(x):
	return max(0,x)

# Probability of x given mu
def P(x,mu):
	return np.exp(-0.5*(x-mu)**2)/np.sqrt(2*np.pi)

# How good does mu describe x?
def R(x,mu):
	return P(x,mu)/ P(x,mu_best(x))

# Save FC Intervals
def save_confidence_band(alpha, mu_list, filename):
	# Save mu1 and mu0 corresponding to given alpha value to file
	mu_list.to_csv(filename, index=False)

# Load FC Intervals
def load_confidence_band(filename):
	# Load the data from the CSV file
	df = pd.read_csv(filename)

	return df[['alpha', 'mu', 'x0', 'x1']]

# Feldman-cousins Neyman construction of confidence intervals
def fc0(x_vals, mu, alpha=0.95):
	"""
	Find x0, x1 such that:
		R(x0)==R(x1)
		int(P(x,mu)dx,x0,x1)>=alpha #ideally would be equal, >= results in overcoverage
	"""

	def integrand(x, mu):
		return P(x, mu)

	# Calculate R(x_vals,mu)
	R_vals = [R(x,mu) for x in x_vals]

	# Rank x values in descending order of R(x, mu)
	ranked_x_vals = [x for _, x in sorted(zip(R_vals, x_vals), reverse=True)]

	neyman_interval = []
	
	# Iterate through ranked x values
	for x in ranked_x_vals:
		neyman_interval.append(x)
		x0 = min(neyman_interval)
		x1 = max(neyman_interval)
		
		# Calculate integral of P(x, mu) between x0 and x1
		integral, _ = quad(integrand, x0, x1, args=(mu,))

		# If integral >= alpha, return x0, x1
		if integral >= alpha:
			print(f'$mu$={mu:.3f}, integral={integral:.4f}')
			return x0, x1

	# If no interval satisfies the condition, return the last calculated interval
	return x0, x1

# Feldman-Cousins using Neyman interval to determine mu Confidence interval
def fc1(x_0, mu_list, alpha):
	"""
	Find mu0 and mu1 such that 
		mu0_list[mu=mu0, x=x_0]
		mu1_list[mu=mu1, x=x_0]
	Linearly interpolate between x vals to find best mu value 
	"""

	def find_corresponding_a(xy_list, x_0, index):
		# Extract a and b values from the xy_list
		a_values = np.array(xy_list['mu'])
		b_values = np.array(xy_list[index])
	
		# Check if x_0 exactly matches any b value
		if x_0 in b_values:
			index_exact_match = np.where(b_values == x_0)[0][0]
			return a_values[index_exact_match]
	
		# If no exact match, find the closest b values (b-1, b+1)
		b_values_sorted_indices = np.argsort(np.abs(b_values - x_0))
		closest_values_indices = b_values_sorted_indices[:2]
		closest_b_values = b_values[closest_values_indices]
	
		# Linear interpolation between the two closest a values
		a_values_interpolated = np.interp(x_0, closest_b_values, a_values[closest_values_indices])
	
		return a_values_interpolated

	# TODO error checking in case alpha is not found in mu_list
	alpha_subset = mu_list[mu_list['alpha'] == alpha]
	mu0 = find_corresponding_a(alpha_subset, x_0, 'x1')
	mu1 = find_corresponding_a(alpha_subset, x_0, 'x0')
	
	return mu0, mu1

mu_list = pd.DataFrame(columns=['alpha', 'mu', 'x0', 'x1'])
for alpha in alpha_list:
	filename = f'FC-intervals/FC-{alpha}.csv'
	print(f'\n========== $\alpha$ = {alpha} ==========\n')

	# If file already exists, use data
	if os.path.exists(filename):
		alpha_mu_list = load_confidence_band(filename)
	else:
		alpha_mu_list = pd.DataFrame(columns=['alpha', 'mu', 'x0', 'x1']) 
		for mu in mu_vals:
			# Determine [x0, x1] such that
			x0, x1 = fc0(x_vals, mu, alpha)
			# [alpha, mu, x1, x0]
			new_row = pd.DataFrame({'alpha': [alpha], 'mu': [mu], 'x0': [x0], 'x1': [x1]})
			alpha_mu_list = pd.concat([alpha_mu_list, new_row], ignore_index=True)
	
		# Save confidence band for current alpha
		save_confidence_band(alpha, alpha_mu_list, filename)
	
	mu_list = pd.concat([mu_list, alpha_mu_list], ignore_index=True)

# Plot confidence intervals
colors = cm.cool(np.linspace(0, 1, len(alpha_list)))
for alpha, color in zip(alpha_list, colors):
	alpha_subset = mu_list[mu_list['alpha'] == alpha]
		
	plt.plot(alpha_subset['x1'], alpha_subset['mu'], color=color)
	plt.plot(alpha_subset['x0'], alpha_subset['mu'], color=color)
	plt.fill_betweenx(alpha_subset['mu'], alpha_subset['x1'], alpha_subset['x0'], 
			color=color, alpha=0.3, label=f'alpha = {alpha:.4f}')

#plt.xlim(-10,10)
plt.ylim(0,8)
plt.xlabel('$X$')
plt.ylabel('$\mu$')
plt.title(f'Feldman-Cousins Confidence Intervals')

plt.legend()
plt.tight_layout()
plt.savefig('FC-Confidence-Intervals.pdf')
plt.show()


# Plot confidence intervals
colors = cm.cool(np.linspace(0, 1, len(alpha_list)))
for alpha, color in zip(alpha_list, colors):
	alpha_subset = mu_list[mu_list['alpha'] == alpha]
		
	# Determine [mu_0, mu_1] for given x_0
	mu0, mu1 = fc1(x_0, mu_list, alpha)
	
	plt.plot(alpha_subset['x1'], alpha_subset['mu'], color=color)
	plt.plot(alpha_subset['x0'], alpha_subset['mu'], color=color)
	plt.fill_betweenx(alpha_subset['mu'], alpha_subset['x1'], alpha_subset['x0'], 
			color=color, alpha=0.3, label=f'alpha = {alpha:.4f}: [$\mu_0$, $\mu_1$] = [{mu0:.3f}, {mu1:.3f}]')
	plt.scatter([x_0, x_0], [mu0, mu1], s=60)

# Plot x_0
plt.axvline(x_0, label=f'$x_0$={x_0:.3f}')

#plt.xlim(-10,10)
plt.ylim(0,8)
plt.xlabel('$X$')
plt.ylabel('$\mu$')
plt.title(f'Feldman-Cousins Confidence Intervals')

plt.legend()
plt.tight_layout()
plt.savefig(f'FC-Confidence-Intervals_x0_{x_0:.3f}.pdf')
plt.show()
