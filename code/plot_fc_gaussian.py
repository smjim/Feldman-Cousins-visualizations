import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Measured X value
x_0 = 2.5

# Confidence level
alpha = 0.9

# Step for Neyman interval visualization
step = 0.01

# Measurement and signal values
x_vals = np.arange(-10,10,0.01)
#mu_vals = np.arange(0,1,0.001)
mu_vals = np.arange(0,5,0.01)

# Nonnegative mu maximizing P(x,mu)
def mu_best(x):
	return max(0,x)

# Probability of x given mu
def P(x,mu):
	return np.exp(-0.5*(x-mu)**2)/np.sqrt(2*np.pi)

# How good does mu describe x?
def R(x,mu):
	return P(x,mu)/ P(x,mu_best(x))

# Plot Neyman interval construction
def plot_neyman_interval(x_vals, mu, i, x_bounds=False):
	blue1 = "#4C72B0"
	blue2 = '#0000FF'

	fig, ax1 = plt.subplots(1, 1, figsize=(8,8))

	# Fill between x0, x1 under curve P(x,mu) if arguments are given
	if x_bounds:
		x_vals_cut = [x for x in x_vals if x_bounds[0] <= x <= x_bounds[1]]
		ax1.fill_between(x_vals_cut, [P(x,mu) for x in x_vals_cut], color=blue1, alpha=0.3, label=f'[$x_0$, $x_1$] = [{x_bounds[0]:.3f}, {x_bounds[1]:.3f}]')

	ax1.plot(x_vals, [P(x,mu) for x in x_vals], label=f'P(x|$\mu$={mu:.3f})', color=blue1)
	ax1.plot(x_vals, [P(x,mu_best(x)) for x in x_vals], label='P(x|$\mu_{best}$(x))', color=blue2)
	ax1.plot(x_vals, [R(x,mu) for x in x_vals], label='R(x)', color='black')
	ax1.set_xlabel('X')
	ax1.set_ylabel('P [1]', color='blue')
	ax1.tick_params(axis='y', colors='blue')
	ax1.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
	ax1.legend(loc='upper left')

	ax2 = ax1.twinx()
	ax2.plot(x_vals, [mu_best(x) for x in x_vals], label='$\mu_{best}(x)$', color='red')
	ax2.set_ylabel('$\mu$', color='red')
	ax2.tick_params(axis='y', colors='red')
	ax2.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
	ax2.legend(loc='upper right')

	plt.title('Construction of Neyman Interval')
	plt.tight_layout()	
	plt.savefig(f'../output/{i:04d}_neyman_interval_{mu:.4f}.png')
	plt.clf()
	#plt.show()

# Feldman-cousins Neyman construction of confidence intervals
def fc0(x_vals, mu, i, alpha=0.95):
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
			# Plot Neyman interval
			if int(mu*1e4) % int(step*1e4) == 0: # annoying floating point precision thing makes 1e4 necessary for int calculation
				plot_neyman_interval(x_vals, mu, i, x_bounds=(x0, x1))
			return x0, x1

	# If no interval satisfies the condition, return the last calculated interval
	return x0, x1

# Feldman-Cousins using Neyman interval to determine mu Confidence interval
def fc1(x_0, mu0_list, mu1_list):
	"""
	Find mu0 and mu1 such that 
		mu0_list[mu=mu0, x=x_0]
		mu1_list[mu=mu1, x=x_0]
	Linearly interpolate between x vals to find best mu value 
	"""

	def find_corresponding_a(xy_list, x_0):
		# Extract a and b values from the xy_list
		a_values = xy_list[:, 0]
		b_values = xy_list[:, 1]
	
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

	mu0 = find_corresponding_a(mu0_list, x_0)
	mu1 = find_corresponding_a(mu1_list, x_0)
	
	return mu0, mu1

# Generate confidence bands
mu0_list = []
mu1_list = []
for i, mu in enumerate(mu_vals):
	# Determine [x0, x1] such that
	x0, x1 = fc0(x_vals, mu, i, alpha)
	# [mu, x0], [mu, x1]
	mu0_list.append(np.array([mu, x1]))
	mu1_list.append(np.array([mu, x0]))

# Plot confidence intervals
mu0_list = np.array(mu0_list)
mu1_list = np.array(mu1_list)
plt.plot(mu1_list[:,1], mu1_list[:,0], color='blue')
plt.plot(mu0_list[:,1], mu0_list[:,0], color='blue')
plt.fill_betweenx(mu_vals, mu1_list[:,1], mu0_list[:,1], color='blue', alpha=0.3)

#plt.xlim(-10,10)
plt.ylim(0,5)
plt.xlabel('$X$')
plt.ylabel('$\mu$')
plt.title(f'Feldman-Cousins Confidence Intervals, alpha={alpha}')

plt.tight_layout()
plt.show()

# Determine [mu_0, mu_1] for given x_0
mu0, mu1 = fc1(x_0, mu0_list, mu1_list)

# Plot confidence intervals
mu0_list = np.array(mu0_list)
mu1_list = np.array(mu1_list)
plt.plot(mu1_list[:,1], mu1_list[:,0], color='blue')
plt.plot(mu0_list[:,1], mu0_list[:,0], color='blue')
plt.fill_betweenx(mu_vals, mu1_list[:,1], mu0_list[:,1], color='blue', alpha=0.3)

# Plot x_0
plt.axvline(x_0, label=f'$x_0$={x_0:.3f}: [$\mu_0$, $\mu_1$] = [{mu0:.3f}, {mu1:.3f}]')
plt.scatter([x_0, x_0], [mu0, mu1], s=60)

#plt.xlim(-10,10)
plt.ylim(0,5)
plt.xlabel('$X$')
plt.ylabel('$\mu$')
plt.title(f'Feldman-Cousins Confidence Intervals, alpha={alpha}')

plt.legend()
plt.tight_layout()
plt.show()
