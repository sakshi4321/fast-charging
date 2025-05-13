import numpy as np
import argparse
from sim_with_seed import sim
import pickle
import os
import csv

class BayesGap(object):
	"""
	BayesGap is a class designed for sequential decision-making in Bayesian Optimization, focusing on efficiently selecting optimal parameter configurations.
	Parameters:
	- args (Namespace): An object containing configuration options such as file paths and optimization parameters.
	Processing Logic:
	- Constructs and initializes various file paths needed for storing policy and arm data across rounds.
	- Retrieves and sets up parameter spaces and design matrices for processing using specialized matrix transformation functions.
	- Establishes statistical metrics and attributes necessary for Bayesian optimization, including likelihood standard deviation and confidence intervals.
	"""

	def __init__(self, args):
		"""Initializes the class instance with given configuration parameters and sets up various file paths and attributes.
		Parameters:
		- args (Namespace): An argument namespace containing necessary configuration options and paths.
		Returns:
		- None: This constructor does not return any value.
		Processing Logic:
		- Constructs file paths for policy, previous round arm bounds, early prediction results, current arm bounds, and next batch based on provided round index.
		- The parameter space and design matrix are initialized using dedicated functions returning their respective data structures.
		- Sets instance attributes for dimensions of the design matrix, batch size, budget, round index, and statistical metrics such as likelihood standard deviation, initial beta, and epsilon."""

		self.policy_file = os.path.join(args.data_dir, args.policy_file)
		self.prev_arm_bounds_file = os.path.join(args.data_dir, args.arm_bounds_dir, str(args.round_idx-1) + '.pkl') # note this is for previous round
		self.prev_early_pred_file = os.path.join(args.data_dir, args.early_pred_dir, str(args.round_idx-1) + '.csv') # note this is for previous round
		self.arm_bounds_file = os.path.join(args.data_dir, args.arm_bounds_dir, str(args.round_idx) + '.pkl')
		self.next_batch_file = os.path.join(args.data_dir, args.next_batch_dir, str(args.round_idx) + '.csv')

		self.param_space = self.get_parameter_space()
		self.num_arms = self.param_space.shape[0]

		self.X = self.get_design_matrix(args.gamma)

		self.num_dims = self.X.shape[1]
		self.batch_size = args.bsize
		self.budget = args.budget
		self.round_idx = args.round_idx

		self.sigma = args.likelihood_std
		self.beta = args.init_beta
		self.epsilon = args.epsilon

		self.standardization_mean = args.standardization_mean
		self.standardization_std = args.standardization_std

		self.eta = self.standardization_std

		pass

	def get_design_matrix(self, gamma):
		"""Generate the design matrix using kernel approximation techniques.
		Parameters:
		- gamma (float): The parameter for the RBF kernel sampler affecting the spread of the kernel.
		Returns:
		- ndarray: A transformed design matrix using Nystroem kernel approximation.
		Processing Logic:
		- Uses Nystroem approximation to map the parameter space into a higher dimensional feature space.
		- The `n_components` of the transformation is set to the number of arms.
		- Ensures reproducibility by setting `random_state` to 1."""

		from sklearn.kernel_approximation import (RBFSampler, Nystroem)
		param_space = self.param_space
		num_arms = self.num_arms

		feature_map_nystroem = Nystroem(gamma=gamma, n_components=num_arms, random_state=1)
		X = feature_map_nystroem.fit_transform(param_space)
		return X

	def run(self):
		"""Run a sequential decision-making process to select the best set of parameters or "arms" based on prior data, updating bounds and predictions iteratively.
		Parameters:
		- self: An instance of the class containing parameters and methods required for execution.
		Returns:
		- tuple or None: Parameters of the best arm selected from the parameter space, or None if it is the initial round.
		Processing Logic:
		- In the first round, initialize upper and lower bounds and a list of proposal arms and gaps.
		- In subsequent rounds, load the previous round's data, update the confidence bounds, and calculate the next set of arms based on the highest expected reward.
		- Adjusts confidence bounds based on prior iteration information.
		- Utilizes sub-functions to select arms based on confidence intervals and predictions."""

		prev_arm_bounds_file = self.prev_arm_bounds_file
		prev_early_pred_file = self.prev_early_pred_file
		arm_bounds_file = self.arm_bounds_file
		next_batch_file = self.next_batch_file

		num_arms = self.num_arms
		batch_size = self.batch_size
		epsilon = self.epsilon
		X = self.X
		round_idx = self.round_idx
		param_space = self.param_space

		def find_J_t(carms):
			"""Identifies the index with the smallest value in a list of calculated bounds and returns the index and the minimum value.
			Parameters:
			- carms (list): List of arms that need to be considered when calculating bounds.
			Returns:
			- tuple: A tuple containing the index with the smallest bound difference (J_t) and the minimum bound difference value (min_B_k_t).
			Processing Logic:
			- The function computes B_k_t values for arms in 'carms' and sets to infinity for others.
			- Uses `np.delete` to omit the current arm when calculating maximum of other arms' upper bounds.
			- Converts the list B_k_ts to NumPy array for vectorized subtraction and finding minimum value.
			- Returns the index of the arm with the smallest difference between modified upper and lower bounds."""

			B_k_ts = []
			for k in range(num_arms):
				if k in carms:
					temp_upper_bounds = np.delete(upper_bounds, k)
					B_k_t = np.amax(temp_upper_bounds)
					B_k_ts.append(B_k_t)
				else:
					B_k_ts.append(np.inf)

			B_k_ts = np.array(B_k_ts) - np.array(lower_bounds)
			J_t = np.argmin(B_k_ts)
			min_B_k_t = np.amin(B_k_ts)
			return J_t, min_B_k_t


		def find_j_t(carms, preselected_arm):
			"""Find the index of the arm with the maximum upper bound, excluding a preselected arm.
			Parameters:
			- carms (list): List of integers representing arms to consider.
			- preselected_arm (int): Index of the arm to exclude from consideration.
			Returns:
			- int: Index of the arm with the highest upper bound from the list of candidate arms.
			Processing Logic:
			- Generate upper bounds for each candidate arm that is not the preselected arm.
			- Use negative infinity as the upper bound for non-candidate arms and the preselected arm, ensuring they are excluded.
			- Determine the arm with the highest upper bound using np.argmax."""

			U_k_ts = []
			for k in range(num_arms):
				if k in carms and k != preselected_arm:
					U_k_ts.append(upper_bounds[k])
				else:
					U_k_ts.append(-np.inf)

			j_t = np.argmax(np.array(U_k_ts))

			return j_t


		def get_confidence_diameter(k):

			return upper_bounds[k] - lower_bounds[k]

		if round_idx == 0:
			X_t = []
			Y_t = []
			proposal_arms = [] 
			proposal_gaps = []
			beta = self.beta
			upper_bounds, lower_bounds = self.get_posterior_bounds(beta)
			best_arm_params = None
		else:

			# load proposal_arms, proposal_gaps, X_t, Y_t, beta for previous round in bounds/<round_idx-1>.pkl
			with open(prev_arm_bounds_file, 'rb') as infile:
				proposal_arms, proposal_gaps, X_t, Y_t, beta = pickle.load(infile)

			# update beta for this round
			beta = np.around(beta * epsilon, 4)

			# get armidx of batch policies and early predictions for previous round in pred/<round_idx-1>.csv

			with open(prev_early_pred_file, 'r', encoding='utf-8-sig') as infile:
				reader = csv.reader(infile, delimiter=',')
				early_pred = np.asarray([list(map(float, row)) for row in reader])
			print('Early predictions')
			print(early_pred)
			print()
			print('Standardized early predictions')
			early_pred[:, -1] = early_pred[:, -1] - self.standardization_mean
			print(early_pred)
			print()

			batch_policies = early_pred[:, :3]
			batch_arms = [param_space.tolist().index(policy) for policy in batch_policies.tolist()]
			X_t.append(X[batch_arms])

			batch_rewards = early_pred[:, 4].reshape(-1, 1) # this corresponds to 5th column coz we are supposed to ignore the 4th column
			Y_t.append(batch_rewards)

			np_X_t = np.vstack(X_t)
			np_Y_t = np.vstack(Y_t)
			upper_bounds, lower_bounds = self.get_posterior_bounds(beta, np_X_t, np_Y_t)
			J_prev_round = proposal_arms[round_idx-1]
			temp_upper_bounds = np.delete(upper_bounds, J_prev_round)
			B_k_t = np.amax(temp_upper_bounds) - lower_bounds[J_prev_round]
			proposal_gaps.append(B_k_t)
			best_arm = proposal_arms[np.argmin(np.array(proposal_gaps))]
			best_arm_params = param_space[best_arm]

		print('Arms with (non-standardized) upper bounds, lower bounds, and mean (upper+lower)/2lifetimes')
		nonstd_upper_bounds = upper_bounds+self.standardization_mean
		nonstd_lower_bounds = lower_bounds+self.standardization_mean
		for ((policy_id, policy_param), ub, lb, mean) in zip(enumerate(param_space), nonstd_upper_bounds, nonstd_lower_bounds, (nonstd_upper_bounds+nonstd_lower_bounds)/2):
			print(policy_id, policy_param, ub, lb, mean, sep='\t')
		with open(arm_bounds_file[:-4]+'_bounds.pkl', 'wb') as outfile:
			pickle.dump([param_space, nonstd_upper_bounds, nonstd_lower_bounds, (nonstd_upper_bounds+nonstd_lower_bounds)/2], outfile)

		print('Round', round_idx)
		print('Current beta', beta)
		batch_arms = []
		candidate_arms = list(range(num_arms)) # an extension of Alg 1 to batch setting, don't select the arm again in same batch
		for batch_elem in range(batch_size):
			J_t, _ = find_J_t(candidate_arms)
			j_t = find_j_t(candidate_arms, J_t)
			s_J_t = get_confidence_diameter(J_t)
			s_j_t = get_confidence_diameter(j_t)
			a_t = J_t if s_J_t >= s_j_t else j_t

			if batch_elem == 0:
				proposal_arms.append(J_t)
			batch_arms.append(a_t)
			candidate_arms.remove(a_t)

		print('Policy indices selected for this round:', batch_arms)

		# save proposal_arms, proposal_gaps, X_t, Y_t, beta for current round in bounds/<round_idx>.pkl
		with open(arm_bounds_file, 'wb') as outfile:
			pickle.dump([proposal_arms, proposal_gaps, X_t, Y_t, beta], outfile)

		# save policies corresponding to batch_arms in batch/<round_idx>.csv
		batch_policies = [param_space[arm] for arm in batch_arms]
		with open(next_batch_file, 'w') as outfile:
			writer = csv.writer(outfile)
			writer.writerows(batch_policies)

		return best_arm_params

	def posterior_theta(self, X_t, Y_t):
		"""Compute the posterior distribution parameters for theta given input data.
		Parameters:
		- X_t (np.ndarray): Feature matrix at time t, with shape (n_samples, num_dims). Represents the input data.
		- Y_t (np.ndarray): Target vector at time t, with shape (n_samples,). Represents the observed outputs.
		Returns:
		- tuple: A tuple containing two elements:
		- np.ndarray: Posterior mean vector of theta, with shape (num_dims,).
		- np.ndarray: Posterior covariance matrix of theta, with shape (num_dims, num_dims).
		Processing Logic:
		- If `X_t` is None, the function returns prior mean and covariance parameters.
		- Posterior mean is computed using the joint product of posterior covariance, transpose of X_t, and Y_t, scaled by noise variance.
		- Posterior covariance is calculated using the inverse of the sum of scaled dot product and prior precision."""

		num_dims = self.num_dims
		sigma = self.sigma
		eta = self.eta
		prior_mean = np.zeros(num_dims)

		prior_theta_params = (prior_mean, eta * eta * np.identity(num_dims))

		if X_t is None:
			return prior_theta_params

		posterior_covar = np.linalg.inv(np.dot(X_t.T, X_t) / (sigma * sigma) + np.identity(num_dims) / (eta * eta))
		posterior_mean = np.linalg.multi_dot((posterior_covar, X_t.T, Y_t))/ (sigma * sigma)

		posterior_theta_params = (np.squeeze(posterior_mean), posterior_covar)
		return posterior_theta_params


	def marginal_mu(self, posterior_theta_params):
		"""Compute the marginal mean and variance for the given posterior distribution parameters.
		Parameters:
		- posterior_theta_params (tuple): A tuple containing the posterior mean and covariance matrix.
		Returns:
		- tuple: A tuple containing the calculated marginal mean and marginal variance.
		Processing Logic:
		- The marginal mean is computed as the dot product between matrix X and the posterior mean vector.
		- The marginal variance is calculated by multiplying the dot product of X and the posterior covariance matrix with X, then summing the result across axis 1."""

		X = self.X
		posterior_mean, posterior_covar = posterior_theta_params

		marginal_mean = np.dot(X, posterior_mean) 
		marginal_var = np.sum(np.multiply(np.dot(X, posterior_covar), X), 1)
		marginal_mu_params = (marginal_mean, marginal_var)

		return marginal_mu_params

	def get_posterior_bounds(self, beta, X=None, Y=None):
		"""
		Returns upper and lower bounds for all arms at every time step.
		"""

		posterior_theta_params = self.posterior_theta(X, Y)
		marginal_mu_params = self.marginal_mu(posterior_theta_params)
		marginal_mean, marginal_var = marginal_mu_params

		upper_bounds = marginal_mean + beta * np.sqrt(marginal_var)
		lower_bounds = marginal_mean - beta * np.sqrt(marginal_var)

		upper_bounds = np.around(upper_bounds, 4)
		lower_bounds = np.around(lower_bounds, 4)

		return (upper_bounds, lower_bounds)


	def get_parameter_space(self):

		policies = np.genfromtxt(self.policy_file,
				delimiter=',', skip_header=0)
		np.random.shuffle(policies)

		return policies[:, :3]

def parse_args():
	"""Parse command-line arguments for a closed-loop optimization framework.
	Parameters:
	- None
	Returns:
	- argparse.Namespace: A namespace containing parsed command-line arguments.
	Processing Logic:
	- Allows specification of input files and directories for policy, data, arm bounds, early prediction, and next batch.
	- Supports setting initial random seed, time budget, batch size, and exploration parameters.
	- Handles Gaussian kernel parameters and standardization statistics for lifetime data."""

	parser = argparse.ArgumentParser(description='Closed-Loop Optimization with early prediction and Bayes Gap.')

	parser.add_argument('--policy_file', nargs='?', default='policies_all.csv')
	parser.add_argument('--data_dir', nargs='?', default='data/')
	parser.add_argument('--log_file', nargs='?', default='log.csv')
	parser.add_argument('--arm_bounds_dir', nargs='?', default='bounds/')
	parser.add_argument('--early_pred_dir', nargs='?', default='pred/')
	parser.add_argument('--next_batch_dir', nargs='?', default='batch/')
	parser.add_argument('--round_idx', default=0, type=int)

	parser.add_argument('--seed', default=0, type=int,
						help='Seed for random number generators')
	parser.add_argument('--budget', default=8, type=int,
						help='Time budget')
	parser.add_argument('--bsize', default=48, type=int,
						help='batch size')

	parser.add_argument('--gamma', default=1, type=float,
						help='kernel bandwidth for Gaussian kernel')
	parser.add_argument('--likelihood_std', default=164, type=float,
						help='standard deviation for the likelihood std')
	parser.add_argument('--init_beta', default=5.0, type=float,
						help='initial exploration constant in Thm 1')
	parser.add_argument('--epsilon', default=0.5, type=float,
						help='decay constant for exploration')

	parser.add_argument('--standardization_mean', default=947.0, type=float,
						help='mean lifetime from batch8')
	parser.add_argument('--standardization_std', default=164, type=float,
						help='std lifetime from batch8')

	return parser.parse_args()


def main():
	"""Main function to execute the Bayesian optimization process.
	Parameters:
	- None
	Returns:
	- None
	Processing Logic:
	- Sets the seed and print options for numerical operations.
	- Checks the existence of required directories within the specified data directory.
	- Initializes a Bayesian optimization agent and runs the process to determine best parameters.
	- If not the first round, simulates the lifetime of the best 'arm' using the data simulator and logs the outcome."""

	args = parse_args()

	np.random.seed(args.seed)
	np.set_printoptions(threshold=np.inf)

	assert (os.path.exists(os.path.join(args.data_dir, args.arm_bounds_dir)))
	assert (os.path.exists(os.path.join(args.data_dir, args.early_pred_dir)))
	assert (os.path.exists(os.path.join(args.data_dir, args.next_batch_dir)))

	agent = BayesGap(args)
	best_arm_params = agent.run()

	if args.round_idx != 0:
		print('Best arm until round', args.round_idx-1, 'is', best_arm_params)
		lifetime_best_arm = sim(best_arm_params[0], best_arm_params[1], best_arm_params[2], variance=False)
		print('Lifetime of current best arm as per data simulator:', lifetime_best_arm)

	# Log the best arm at the end of each round
	log_path = os.path.join(args.data_dir, args.log_file)
	with open(log_path, "a") as log_file:
		print('Logging data...')
		if args.round_idx == 0:
			log_file.write(str(args.init_beta)  + ',' +
                          str(args.gamma)      + ',' +
                          str(args.epsilon)    + ',' +
                          str(args.seed))
		elif args.round_idx == args.budget:
         		log_file.write(',' + str(lifetime_best_arm) + '\n')
		else:
         		log_file.write(',' + str(lifetime_best_arm))

if __name__ == '__main__':

	main()
