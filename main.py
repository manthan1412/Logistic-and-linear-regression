## python3

from LinearClassification import PerceptronLearning as PL
from LinearClassification import PocketAlgorithm as PA
from LinearRegression import LinearRegression as LR
from LogisticRegression import LogisticRegression as LoR
from copy import deepcopy


def read_classification_data(filename):
	promissed_data = []
	classification_data = []
	with open(filename) as f:

		for line in f:
			if line:
				row = list(map(float, line.strip().split(',')))
				promissed_row = row[:4]
				classification_row = row[:3]
				classification_row.append(row[4])
				promissed_data.append(promissed_row)
				classification_data.append(classification_row)
	return promissed_data, classification_data

def perceptron_learning(data=None, alpha=0.001):
	print("Performing Perceptron Learning...")
	pl = PL(data=data, alpha=alpha)
	print("Weights: ", pl.weights())
	print("Accuracy: ", pl.accuracy())

def pocket_algorithm(data=None, alpha=0.001, iterations=7000):
	print("\nPerforming Pocket Algorithm...")
	pa = PA(data=data, alpha=alpha, iterations=iterations)
	print("Weights: ", pa.weights())
	print("Accuracy: ", pa.accuracy())
	pa.plot()

def linear_regression(data=None):
	print("\nPerforming Linear Regression...")
	lr = LR(data=data)
	print("Weights: ", lr.W_opt())

def logistic_regression(data=None, eta=0.002, iterations=7000):
	print("\nPerforming Logistic Regression...")
	lr = LoR(data=data, eta=eta, iterations=iterations)
	print("Weights: ", lr.weights())
	print("Accuracy: ", lr.accuracy())

if __name__ == "__main__":
	promissed_data, classification_data = read_classification_data('classification.txt')
	perceptron_learning(promissed_data, 0.001)
	classification_copy = deepcopy(classification_data)
	pocket_algorithm(classification_data, 0.1, 7000)
	# print(promissed_data)
	# print(classification_data)
	# pl = PL(data=promissed_data, alpha=0.001)
	# print(pl._get_data())

	regression_data = []
	with open('linear-regression.txt') as f:
		for line in f:
			if line:
				row = list(map(float, line.strip().split(',')))
				regression_data.append(row)
	# print(regression_data)
	linear_regression(regression_data)
	logistic_regression(classification_copy, 0.02, 7000)
