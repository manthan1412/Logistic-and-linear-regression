from LinearClassification import PerceptronLearning as PL
from LinearClassification import PocketAlgorithm as PA
from LinearRegression import LinearRegression as LR
from LogisticRegression import LogisticRegression as LoR


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
	pl = PL(data=data, alpha=alpha)

def pocket_algorithm(data=None, alpha=0.001, iterations=7000):
	pa = PA(data=data, alpha=alpha, iterations=iterations)
	pa.plot()

def linear_regression(data=None):
	lr = LR(data=data)
	w_opt = lr.W_opt()
	print(lr)

if __name__ == "__main__":
	promissed_data, classification_data = read_classification_data('classification.txt')
	# perceptron_learning(promissed_data, 0.001)
	pocket_algorithm(classification_data, 0.1, 700)
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

