import numpy as np
from numpy.random import randn, randint
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt



class LinearClassification(object):

	def __init__(self, data, alpha):
		self.__data = data
		if not data or type(data) != list or len(data) == 0:
			raise ValueError('No data or data with length zero given')
		if not alpha:
			print("Using default value of alpha=0.002")
			alpha = 0.002
		self._alpha = alpha

		self.__dim = len(data[0])
		# self.__numpy_data = []
		self._numpy_data = np.array([[] for _ in range(self.__dim)])
		self._target = []
		self.__N = len(self.__data)
		for x in self.__data:
			x.insert(0, 1.0)
			self._numpy_data = np.c_[ self._numpy_data, np.array(x[:-1]) ]
			self._target.append(x[-1])
		# print(self._numpy_data[:,1])
		# print(self._target)
		self._target = np.array(self._target)
		self.__lowest = len(data)
		self.__init_W()

	def __init_W(self):
		self._W = randn(self.__dim)
		# print(self._W)

	def _update_W(self, predicted):
		accuracy = (predicted == self._target)
		violated_indices = np.where(accuracy == False)
		if len(violated_indices[0]) == 0:
			return False, 0

		# if self.__lowest > len(violated_indices[0]):
		# 	self.__lowest = len(violated_indices[0])
		# 	with open('temp.txt', 'a') as f:
		# 		f.write(str(self.__lowest) + "\n")
		# 		if self.__lowest < 6:
		# 			f.write(str(violated_indices[0].tolist()) + "\n")

		index = violated_indices[0][randint(len(violated_indices[0]))]
		# print(index)
		# print(self._numpy_data[:, index])
		if predicted[index] == 1.0:
			print(len(violated_indices[0]), "-")
			self._W -= self._alpha*self._numpy_data[:, index]
		else:
			print(len(violated_indices[0]), "+")
			self._W += self._alpha*self._numpy_data[:, index]
		# print(self._W)
		return True, len(violated_indices[0])

	def _next_iteration(self):
		predicted = []
		for i in range(self.__N):
			predicted.append(np.sign(self._W.dot(self._numpy_data[:, i])))
		# self.__update_W(np.array(predicted))
		return predicted

	def _get_data(self):
		return self.__data


class PerceptronLearning(LinearClassification):

	def __init__(self, data, alpha=None):
		LinearClassification.__init__(self, data, alpha)
		# print(self._numpy_data[:, 0:2])
		# print(self._target[0])
		# print(self._W)
		updated = True
		i = 1
		
		while updated:
			self.__predicted = self._next_iteration()
			updated, missclassified = self._update_W(np.array(self.__predicted))
			print(i)
			i+=1
		print(self._W)
		with open('temp.txt', 'a') as f:
			f.write(str(self._W.tolist()) + "\n")


class PocketAlgorithm(LinearClassification):

	def __init__(self, data, alpha=None, iterations=7000):
		LinearClassification.__init__(self, data, alpha)

		updated = True

		self.__best_W = None
		self.__lowest_missclassified = len(data)
		self.__missclassified = []

		for i in range(iterations):
			self.__predicted = self._next_iteration()
			updated, missclassified = self._update_W(np.array(self.__predicted))
			self.__missclassified.append(missclassified)
			if self.__lowest_missclassified > missclassified:
				self.__lowest_missclassified = missclassified
				self.__best_W = self._W
			print(i)
		# print(self._W)
		with open('pocket.txt', 'a') as f:
			f.write(str(self.__best_W.tolist()) + "\n" + str(self.__lowest_missclassified) + "\n")


	def plot(self):
		plt.plot(self.__missclassified)
		plt.xlabel("iterations")
		plt.ylabel("Missclassified points")
		plt.show()