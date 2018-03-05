import numpy as np
from numpy.random import randn, randint
from numpy.linalg import inv

class LinearRegression(object):

	def __init__(self, data):
		self.__data = data
		if not data or type(data) != list or len(data) == 0:
			raise ValueError('No data or data with length zero given')

		self.__dim = len(data[0])

		self._D = np.array([[] for _ in range(self.__dim)])
		self._y = []
		self.__N = len(self.__data)
		for x in self.__data:
			x.insert(0, 1.0)
			self._D = np.c_[ self._D, np.array(x[:-1]) ]
			self._y.append(x[-1])
		# print(self._D.shape)
		# print(self._y)
		self._y = np.array(self._y)
		self.__regression()

	def __regression(self):
		inverse = inv(self._D.dot(self._D.T))
		# print(inverse.shape)
		self._W = inverse.dot(self._D)
		self._W = self._W.dot(self._y)
		# print(self._W)


	def W_opt(self):
		return self._W.tolist()

	def __repr__(self):
		return repr(self._W.tolist())