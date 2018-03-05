import numpy as np
from numpy.random import randn, randint
from numpy.linalg import inv
from numpy import exp

class LogisticRegression(object):

	def __init__(self, data, eta=None, iterations=7000):
		self.__data = data
		if not data or type(data) != list or len(data) == 0:
			raise ValueError('No data or data with length zero given')
		if not eta:
			print("Using default value of eta=0.002")
			eta = 0.002

		self.__eta = eta
		self.__dim = len(data[0])
		self.__iterations = iterations
		
		self._numpy_data = np.array([[] for _ in range(self.__dim)])
		self._y = []
		self.__N = len(self.__data)
		for x in self.__data:
			x.insert(0, 1.0)
			self._numpy_data = np.c_[ self._numpy_data, np.array(x[:-1]) ]
			self._y.append(x[-1])
		# print(self._numpy_data[:,1])
		# print(self._target)
		self._y = np.array(self._y)
		# print(self._numpy_data.shape, self._y.shape)

		self.__init_W()
		self.__regression()

	def __init_W(self):
		self.__W = randn(self.__dim)


	def __regression(self):
		for _ in range(self.__iterations):
			dw_sum = 0
			for i in range(self.__N):
				w_x = self.__W.dot(self._numpy_data[:, i])

				y_w_x = self._y[i]*w_x
				# print(y_w_x)
				# y_w_x = (self._target.dot(self.__W)).dot(self.__numpy[:, i])
				e_y_w_x = exp(-1*y_w_x)
				# print(e_y_w_x)
				denominator = e_y_w_x + 1
				nominator = e_y_w_x*-1*(self._y[i] * self._numpy_data[:, i])
				# print(nominator)
				# print(denominator)
				nominator /= denominator
				dw_sum += nominator
			dw = dw_sum / self.__N
			# print(dw)
			# print(self.__W)
			self.__W -= self.__eta*dw
			# print(self.__W)
			break

