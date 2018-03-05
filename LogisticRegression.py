import numpy as np
from numpy.random import randn, randint
from numpy.linalg import inv

class LogisticRegression(object):

	def __init__(self, data):
		self.__data = data
		if not data or type(data) != list or len(data) == 0:
			raise ValueError('No data or data with length zero given')