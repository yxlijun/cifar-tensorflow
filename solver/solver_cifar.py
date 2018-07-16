from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import numpy as np 

class Solver(object):
	"""docstring for Solver"""
	def __init__(self, arg):
		super(Solver, self).__init__()
		self.arg = arg
		