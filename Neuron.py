import math

'''
The purpose of this class is in order to efficiently manage neurons
and utilising instance variables to track the associated weights and perform
the appropriate calculations while keeping the codebase human readable.
'''

class Neuron():
	# Upon initialising a neuron, the weights are normalised and stored along with the net.
	def __init__(self,weights):
		self.weights = weights
		self.net = 0
		self.normaliseWeights()

	def normaliseWeights(self):
		self.weights = getNormalised(self.weights)

	def calcNet(self,normInputs):
		self.checkDimensions(normInputs);
		self.net = sum([input_num*weight for input_num,weight in zip(normInputs,self.weights)])
		return self.net

	def updateWeights(self,normInputs,lr):
		self.checkDimensions(normInputs);
		self.weights = [weight+lr*(input_num-weight) for input_num,weight in zip(normInputs,self.weights)]
		self.weights = getNormalised(self.weights)
		return self.weights

	def checkDimensions(self,normInputs):
		# Ensures that the dimensions of the inputs and weights match as expected
		if(len(normInputs) != len(self.weights)):
			raise Exception('Inputs and weights dimensions are unequal respectively:',len(normInputs),len(self.weights))

	def __str__(self):
		return "<Neuron w/ weights: {}, net: {}>\n".format(','.join(map(str,self.weights)),self.net)

def getNormalised(input_list):
	normDenom = math.sqrt(sum([w**2 for w in input_list]))
	return [num/normDenom for num in input_list]