import math

class Neuron():
	def __init__(self,weights):
		self.weights = weights
		self.net = 0
		self.normaliseWeights()

	def normaliseWeights(self):
		self.weights = getNormalised(self.weights)

	def calcNet(self,normInputs):
		# normalized inputs and normalized weights
		if(len(normInputs) != len(self.weights)):
			raise Exception('Inputs and weights dimensions are unequal')

		self.net = sum([input_num*weight for input_num,weight in zip(normInputs,self.weights)])
		return self.net

	def updateWeights(self,normInputs,lr):
		if(len(normInputs) != len(self.weights)):
			raise Exception('Inputs and weights dimensions are unequal')

		self.weights = [weight+lr*(input_num-weight) for input_num,weight in zip(normInputs,self.weights)]
		return self.weights

	def __str__(self):
		return "<Neuron w/ weights: {}, net: {}>\n".format(','.join(map(str,self.weights)),self.net)

def getNormalised(input_list):
	normDenom = math.sqrt(sum([w**2 for w in input_list]))
	for pos,num in enumerate(input_list):
		input_list[pos] = num/normDenom
	return input_list