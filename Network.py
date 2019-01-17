import math
import random
from Neuron import Neuron
from Neuron import getNormalised
import pandas as pd

class Network():

	def __init__(self,numNeurons):
		self.neurons = []
		self.inputs = []
		for i in range(numNeurons):
			self.neurons.append(Neuron(weights=[random.uniform(0,1),
											random.uniform(0,1),random.uniform(0,1)]))

	def train(self,inputs,lr):
		normalised_inputs = getNormalised(inputs)
		self.inputs.append(normalised_inputs)
		largestNum = 0
		posWithLargestScore = 0
		for pos,neuron in enumerate(self.neurons):
			netScore = neuron.calcNet(normalised_inputs)
			if netScore > largestNum:
				largestNum = netScore
				posWithLargestScore = pos

		winningNeuron = self.neurons[posWithLargestScore]
		winningNeuron.updateWeights(normalised_inputs,lr)

	def __str__(self):
		return "<Network w/ neurons:\n {}\n and inputs: \n{}>".format(','.join([str(n) for n in self.neurons]),self.inputs)


def main():

	network = Network(4)
	lr = 0.1
	df = pd.read_csv('data.csv')
	df.dropna(inplace=True)
	for index,row in df.iterrows():
		network.train(row,lr)
	print(network)

if __name__ == '__main__':
	main()