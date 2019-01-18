from Neuron import Neuron
from Neuron import getNormalised
from Visualiser import Visualiser
import math
import random

class Network():

	def __init__(self,numNeurons):
		self.neurons = []
		for i in range(numNeurons):
			self.neurons.append(Neuron(weights=[random.uniform(0,1),
											random.uniform(0,1),random.uniform(0,1)]))

	def train(self,inputs,lr):
		normalised_inputs = getNormalised(inputs)
		posWithLargestScore = self.closestNeuron(normalised_inputs)
		winningNeuron = self.neurons[posWithLargestScore]
		winningNeuron.updateWeights(normalised_inputs,lr)

	def predict(self,all_inputs):
		clustered_dict = {index:[] for index,neuron in enumerate(self.neurons)} #initialise positions
		inputColours = {0:'r',1:'b',2:'g',3:'c'}
		visualiser = Visualiser(size=111)
		for index,neuron in enumerate(self.neurons):
			visualiser.add(neuron.weights[0],neuron.weights[1],neuron.weights[2],[(0,0,0)],'^')

		for index,norm_input in all_inputs.iterrows():
			winningNeuronPos = self.closestNeuron(getNormalised(norm_input))
			winningNeuron = self.neurons[winningNeuronPos]
			visualiser.add(norm_input[0],norm_input[1],norm_input[2],
				[(math.fabs(winningNeuron.weights[0]),math.fabs(winningNeuron.weights[1]),math.fabs(winningNeuron.weights[2]))],'o')
			clustered_dict[winningNeuronPos].append(norm_input)#[str(i) for i in norm_input]) use for debugging
		
		visualiser.show()
		return clustered_dict

	def closestNeuron(self,normalised_inputs):
		largestNum = 0
		posWithLargestScore = 0
		for pos,neuron in enumerate(self.neurons):
			netScore = neuron.calcNet(normalised_inputs)
			if netScore > largestNum:
				largestNum = netScore
				posWithLargestScore = pos
		return posWithLargestScore

	def __str__(self):
		return "<Network w/ neurons:\n {}\n>".format(','.join([str(n) for n in self.neurons]))
