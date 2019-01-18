import math
import random
import pandas as pd
from Neuron import Neuron
from Neuron import getNormalised
from Visualiser import Visualiser

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
		inputColours = {0:'r',1:'b',2:'g'}
		visualiser = Visualiser(size=111)
		for index,neuron in enumerate(self.neurons):
			visualiser.add(neuron.weights[0],neuron.weights[1],neuron.weights[2],'y','^')

		for index,norm_input in all_inputs.iterrows():
			winningNeuron = self.closestNeuron(getNormalised(norm_input))
			visualiser.add(norm_input[0],norm_input[1],norm_input[2],inputColours[winningNeuron],'o')
			clustered_dict[winningNeuron].append(norm_input)#[str(i) for i in norm_input]) use for debugging
		
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

def main():
	network = Network(numNeurons=3)
	lr = 0.1
	epochs = 600
	df = pd.read_csv('data.csv',header=None)
	df.dropna(inplace=True)
	for i in range(epochs):
		for index,row in df.iterrows():
			network.train(row,lr)

	clustered_dict = network.predict(df)
	print(network)

if __name__ == '__main__':
	main()

# if 4 neurons are used then one is left unused as a cluster i.e it is extra
# if 3 neurons all are used
