import pandas as pd
from Network import Network

'''
A kohonen network implemented by finding the closest neuron via normalisation
and net calculation versus the popular shortest euclidian distance.

TODO: Explore and implement 3D version of Kohonen network.
'''

def main():
	network = Network(numNeurons=4)
	lr = 0.01
	epochs = 500
	df = pd.read_csv('data.csv',header=None)
	df.dropna(inplace=True)
	for i in range(epochs):
		for index,row in df.iterrows():
			network.train(row,lr)

	clustered_dict = network.evaluate(df)
	print(network,"\n\n\n",clustered_dict)

if __name__ == '__main__':
	main()