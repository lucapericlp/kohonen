import pandas as pd
from Network import Network

def main():
	network = Network(numNeurons=4)
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
# if 10 neurons are used, then we have 4 clusters!
