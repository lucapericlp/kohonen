from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class Visualiser():
	def __init__(self,size):
		self.fig = plt.figure()
		self.ax = self.fig.add_subplot(size,projection='3d')
		self.ax.set_xlabel('X Label')
		self.ax.set_ylabel('Y Label')
		self.ax.set_zlabel('Z Label')

	def add(self,x,y,z,colour,marker):
		self.ax.scatter(x,y,z,c=colour,marker=marker)

	def show(self):
		plt.show()