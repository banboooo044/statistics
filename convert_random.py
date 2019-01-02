import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.integrate
from scipy.stats import norm,uniform

class Convert_Random:
	""" 任意の確率関数に従う乱数を生成 """
	def __init__(self,f,Nsim = 100000):
		np.random.seed()
		self.f = f
		self.f_normalized = lambda x: f(x) / scipy.integrate.quad(f, -float('inf'), float('inf'))[0]
		gv = norm(loc=0.0, scale=1.0)
		g = gv.pdf

		xopt = scipy.optimize.fmin(lambda x: - self.f(x) / g(x), 0.0, disp=False)[0]
		M = self.f(xopt) / g(xopt)
		Y = gv.rvs(size=Nsim)
		U = uniform.rvs(size=Nsim) * M * g(Y)
		self.X = Y[U <= f(Y)]

	def generateRandom(self,num=1):
		return np.random.choice(self.X,num)    

	def drawGragh(self,lim = [-10,10,1000]):
		x = np.linspace(lim[0], lim[1], lim[2])
		y = self.f_normalized(x)
		plt.plot(x, y, 'r-', lw=2)
		plt.hist(self.X,bins = 50,normed = True)
		plt.show()

	def showSample(self):
		return len(self.X),self.X

if __name__ == "__main__":
	f = lambda x: np.exp(-x**2 / 2) * (np.sin(6*x)**2 + 3 * np.cos(x)**2 * np.sin(4*x)**2 + 1)
	CV = Convert_Random(f)
	CV.drawGragh()


