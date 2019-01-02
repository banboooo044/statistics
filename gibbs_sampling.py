import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
from scipy.stats import norm,uniform,multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

maxSample = 1000

x = [5.0]
y = [5.0]
# sigma1 = sigma2 = 1,mean1 = mean2 = 0 の2つの標準正規分布からのサンプル。
#ただし、共分散 cov の値によって、グラフが変化する。
sigma = 0.8
for i in range(2*maxSample):
    x.append(norm.rvs(loc=sigma*y[-1],scale=1.0))
    y.append(norm.rvs(loc=sigma*x[-1],scale=1.0))
    
x = x[2::4]
y = y[4::4]
x1 = [5.0]
x2 = [5.0,5.0]
ims = []

fig = plt.figure(figsize=(10,10))
plt.subplot(111)
plt.title('2d norm Gibbs sampling')
for i in range(len(y)):
    x1.extend([x[i]]*2)
    x2.extend([y[i]]*2)
if len(x1) < len(x2):
    x1.append(x1[-1])
elif len(x1) > len(x2):
    x2.append(x2[-1])

for i in range(len(x1)):
	ims.append(plt.plot(x1[:i+1],x2[:i+1],color='black',  linestyle='solid', linewidth = 1.0, marker='o'))

ani = animation.ArtistAnimation(fig, ims, interval=100)
#plt.show()
ani.save("gibbs_sampling.gif", writer="imagemagick", fps=5, dpi=64)