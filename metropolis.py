#三次元正規分布のサンプリング
#提案分布g(x)が詳細釣り合いを満たす場合(正規分布)を用いる

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
from scipy.stats import norm,uniform,multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

N_sample = 500
cov1_2 = 0.3
cov2_3 = 0.5
cov3_1 = 0.7
sigma = np.array([[1,cov1_2,cov3_1],[cov1_2,1,cov2_3],[cov3_1,cov2_3,1]])
sigma_inv = np.linalg.inv(sigma)
x,y,z = [],[],[]

def P(X):
    global sigma_inv
    x = np.array(X)
    return np.exp((-0.5)*np.dot(np.dot(x,sigma_inv),x))

def update(X):
    ret = [0]*3
    for i in range(3):
        ret[i] = X[i] + float(norm.rvs(loc=0.0,scale=1.0))
    return ret

cur_x = [0.0,0.0,0.0]
#burn in 
for i in range(10):
    next_x = update(cur_x)
    r = P(next_x) / P(cur_x)
    if uniform.rvs(loc=0,scale=1.0) < r:
        cur_x = next_x

cnt = 0
ims = []
fig = plt.figure(figsize=(10,10)) 
ax = Axes3D(fig)

x.append(cur_x[0])
y.append(cur_x[1])
z.append(cur_x[2])

while cnt < (N_sample-1):
    next_x = update(cur_x)
    r = P(next_x) / P(cur_x)
    
    if uniform.rvs(loc=0,scale=1.0) < r:
        cur_x = next_x
        x.append(cur_x[0])
        y.append(cur_x[1])
        z.append(cur_x[2])
        cnt += 1
         
for i in range(len(x)):
    ims.append(ax.plot(x[:i+1],y[:i+1],z[:i+1],marker='o',color="black",linestyle="dashed"))
    

ax.set_title('3d norm scatter g(x)=norm(scale=2.0)')
ani = animation.ArtistAnimation(fig, ims, interval=100)
#plt.show()
ani.save("metropolis_scale1.0.gif", writer="imagemagick", fps=5, dpi=64)