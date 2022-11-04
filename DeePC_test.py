from math import ceil
import numpy as np 
import cvxpy as cp 
import random
import matplotlib.pyplot as plt
from deepc import DeePC
from numpy import linalg
import pandas as pd

# Define system
A = np.array([[1.01,0.01,0],[0.01,1.01,0.01],[0,0.01,1.01]])
B = np.eye(3)
C = np.eye(3)
(n,m) = B.shape                                 # n = number of states, m = number of inputs
p = n                                           # p = number of output
        
# Simulation    
x0 = np.array([[0],[0],[1]])
t_step = 0.1
t_sim = 20 #4.3#20.0                                    # simulation time 
n_sim = ceil(t_sim/t_step)                     # Number of simulation step n_sim >= (m+1)*(T_ini + N + n(B)) - 1
x = x0 
xData = np.empty((n,n_sim+1))
yData = np.empty((n,n_sim+1))
xData[:,[0]] = x0
yData[:,[0]] = x0
uData = np.empty((m,n_sim))

# Generate random input sequence
np.random.seed(1)
u_seq = np.array(np.random.rand(m,n_sim))
# measurement noise
mu = 0 #0.5
sigma = 0.01 # 0.2

# Collect data
for t in range(n_sim):
    u = u_seq[:, t].reshape(m, 1)
    w = np.random.normal(mu, sigma, size=(n, 1))
    x = A@x+B@u
    y = C@x+w
    uData[:,[t]] = u 
    xData[:,[t+1]] = x
    yData[:,[t+1]] = y

# set parameters
params = {}
params['uData'] = uData
params['yData'] = yData[:,1:] #xData[:,1:]
params['N'] = 5
params['Q'] = np.eye(3)
params['R'] = np.eye(3)*10
params['lambda_slack'] = 10**3#10**3# 10**7
params['lambda_g'] = 100#300# 100

# create controller    
controller = DeePC(params)
# controller.create_Hankel_check()

# offline computation
controller.create_Hankel_check()      
controller.create_Hankel()  

# Simulation    
x0 = np.array([[-20],[5],[1]])
t_step = 0.1
t_sim = 1.0                                    # simulation time
n_sim = ceil(t_sim/t_step)                     # Number of simulation step
x = x0 
xData_ini = np.empty((n,n_sim+1))
xData_ini[:,[0]] = x0; 
uData_ini = np.empty((m,n_sim))
yData_ini = np.empty((n,n_sim+1))
yData_ini[:,[0]] = x0; 
    
# Generate random input sequence
np.random.seed(5)
u_seq = np.array(np.random.rand(m,n_sim))

# Generate initial state
for t in range(n_sim):
    u = u_seq[:, t].reshape(m, 1)
    w = np.random.normal(mu, sigma, size=(n, 1))
    x = A@x+B@u
    y = C@x+w
    uData_ini[:,[t]] = u 
    xData_ini[:,[t+1]] = x
    yData_ini[:,[t+1]] = y

# Simulation    
x0 = xData_ini[:,-1]
y0 = yData_ini[:,-1]
u_ini = np.reshape(uData_ini[:,-n:],(-1, 1), order='F') 
y_ini = np.reshape(yData_ini[:,-n:],(-1, 1), order='F') 
t_step = 0.1
t_sim = 10.0;                                  # simulation time
n_sim = ceil(t_sim/t_step)                     # Number of simulation step
x = x0.reshape(m, 1)
xLog = np.zeros((n,n_sim+1))
xLog[:,0] = x0
uLog = np.empty((m,n_sim))
costLog = np.empty((1,n_sim))
yLog = np.zeros((n,n_sim+1))
yLog[:,0] = y0
    
# # run simulation
for t in range(n_sim):
    u = controller.computeInput(u_ini,y_ini).reshape(m, 1)
    # u = np.zeros((m,1))
    x = A@x+B@u    
    w = np.random.normal(mu, sigma, size=(n, 1))
    y = C@x+w
    u_vec = np.reshape(u,(-1, 1))
    x_vec = np.reshape(x,(-1, 1))
    y_vec = np.reshape(y,(-1, 1))
    cost = x_vec.T @ params['Q'] @ x_vec + u_vec.T @ params['R'] @ u_vec
    uLog[:,[t]] = u_vec
    xLog[:,[t+1]] = x_vec
    yLog[:,[t+1]] = y_vec
    costLog[0,[t]] = cost
    u_ini = np.block([[u_ini[n:]],[u_vec]])
    y_ini = np.block([[y_ini[n:]],[y_vec]])

# Plot system evolution    
print(np.sum(costLog))
y_plot = plt.figure(1)
time = np.linspace(0,t_sim,n_sim+1)

plt.subplot(311)
plt.plot(time,xLog[0,:])
plt.ylabel('y1')
plt.title('State')

plt.subplot(312)
plt.plot(time,xLog[1,:])   
plt.ylabel('y2')

plt.subplot(313)
plt.plot(time,xLog[2,:])
plt.ylabel('y3')


# Plot input evolution    
u_plot = plt.figure(2)
time = np.linspace(0,t_sim,n_sim);
plt.subplot(311)
plt.step(time,uLog[0,:])
plt.ylabel('u1')
plt.title('Input')

plt.subplot(312)
plt.step(time,uLog[1,:])   
plt.ylabel('u2')

plt.subplot(313)
plt.step(time,uLog[2,:])
plt.ylabel('u3')

plt.show()

