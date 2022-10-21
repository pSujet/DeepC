from math import ceil
import numpy as np 
import cvxpy as cp 
import random
import matplotlib.pyplot as plt
from deepc import DeePC
from numpy import linalg

if __name__ == '__main__':
    # ========= Collect Data START =========
    # Define system
    A = np.array([[1.01,0.01,0],[0.01,1.01,0.01],[0,0.01,1.01]])
    B = np.eye(3)
    C = np.eye(3)
    (n,m) = B.shape                                 # n = number of states, m = number of inputs
    p = n                                           # p = number of output
          
    # Simulation    
    x0 = np.array([[0],[0],[1]])
    t_step = 0.1
    t_sim = 4.0;                                   # simulation time 
    n_sim = ceil(t_sim/t_step)                     # Number of simulation step n_sim >= (m+1)*(T_ini + N + n(B) - 1)
    x = x0; 
    xData = np.empty((n,n_sim+1)); 
    xData[:,[0]] = x0; 
    uData = np.empty((m,n_sim));
       
    # Generate random input sequence
    np.random.seed(1)
    u_seq = np.array(np.random.rand(m,n_sim))

    # Collect data
    for t in range(n_sim):
        u = u_seq[:, t].reshape(m, 1)
        x = A@x+B@u
        uData[:,[t]] = u 
        xData[:,[t+1]] = x
    # ========= Collect Data END =========   
      
    
    # ============= DeePC =============
    # set parameters
    params = {}
    params['uData'] = uData
    params['yData'] = xData[:,1:]
    params['N'] = 5
    params['Q'] = np.eye(3)
    params['R'] = np.eye(3)*10
    
    # create controller    
    controller = DeePC(params)
    # controller.create_Hankel_check()
    
    # offline computation
    controller.create_Hankel()       
    # ============= DeePC =============
    
    # ========= Simulate new initial state START =========
    # Simulation    
    x0 = np.array([[-20],[5],[1]])
    t_step = 0.1
    t_sim = 1.0;                                   # simulation time
    n_sim = ceil(t_sim/t_step)                     # Number of simulation step
    x = x0; 
    xData_ini = np.empty((n,n_sim+1)); 
    xData_ini[:,[0]] = x0; 
    uData_ini = np.empty((m,n_sim));
       
    # Generate random input sequence
    np.random.seed(5)
    u_seq = np.array(np.random.rand(m,n_sim))

    # Collect data
    for t in range(n_sim):
        u = u_seq[:, t].reshape(m, 1)
        x = A@x+B@u
        uData_ini[:,[t]] = u 
        xData_ini[:,[t+1]] = x
    # ========= Simulate new initial state END =========
    
       
    # ========= Simulate controller START =========       
    # Simulation    
    x0 = xData_ini[:,-1]
    u_ini = np.reshape(uData_ini[:,-n:],(-1, 1), order='F') 
    y_ini = np.reshape(xData_ini[:,-n:],(-1, 1), order='F') 
    t_step = 0.1
    t_sim = 5.0;                                  # simulation time
    n_sim = ceil(t_sim/t_step)                     # Number of simulation step
    x = x0; 
    xLog = np.zeros((n,n_sim+1)); 
    xLog[:,0] = x0; 
    uLog = np.empty((m,n_sim));
       
    # Collect data
    for t in range(n_sim):
        u = controller.computeInput(u_ini,y_ini)
        # print("before: " + str(x))
        x = A@x+B@u
        # print(B@u)
        # print("after: " + str(x))
        
        u_vec = np.reshape(u,(-1, 1))
        x_vec = np.reshape(x,(-1, 1))
        uLog[:,[t]] = u_vec
        xLog[:,[t+1]] = x_vec
        y = x_vec
        u_ini = np.block([[u_ini[n:]],[u_vec]])
        y_ini = np.block([[y_ini[n:]],[y]])
    # ========= Simulate controller END =========

    # Plot system evolution    
    y_plot = plt.figure(1)
    time = np.linspace(0,t_sim,n_sim+1);
    plt.subplot(311)
    plt.step(time,xLog[0,:])
    plt.ylabel('y1')
    
    plt.subplot(312)
    plt.step(time,xLog[1,:])   
    plt.ylabel('y2')
    
    plt.subplot(313)
    plt.step(time,xLog[2,:])
    plt.ylabel('y3')
    
    
    # Plot system evolution    
    u_plot = plt.figure(2)
    time = np.linspace(0,t_sim,n_sim);
    plt.subplot(311)
    plt.step(time,uLog[0,:])
    plt.ylabel('u1')
    
    plt.subplot(312)
    plt.step(time,uLog[1,:])   
    plt.ylabel('u2')
    
    plt.subplot(313)
    plt.step(time,uLog[2,:])
    plt.ylabel('u3')
    
    plt.show()
    
    