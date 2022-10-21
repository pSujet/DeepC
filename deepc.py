# Author: Sujet Phodapol

import numpy as np; 
import cvxpy as cp; 
import control as ctrl
from numpy.linalg import matrix_rank

class DeePC:
    def __init__(self, params, constr_u = False, constr_y = False):
        # System dynamic             
        self.uData = params['uData']                    # data of inputs
        self.yData = params['yData']                    # data of outputs
        self.m = self.uData.shape[0]                    # number of inputs 
        self.p = self.yData.shape[0]                    # number of outputs 
        self.n = self.p                                 # number of states 
       
        # Planning horizon
        self.N = params['N']                            # planning horizon
        self.T_ini = self.n                             # initial horizon
        self.nB = self.n                                # minimal state representation
        self.T_L = self.T_ini + self.N                  # rows of Hankel_PF
        self.T = self.uData.shape[1]                    # total number of time steps from data (columns of data)
        
        
        # Objective function weight
        self.Q = params['Q']
        self.R = params['R'] 
        # self.QT = params['P']                           # Terminal weight     
              
        # Constraints
        if constr_u:
            self.ulim = params['ulim']
        if constr_y:
            self.ylim = params['ylim']
    
    def test(self,u):
        self.test = u
    
    def create_Hankel_check(self):
        L = self.T_ini + self.N + self.nB
        Hankel_Ud = np.empty(((L)*self.m,self.T-L+1))
        for i in range(L):
            Hankel_Ud[self.m*i:self.m*(i+1),:] = self.uData[:,i:self.T-L+i+1]
        if matrix_rank(Hankel_Ud) == L*self.m:
            print("We are good to go!!")
        else:
            print("You are wrong")
    
    def create_Hankel(self):
        m = self.m
        p = self.p
        T = self.T
        T_L = self.T_L
        Hankel_U = np.empty((T_L*m,T-T_L+1))
        Hankel_Y = np.empty((T_L*p,T-T_L+1))
        for i in range(T_L):
            Hankel_U[m*i:m*(i+1),:] = self.uData[:,i:T-T_L+i+1]
            Hankel_Y[p*i:p*(i+1),:] = self.yData[:,i:T-T_L+i+1]
            
        # create Hankel_PF
        self.Hankel_U_p = Hankel_U[:self.m*self.T_ini,:] 
        self.Hankel_U_f = Hankel_U[self.m*self.T_ini:,:]

        self.Hankel_Y_p = Hankel_Y[:self.m*self.T_ini,:]
        self.Hankel_Y_f = Hankel_Y[self.m*self.T_ini:,:]  
        
        self.Hankel_P = np.block([[self.Hankel_U_p],[self.Hankel_Y_p]])  
        self.Hankel_PF = np.block([[self.Hankel_U_p],[self.Hankel_Y_p],[self.Hankel_U_f],[self.Hankel_Y_f]])
        
    # def computIni(self,u_ini,y_ini):
    #     # create ini_matrix
    #     ini_matrix = np.empty((self.n*self.m + self.n*self.p,1))
    #     ini_matrix[:self.n*self.m,:] = np.reshape(u_ini)                                # last n element
    #     ini_matrix[self.n*self.m:self.n*self.m+self.n*self.p,:] = np.reshape(y_ini)     # last n element

    
    def computeDeePC(self,u_ini,y_ini):
        # Set-up DeePC and solve the optimization problem
        # Define optimizer
        g = cp.Variable((self.T - self.T_ini - self.N + 1,1))
        u = cp.Variable((self.N*self.m,1))
        y = cp.Variable((self.N*self.p,1))
        ini_matrix = np.block([[u_ini],[y_ini]])

        # Define DeePC problem
        objective = 0
        constraints = []
        for i in range(self.N):
            objective += cp.quad_form(y[self.p*i:self.p*(i+1),0],self.Q)+ cp.quad_form(u[self.m*i:self.m*(i+1),0],self.R)    
        constraints += [self.Hankel_P @ g == ini_matrix]     
        constraints += [self.Hankel_U_f @ g == u]
        constraints += [self.Hankel_Y_f @ g == y]
        problem = cp.Problem(cp.Minimize(objective), constraints)        
        problem.solve()
        
        return (g.value, u.value, y.value)
            
    def computeInput(self, u_ini, y_ini):
        # Plan optimal controls and states over the next T samples
        (g, uPred, yPred) = self.computeDeePC(u_ini,y_ini)

        # Apply the first control action in the predicted optimal sequence
        return uPred[:self.m,0]
            
        
