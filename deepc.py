# Author: Sujet Phodapol

import numpy as np
import cvxpy as cp
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
        self.lambda_slack = params['lambda_slack']        # Slack weight
        self.lambda_g = params['lambda_g'] 
              
        # Constraints
        if constr_u:
            self.ulim = params['ulim']
        else:
            self.ulim = 100000
        if constr_y:
            self.ylim = params['ylim']   
        else:
            self.ylim = 100000 
    
    def create_Hankel_check(self):
        L = self.T_ini + self.N + self.nB
        Hankel_Ud = np.empty(((L)*self.m,self.T-L+1))
        for i in range(L):
            Hankel_Ud[self.m*i:self.m*(i+1),:] = self.uData[:,i:self.T-L+i+1]
        print(matrix_rank(Hankel_Ud))
        print(Hankel_Ud.shape)
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

        self.Hankel_Y_p = Hankel_Y[:self.p*self.T_ini,:]
        self.Hankel_Y_f = Hankel_Y[self.p*self.T_ini:,:]  
        
        self.Hankel_P = np.block([[self.Hankel_U_p],[self.Hankel_Y_p]])  
        self.Hankel_F = np.block([[self.Hankel_U_f],[self.Hankel_Y_f]])
        self.Hankel_PF = np.block([[self.Hankel_U_p],[self.Hankel_Y_p],[self.Hankel_U_f],[self.Hankel_Y_f]])
        
        # print(self.Hankel_PF)
        print(self.Hankel_PF.shape)
        print(matrix_rank(self.Hankel_PF))          
            
    def computeDeePC(self,u_ini,y_ini):
        # Set-up DeePC and solve the optimization problem
        # Define optimizer
        g = cp.Variable((self.T - self.T_ini - self.N + 1,1))
        u = cp.Variable((self.N*self.m,1))
        y = cp.Variable((self.N*self.p,1))
        uy = cp.vstack([u,y])
        slack = cp.Variable((self.T_ini*self.p,1))
        ulim = 1

        # Define DeePC problem
        objective = 0
        constraints = []
        for i in range(self.N):
            objective += cp.quad_form(y[self.p*i:self.p*(i+1),0],self.Q)+ cp.quad_form(u[self.m*i:self.m*(i+1),0],self.R) 
            constraints += [-ulim <= u[self.m*i:self.m*(i+1),0], u[self.m*i:self.m*(i+1),0] <= ulim]
        objective += self.lambda_g * cp.norm(g,2)**2  # regularization 
        objective += self.lambda_slack * cp.norm(slack,2)**2
        constraints += [self.Hankel_U_p @ g == u_ini]           
        constraints += [self.Hankel_Y_p @ g == y_ini + slack] 
        constraints += [self.Hankel_F @ g == uy]
        problem = cp.Problem(cp.Minimize(objective), constraints)  
        problem.solve()      
        return (g.value, u.value, y.value)   
    
    def computeDeePC_GROUSE(self,u_ini,y_ini,U_predictor):
        # Set-up DeePC and solve the optimization problem
        # Define optimizer
        (row,col) = U_predictor.shape
        g = cp.Variable((col,1))
        u = cp.Variable((self.N*self.m,1))
        y = cp.Variable((self.N*self.p,1))
        uy = cp.vstack([u,y])
        slack = cp.Variable((self.T_ini*self.p,1))
        ulim = self.ulim
        
        # create Hankel_PF
        # U1 * g = [uini;yini;u;y]
        yini_start = self.m*self.T_ini
        uf_start = yini_start + self.p * self.T_ini
        yf_start = uf_start + self.m * self.N
        Hankel_U_p = U_predictor[:yini_start,:] 
        Hankel_Y_p = U_predictor[yini_start:uf_start,:]

        Hankel_U_f = U_predictor[uf_start:yf_start,:]
        Hankel_Y_f = U_predictor[yf_start:,:]  
        
        Hankel_F = np.block([[Hankel_U_f],[Hankel_Y_f]])

        # Define DeePC problem
        objective = 0
        constraints = []
        for i in range(self.N):
            objective += cp.quad_form(y[self.p*i:self.p*(i+1),0],self.Q)+ cp.quad_form(u[self.m*i:self.m*(i+1),0],self.R) 
            # objective += cp.quad_form(y[self.p*i:self.p*(i+1),0],self.Q)+ cp.quad_form(u[self.m*i:self.m*(i+1),0],self.R)
            constraints += [-ulim <= u[self.m*i:self.m*(i+1),0], u[self.m*i:self.m*(i+1),0] <= ulim]
        objective += self.lambda_g * cp.norm(g,2)**2  # regularization 
        objective += self.lambda_slack * cp.norm(slack,2)**2
        constraints += [Hankel_U_p @ g == u_ini]           
        constraints += [Hankel_Y_p @ g == y_ini + slack] 
        constraints += [Hankel_F @ g == uy]
        problem = cp.Problem(cp.Minimize(objective), constraints)  
        problem.solve()      
        return (g.value, u.value, y.value)  
       
    def computeInput(self, u_ini, y_ini):
        # Plan optimal controls and states over the next T samples
        (g, uPred, yPred) = self.computeDeePC(u_ini,y_ini)

        # Apply the first control action in the predicted optimal sequence
        return uPred[:self.m,0]
    
    def computeInputGROUSE(self, u_ini, y_ini,U_predictor):
        # Plan optimal controls and states over the next T samples
        (g, uPred, yPred) = self.computeDeePC_GROUSE(u_ini,y_ini,U_predictor)
        # (g, uPred, yPred) = self.computeDeePC(u_ini,y_ini)

        # Apply the first control action in the predicted optimal sequence
        return uPred[:self.m,0]
    
    def create_U1(self, low_dimension_rank):
        (u,s,v) = np.linalg.svd(self.Hankel_PF, full_matrices=True)
        U1 = u[:,:low_dimension_rank]
        return U1
    
    def grouse_update(self,U_Omega,v_obs,step_size):
        # Estimate weights: U_Omega = Delta_Omega*Ut
        w = np.linalg.pinv(U_Omega)@v_obs
        norm_w = np.linalg.norm(w)
        
        # Predict full vector:
        p = U_Omega@w
        norm_p = np.linalg.norm(p)
        
        # Compute residual:
        r = v_obs - U_Omega@w;  
        norm_r = np.linalg.norm(r)
        
        # Update subpace:
        sigma = norm_r*norm_w;
        q = step_size*sigma;
        U_update = U_Omega + ((np.cos(q)-1)*p/norm_p + np.sin(q)*r/norm_r)@np.transpose(w)/norm_w
        
        return U_update
        
            
    def computeDeePCIPen(self,u_ini,y_ini):
        # Set-up DeePC and solve the optimization problem
        # Define optimizer
        g = cp.Variable((self.T - self.T_ini - self.N + 1,1))
        u = cp.Variable((self.N*self.m,1))
        y = cp.Variable((self.N*self.p,1))
        uy = cp.vstack([u,y])
        slack = cp.Variable((self.T_ini*self.p,1))
        ulim = 1
        ylim = np.pi

        # Define DeePC problem
        objective = 0
        constraints = []
        for i in range(self.N):
            objective += cp.quad_form(y[self.p*i:self.p*(i+1),0],self.Q)# + cp.quad_form(u[self.m*i:self.m*(i+1),0],self.R) 
            constraints += [-ulim <= u[self.m*i:self.m*(i+1),0], u[self.m*i:self.m*(i+1),0] <= ulim]
            constraints += [-ylim <= y[self.p*i:self.p*(i+1),0], y[self.p*i:self.p*(i+1),0] <= ylim]
        objective += self.lambda_g * cp.norm(g,2)**2  # regularization 
        objective += self.lambda_slack * cp.norm(slack,2)**2
        constraints += [self.Hankel_U_p @ g == u_ini]           
        constraints += [self.Hankel_Y_p @ g == y_ini + slack] 
        constraints += [self.Hankel_F @ g == uy]
        problem = cp.Problem(cp.Minimize(objective), constraints)  
        problem.solve()      
        return (g.value, u.value, y.value)         
