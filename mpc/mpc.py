import numpy as np; 
import cvxpy as cp; 
import control as ctrl

def computePredictedOptimalControls(mpcProblemData, x0):
    # Extract system matrices
    A = mpcProblemData['A']; 
    B = mpcProblemData['B']; 
    C = mpcProblemData['C']; 
    (n,m) = B.shape    
    # Planning horizon
    T = mpcProblemData['T']
    # objective function definition
    Q = mpcProblemData['Q']; 
    R = mpcProblemData['R']; 
    QT = mpcProblemData['QT']    
    # Constraints
    ulim = mpcProblemData['ulim']; 
    ylim = mpcProblemData['ylim']
    # Terminal constraint: MT*xT <= mT
    MT = mpcProblemData['MT']; 
    mT = mpcProblemData['mT']

    # Set-up and solve planning problem
    x = cp.Variable((n,T+1)); 
    u = cp.Variable((m,T))
    objective = 0; 
    constraints = [ x[:,[0]]== x0 ]
    for t in range(T):
        objective += cp.quad_form(x[:,t],Q)+ cp.quad_form(u[:,t],R)
        constraints += [ x[:, t+1]==A@x[:,t] + B@ u[:, t] ]
        constraints += [-ulim <= u[:,[t]], u[:,[t]] <= ulim]
        constraints += [-ylim <= C@x[:,[t]], C@x[:,[t]] <= ylim]
    constraints += [ MT@x[:,[T]] <= mT ]
    objective += cp.quad_form(x[:,T], QT)
    problem=cp.Problem(cp.Minimize(objective), constraints)
    
    problem.solve()
    return (x.value, u.value)

def mpcController(mpcProblemData, x):
    # Plan optimal controls and states over the next T samples
    (xPred, uPred) = computePredictedOptimalControls(mpcProblemData, x)

    # Apply the first control action in the predicted optimal sequence
    return uPred[:,[0]]