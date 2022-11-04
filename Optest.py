from math import ceil
import numpy as np 
import cvxpy as cp 
import random
import matplotlib.pyplot as plt

# Set-up DeePC and solve the optimization problem
# Define optimizer
I = np.eye(3)
g = cp.Variable((3,1))
x = cp.Variable((3,1))

# Define Optimization problem
objective = cp.norm(x,2)**2
constraints = [I@g == x]
problem = cp.Problem(cp.Minimize(objective), constraints)        
problem.solve(verbose=True)
print(g.value)
print(x.value)