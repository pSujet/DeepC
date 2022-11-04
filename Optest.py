from math import ceil
import numpy as np 
import cvxpy as cp 
import random
import matplotlib.pyplot as plt
from numpy.linalg import matrix_rank

# Set-up DeePC and solve the optimization problem
# Define optimizer
n = 4
m = 1000
I = np.eye(n,m)
print(matrix_rank(I))
# g = cp.Variable((3,1))
a = cp.Variable((1,1))
b = cp.Variable((1,1))
c = cp.Variable((1,1))
x = cp.Variable((n,1))
# g = cp.bmat([a,b,c])
g = cp.Variable((m,1))

# Define Optimization problem
objective = cp.norm(x,2)**2
constraints = [I @ g == x]
problem = cp.Problem(cp.Minimize(objective), constraints)        
problem.solve()
# problem.solve(verbose=True)
# print(g.value)
# print(x.value)


