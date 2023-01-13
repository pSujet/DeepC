from math import ceil
import numpy as np 
import cvxpy as cp 
import random
import matplotlib.pyplot as plt
from deepc import DeePC
from numpy import linalg
import pandas as pd

n = 30;
d = 10;
U_true = np.array([[-0.5613,0.7346],[-0.8269,-0.5173],[-0.0352,0.4391]])
U_ini = np.array([[-0.5062,0.7332],[-0.4828,-0.6772],[-0.7146,-0.0618]])
    
print(U_true)
   
      
       