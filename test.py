import numpy as np 
from deepc import DeePC
from numpy import linalg

# A = np.array([[1.01,0.01,0],[0.01,1.01,0.01],[0,0.01,1.01]])
# params = {}
# params['A'] = np.array([[1.01,0.01,0],[0.01,1.01,0.01],[0,0.01,1.01]])
# # params['B'] = np.eye(3)
# # params['C'] = np.eye(3)
# controller = DeepC(params)
# controller.test(20)
# print(controller.test)
# # w, v = linalg.eig(np.eye(3))
# # print(w)
# # print(v)
A = np.reshape(np.array([[1],[2],[3]]),(-1, 1))
print(A)
print(np.block([[A],[A]]))