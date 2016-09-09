import numpy as np
from numpy import linalg as la

file = open("data.txt")
data = np.genfromtxt(file, delimiter=",").T
file.close()
m, n = data.shape
print "m =", m
print "n =", n

mean = np.mean(data, axis=1, keepdims=True)
sdata = data - mean

# print "mean of sdata =", np.mean(sdata, axis=1, keepdims=True) 

cov = np.cov(sdata, bias=True)
w, v = la.eig(cov)
sorted_indices = np.argsort(w)

print "eigenvalues =", w
print "sorted_indices =", sorted_indices