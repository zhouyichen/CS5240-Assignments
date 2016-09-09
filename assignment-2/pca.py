import numpy as np
from numpy import linalg as la

# part 1
file = open("data.txt")
data = np.genfromtxt(file, delimiter=",").T
file.close()
m, n = data.shape
print "m =", m
print "n =", n

#part 2
mean = np.mean(data, axis=1, keepdims=True)
sdata = data - mean

# print "mean of sdata =", np.mean(sdata, axis=1, keepdims=True) 

cov = np.cov(sdata, bias=True)
w, v = la.eig(cov)
sorted_indices = np.argsort(w)

print "eigenvalues =", w
print "sorted_indices =", sorted_indices

# part 3
total_variance = np.sum(w)
print "total_variance =", total_variance

R_l = np.zeros(m)

cumulated_variance = 0
for i in range(m):
	cumulated_variance += w[sorted_indices[m - 1 - i]]
	rl = cumulated_variance / total_variance
	print "l =", i + 1, "and R(l) =", rl
	R_l[i] = rl

Q = np.zeros([m, m])
for i in range(m):
	Q[:, i] = v[:, sorted_indices[m - 1 - i]]

x = np.mat([0.21, 0.72, 0.06, 0.36, -0.12, 0.04, 0.00, 0.46, 0.27, 0.59, 0.70]).T
y = Q.T * (x - mean)
print "x =\n", x
print "y =\n", y

for i in range(m):
	l = i + 1
	Q_hat = Q[:, :l]
	y_hat = Q_hat.T * (x - mean)
	# print "l =", l, " and y_hat =\n", y_hat

for i in range(m):
	l = i + 1
	Q_hat = Q[:, :l]
	y_hats = np.dot(Q_hat.T, sdata)
	y_estimate = np.zeros([m, n])
	y_estimate[:i+1, :] = y_hats
	x_hats = np.dot(Q, y_estimate) + mean
	diff = x_hats - data
	square_error = np.power(diff, 2)
	sum_squared_error = np.sum(square_error)
	print "l =", l, " and sum-squared error E =", sum_squared_error
	
