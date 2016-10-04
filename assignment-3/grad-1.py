import numpy as np

# Read Data
f = open("co2-data.txt")
data = np.genfromtxt(f, delimiter=",")
f.close()

n = data.shape[0]

D = np.matrix(np.empty([n,10]))
x = np.array(data[:,1]) / 100000
y = np.array(data[:,2]) / 100
z = np.array(data[:,3]) / 10000
xx = x * x
yy = y * y
zz = z * z
xy = x * y
xz = x * z
yz = y * z
D[:,0:9] = np.matrix([xx, yy, zz, xy, yz, xz, x, y, z]).T
D[:,9] = np.ones([n,1])

v = np.matrix(data[:,0]).T / 100

a = np.zeros([10, 1])
alpha = 3.879 * 10 ** -2

output_file = open('output1.txt', 'w')
output_file.write('Alpha: '+ str(alpha) + '\n')
output_file.write('Initial a: ' + str(a.T) + '\n')
for i in range(1, 201):
	output_file.write('Iteration number: ' + str(i) + '\n')
	diff = D * a - v
	sum_squared_error = diff.T * diff
	output_file.write('Sum squared error: ' + str(sum_squared_error[0,0]) + '\n')
	# if (i == 200): print sum_squared_error[0,0]
	a -= 2 * alpha * D.T * diff

output_file.write('Final a: ' + str(a.T) + '\n')
diff = D * a - v
sum_squared_error = diff.T * diff
output_file.write('Final sum squared error: ' + str(sum_squared_error[0,0]) + '\n')


# opt_a = np.linalg.inv(D.T * D) * D.T * v
# print opt_a
# diff = D * opt_a - v
# sum_squared_error = diff.T * diff
# print sum_squared_error


