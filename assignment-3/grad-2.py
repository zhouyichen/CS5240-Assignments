import numpy as np

# Read Data
f = open("co2-data.txt")
data = np.genfromtxt(f, delimiter=",")
f.close()

n = data.shape[0]

x = np.array(data[:,1]) / 100000
y = np.array(data[:,2]) / 100
z = np.array(data[:,3]) / 10000
xx = x * x
yy = y * y
zz = z * z
D = [xx, yy, zz]
v = data[:,0] / 100

a = np.ones(3)
alpha = 5.739

def get_diff(D, a):
	total = np.zeros(v.shape)
	for i in range(3):
		sq = D[i]
		total += sq / (a[i] + sq)
	return total - v

def get_gradient(D, a, diff):
	'''
		diff = f(x, y, z, a) - v
	'''
	gradient = np.zeros(3)
	for i in range(3):
		sq = D[i]
		gradient[i] = 2 * np.sum(diff * (-sq / np.square(a[i] + sq)))
	return gradient

output_file = open('output2.txt', 'w')
output_file.write('Alpha: '+ str(alpha) + '\n')
output_file.write('Initial a: ' + str(a.T) + '\n')
for i in range(1, 201):
	output_file.write('Iteration number: ' + str(i) + '\n')
	diff = get_diff(D, a)
	sum_squared_error = np.sum(diff * diff)
	output_file.write('Sum squared error: ' + str(sum_squared_error) + '\n')
	gradient = get_gradient(D, a, diff)
	a -= alpha * gradient

output_file.write('Final a: ' + str(a.T) + '\n')
diff = get_diff(D, a)
sum_squared_error = np.sum(diff * diff)
print(sum_squared_error)
output_file.write('Final sum squared error: ' + str(sum_squared_error) + '\n')


# evaluation of alpha
# results = []
# for i in range(1000):
# 	start = 5.7
# 	a = np.ones(3)
# 	alpha = start + i / 10000.0

# 	for i in range(1, 201):
# 		diff = get_diff(D, a)
# 		sum_squared_error = np.sum(diff * diff)
# 		gradient = get_gradient(D, a, diff)
# 		a -= alpha * gradient
# 	diff = get_diff(D, a)
# 	sum_squared_error = np.sum(diff * diff)
# 	results.append([alpha, sum_squared_error])
# results.sort(key=lambda i: i[1])
# print results[:10]




