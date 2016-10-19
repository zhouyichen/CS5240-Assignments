import numpy as np

# Part 2 Step 1
def bestSimilarityTransform(source, target):
	'''
	rearrange source points into a 2*n by 4 matrix
		x1, y1,  1, 0
		y1, -x1, 0, 1
		...
	'''
	n_points = source.shape[1]
	source_mat = np.zeros((2 * n_points, 4))
	for i in range(n_points):
		p = source[:, i]
		source_mat[2*i] = np.array([p[0], p[1], 1, 0])
		source_mat[2*i+1] = np.array([p[1], -p[0], 0, 1])

	'''
	rearrange target points into a 2*n by 1 matrix
		x1
		y1
		...
	'''
	target = target.reshape([2 * n_points, 1], order='F')

	'''
	calculate the coefficients
	c = [s * cos,
		-sin,
		t1,
		t2]
	'''
	c = np.linalg.pinv(source_mat).dot(target)

	n_sin = c[1, 0]
	cos = np.sqrt(1 - n_sin * n_sin)
	s = c[0, 0] / cos

	R = np.mat([
		[cos, n_sin],
		[-n_sin, cos]
		])
	T = np.mat([[c[2, 0]], [c[3, 0]]])

	return s, R, T

# Part 2 Step 2
data = np.genfromtxt('coord.txt', delimiter=",")

source = data[:, 0:2].T
target = data[:, 2:4].T

# Part 2 Step 3, 4
def transformPoints(p, s, R, T):
	return s * R * p + T

def calculate_mean_squared_error(source, target, s, R, T):
	total = 0
	n_points = source.shape[1]
	for i in range(n_points):
		u = np.mat(source[:, i]).T
		vp = transformPoints(u, s, R, T)
		diff = vp - np.mat(target[:, i]).T
		total += diff.T * diff
	D = total / n_points
	return D[0, 0]

s1, R1, T1 = bestSimilarityTransform(source, target)
print 's1:', s1
print 'R1:'
print R1
print 'T1:'
print T1
print 'Mean squared error:', calculate_mean_squared_error(source, target, s1, R1, T1)

# Part 2 Step 5
s2, R2, T2 = bestSimilarityTransform(target, source)
print 's2:', s2
print 'R2:'
print R2
print 'T2:'
print T2
print 'Mean squared error:', calculate_mean_squared_error(target, source, s2, R2, T2)

# Part 2 Step 6
print 's1 * s2 =', s1 * s2
print 'R1 * R2 ='
print R1 * R2
print 'T1 + s1 * R1 * T2 ='
print T1 + s1 * R1 * T2
print 'T2 + s2 * R2 * T1 ='
print T2 + s2 * R2 * T1