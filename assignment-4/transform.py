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
def transformPoint(p, s, R, T):
	return s * R * p + T

def calculate_mean_squared_error(source, target, s, R, T):
	total = 0
	n_points = source.shape[1]
	for i in range(n_points):
		u = np.mat(source[:, i]).T
		vp = transformPoint(u, s, R, T)
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

# Part 3 Step 3
from scipy import misc
source_image = misc.imread('source.jpg')

def inverseTransformPoint(q, s, R, T):
	return np.linalg.inv(R) * (q - T) / s

# Part 3 Step 4
def truncateTransform(source, s, R, T):
	h, w, _ = source.shape

	# calculate the size of the result image
	c1 = transformPoint(np.mat([[0], [0]]), s, R, T)
	c2 = transformPoint(np.mat([[w-1], [0]]), s, R, T)
	c3 = transformPoint(np.mat([[w-1], [h-1]]), s, R, T)
	c4 = transformPoint(np.mat([[0], [h-1]]), s, R, T)
	max_0 = max(c1[0, 0], c2[0, 0], c3[0, 0], c4[0, 0])
	max_1 = max(c1[1, 0], c2[1, 0], c3[1, 0], c4[1, 0])
	min_0 = min(c1[0, 0], c2[0, 0], c3[0, 0], c4[0, 0])
	min_1 = min(c1[1, 0], c2[1, 0], c3[1, 0], c4[1, 0])
	
	result_w = int(round(max_0 - min_0)) + 1
	result_h = int(round(max_1 - min_1)) + 1

	# extra translation to make sure the whole transformed image is within the picture
	extra_T = np.mat([[-min_0], [-min_1]])

	result = np.zeros((result_h, result_w, 3))
	for index in np.ndindex(result_w, result_h):
		# Compute the corresponding coordinates in the source image
		p = inverseTransformPoint(np.mat([result_w - 1 - index[0], result_h - 1 - index[1]]).T - extra_T, s, R, T)
		# Truncate real-number coordinates to integer number
		u1 = w - 1 - int(p[0, 0])
		u2 = h - 1 - int(p[1, 0])
		# if u not in the source image, just ignore
		if u1 >= w or u2 >= h or u1 < 0 or u2 < 0:
			continue
		result[index[1], index[0]] = source[u2, u1]
	return result

# Part 3 Step 5
# result_t = truncateTransform(source_image, s1, R1, T1)
# misc.imsave('result-t.jpg', result_t)

# Part 3 Step 6
def roundTransform(source, s, R, T):
	h, w, _ = source.shape

	# calculate the size of the result image
	c1 = transformPoint(np.mat([[0], [0]]), s, R, T)
	c2 = transformPoint(np.mat([[w-1], [0]]), s, R, T)
	c3 = transformPoint(np.mat([[w-1], [h-1]]), s, R, T)
	c4 = transformPoint(np.mat([[0], [h-1]]), s, R, T)
	max_0 = max(c1[0, 0], c2[0, 0], c3[0, 0], c4[0, 0])
	max_1 = max(c1[1, 0], c2[1, 0], c3[1, 0], c4[1, 0])
	min_0 = min(c1[0, 0], c2[0, 0], c3[0, 0], c4[0, 0])
	min_1 = min(c1[1, 0], c2[1, 0], c3[1, 0], c4[1, 0])
	
	result_w = int(round(max_0 - min_0)) + 1
	result_h = int(round(max_1 - min_1)) + 1

	# extra translation to make sure the whole transformed image is within the picture
	extra_T = np.mat([[-min_0], [-min_1]])

	result = np.zeros((result_h, result_w, 3))
	for index in np.ndindex(result_w, result_h):
		# Compute the corresponding coordinates in the source image
		p = inverseTransformPoint(np.mat([result_w - 1 - index[0], result_h - 1 - index[1]]).T - extra_T, s, R, T)
		# round real-number coordinates to integer number
		u1 = w - 1 - int(round(p[0, 0]))
		u2 = h - 1 - int(round(p[1, 0]))
		# if u not in the source image, just ignore
		if u1 >= w or u2 >= h or u1 < 0 or u2 < 0:
			continue
		result[index[1], index[0]] = source[u2, u1]
	return result

# Part 3 Step 7
# result_r = roundTransform(source_image, s1, R1, T1)
# misc.imsave('result-r.jpg', result_r)

# Part 4 Step 2
def bilinearTransform(source, s, R, T):
	h, w, _ = source.shape

	# calculate the size of the result image
	c1 = transformPoint(np.mat([[0], [0]]), s, R, T)
	c2 = transformPoint(np.mat([[w-1], [0]]), s, R, T)
	c3 = transformPoint(np.mat([[w-1], [h-1]]), s, R, T)
	c4 = transformPoint(np.mat([[0], [h-1]]), s, R, T)
	max_0 = max(c1[0, 0], c2[0, 0], c3[0, 0], c4[0, 0])
	max_1 = max(c1[1, 0], c2[1, 0], c3[1, 0], c4[1, 0])
	min_0 = min(c1[0, 0], c2[0, 0], c3[0, 0], c4[0, 0])
	min_1 = min(c1[1, 0], c2[1, 0], c3[1, 0], c4[1, 0])
	
	result_w = int(round(max_0 - min_0)) + 1
	result_h = int(round(max_1 - min_1)) + 1

	# extra translation to make sure the whole transformed image is within the picture
	extra_T = np.mat([[-min_0], [-min_1]])

	result = np.zeros((result_h, result_w, 3))
	for index in np.ndindex(result_w, result_h):
		# Compute the corresponding coordinates in the source image
		p = inverseTransformPoint(np.mat([result_w - 1 - index[0], result_h - 1 - index[1]]).T - extra_T, s, R, T)
		# round real-number coordinates to integer number
		x1 = w - 1 - int(p[0, 0])
		y1 = h - 1 - int(p[1, 0])
		x2 = w - 1 - int(p[0, 0] + 1)
		y2 = h - 1 - int(p[1, 0] + 1)

		# if u not in the source image, just ignore
		if x1 >= w or y1 >= h or x2 < 0 or y2 < 0:
			continue

		# bilinear interpolation
		x = w - 1 - p[0, 0]
		y = h - 1 - p[1, 0]
		xt = x - x2
		yt = y - y2
		c1 = xt * source[y1, x1] + (1 - xt) * source[y1, x2]
		c2 = xt * source[y2, x1] + (1 - xt) * source[y2, x2]
		c3 = yt * c1 + (1 - yt) * c2

		result[index[1], index[0]] = c3
	return result

# Part 4 Step 3
# result_b = bilinearTransform(source_image, s1, R1, T1)
# misc.imsave('result-b.jpg', result_b)


