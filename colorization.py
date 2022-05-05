import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import linalg

#File path for images
monochromatic_image_path = "example2.bmp"
scribbled_image_path = "example2_marked.bmp"
window_size = 1
"""
###
#X#
###
for window size 1
"""

#Convert from filepath to image in np.ndarray
monochromatic_image = cv2.imread(monochromatic_image_path)
#monochromatic_image = cv2.normalize(monochromatic_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
h, w, _ = monochromatic_image.shape
scribbled_image = cv2.imread(scribbled_image_path)
#scribbled_image = cv2.normalize(scribbled_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
scribble_h, scribble_w, _ = scribbled_image.shape

assert h == scribble_w and w == scribble_w, "Monochromatic image and scribbled image must have same dims"


scribbled = np.sum(abs(scribbled_image - monochromatic_image),2) > 0.0001

#Convert BGR image to YUV colorspace
monochromatic_yuv = cv2.cvtColor(monochromatic_image, cv2.COLOR_BGR2YUV)/255
scribble_yuv = cv2.cvtColor(scribbled_image, cv2.COLOR_BGR2YUV)/255

#get the luminance layer of the monochromatic image
mono_y, _, _ = cv2.split(monochromatic_yuv)

#get the scribble u,v
_, scribble_u, scribble_v = cv2.split(scribble_yuv)

pixel_index = 0
constraint_index = 0
constraint_row = np.zeros((2*window_size+1)**2 * h * w)
constraint_column = np.zeros((2*window_size+1)**2 * h * w)
indices = np.arange(h*w).reshape((h,w),order='F')
#numpy.core._exceptions.MemoryError: Unable to allocate 53.6 GiB for an array with shape (84800, 265, 320) and data type float64
constraints = np.zeros((2*window_size+1)**2 * h * w)

for x in range(w):
	for y in range(h):
		#if not scribbled
		if not scribbled[y,x]:
			window = np.zeros((2*window_size+1)**2)
			window_index = 0

			#apply get y values for window
			for window_y in range(2*window_size+1):	
				for window_x in range(2*window_size+1):
					#accounts for bounds of image
					i = min(max(y+window_y - window_size,0), h-1)
					j = min(max(x+window_x - window_size,0), w-1)
					# we only count neighbors
					if i == y and j == x:
						continue
					constraint_row[constraint_index] = pixel_index
					constraint_column[constraint_index] = indices[i,j]
					window[window_index] = mono_y[i,j]
					window_index += 1
					constraint_index += 1

			window[window_index] = mono_y[y,x]
			mean = np.mean(window[:window_index+1])
			variance = np.mean((window[:window_index+1] - mean)**2)
			if variance < 0.001:
				variance = 0.001
			window[:window_index] = np.exp(-(window[:window_index] - mono_y[y,x])**2/(2*variance))
			window[:window_index] = window[:window_index]/np.sum(window[:window_index])
			constraints[constraint_index-window_index:constraint_index] = -window[:window_index]
			
			# for window_y in range(2*window_size+1):	
			# 	for window_x in range(2*window_size+1):
			# 		#accounts for bounds of image
			# 		i = min(max(window_y - window_size,0), h)
			# 		j = min(max(window_x - window_size,0), w)
			# 		if i == x and j == y:
			# 			continue
			# 		constraints[pixel_index,i,j] = window[window_index]
			# 		window_index += 1

		constraints[constraint_index] = 1
		constraint_row[constraint_index] = pixel_index
		constraint_column[constraint_index] = indices[y,x]
		constraint_index += 1
		pixel_index += 1

constraints = constraints[:constraint_index]
constraint_row = constraint_row[:constraint_index]
constraint_column = constraint_column[:constraint_index]

A = sparse.csr_matrix((constraints,(constraint_row, constraint_column)), (h*w,h*w))

b = np.zeros(h*w)
scribble_u_flatten = scribble_u.flatten('F')
indices_flatten = np.nonzero(scribbled.flatten('F'))
b[indices_flatten] = scribble_u_flatten[indices_flatten]
chromo_u = linalg.spsolve(A,b).reshape(h,w,order='F')

b = np.zeros(h*w)
scribble_v_flatten = scribble_v.flatten('F')
indices_flatten = np.nonzero(scribbled.flatten('F'))

b[indices_flatten] = scribble_v_flatten[indices_flatten]
chromo_v = linalg.spsolve(A,b).reshape(h,w,order='F')

chromo_yuv =  np.float32(np.stack((mono_y, chromo_u, chromo_v),2))
print(chromo_yuv.shape)

print(monochromatic_yuv.shape)
print(type(monochromatic_yuv))
print(np.float32(mono_y))
print(np.min(chromo_u))

chromo_image = cv2.cvtColor(chromo_yuv, cv2.COLOR_YUV2RGB)
#chromo_image = cv2.normalize(chromo_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
plt.imshow(chromo_image)
plt.show()



