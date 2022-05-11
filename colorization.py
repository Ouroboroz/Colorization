import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from pypardiso import spsolve
from scipy.sparse import linalg

#File path for images
monochromatic_image_path = "craig_mono.bmp"
partial_image_path = None
scribbled_image_path = "craig_scribble.bmp"
#for saving the generated image
save = True
save_path = "craig_colorized_iterative.bmp"
#for displaying in implot6
display = True

weights_type = 0

window_size = 1

iterative = True

verbose = True
"""
###
#X#
###
for window size 1
"""


def colorize_partial(mnochromatic_image, partial_image, scribbled_image, fill_count = 1000):
	partial_yuv = cv2.cvtColor(partial_image, cv2.COLOR_BGR2YUV)
	
	pixel_count = 0
	h,w, _ = partial_image.shape
	while pixel_count < fill_count:
		x = np.random.randint(h)
		y = np.random.randint(w)
		if partial_yuv[x,y,1] != partial_yuv[x,y,2]:
			scribbled_image[x,y,:] = partial_image[x,y,:]
			pixel_count += 1
	return colorize(monochromatic_image, scribbled_image)

def colorize(monochromatic_image, scribbled_image):
	scribbled = np.sum(abs(scribbled_image - monochromatic_image),2) > 0.01
	h,w,_ = monochromatic_image.shape
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
				if weights_type == 0:
					window[:window_index] = np.exp(-(window[:window_index] - mono_y[y,x])**2/(2*variance))
				elif weights_type == 1:
					window[:window_index] = 1 + 1/variance*(window[:window_index] - mean)*(mono_y[y,x] - mean)
				elif weights_type == 2:
					zeros = np.zeros(window.shape).astype(int)
					zeros[:window_index] = (window[:window_index] - mono_y[y,x])**2 > 0.001
					zeros = zeros.astype(bool)
					#print(zeros)
					#print(min((window[:window_index] - mono_y[y,x])**2))
					window[:window_index] = np.exp(-(window[:window_index] - mono_y[y,x])**2/(2*variance))
					window[zeros] = 0
				if np.sum(window[:window_index]) != 0:
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
	chromo_u = spsolve(A,b).reshape(h,w,order='F')

	b = np.zeros(h*w)
	scribble_v_flatten = scribble_v.flatten('F')
	indices_flatten = np.nonzero(scribbled.flatten('F'))

	b[indices_flatten] = scribble_v_flatten[indices_flatten]
	chromo_v = spsolve(A,b).reshape(h,w,order='F')

	chromo_yuv =  np.float32(np.stack((mono_y, chromo_u, chromo_v),2))
	return chromo_yuv


if __name__ == "__main__":

	partial_image = cv2.imread(partial_image_path) if partial_image_path else None 

	#Convert from filepath to image in np.ndarray
	monochromatic_image = cv2.imread(monochromatic_image_path)
	#monochromatic_image = cv2.normalize(monochromatic_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
	h, w, _ = monochromatic_image.shape
	scribbled_image = cv2.imread(scribbled_image_path)
	#scribbled_image = cv2.normalize(scribbled_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
	scribble_h, scribble_w, _ = scribbled_image.shape

	assert h == scribble_w and w == scribble_w, "Monochromatic image and scribbled image must have same dims"
	
	if not iterative:
		if partial_image_path is None:
			chromo_yuv = colorize(monochromatic_image, scribbled_image)
		else:
			chromo_yuv = color_partial(monochromatic_image, partial_image, scribbled_image)

		if display:
			chromo_image = cv2.cvtColor(chromo_yuv, cv2.COLOR_YUV2RGB)
			#chromo_image = cv2.normalize(chromo_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
			plt.imshow(chromo_image)
			plt.show()

		if save:
			chromo_image = cv2.cvtColor(chromo_yuv, cv2.COLOR_YUV2BGR)*255
			cv2.imwrite(save_path, chromo_image)
	else:
		h_bounds = (0,h//4, h//4*2, h//4*3,h)
		w_bounds = (0,w//4, w//4*2, w//4*3,w)
		for i in range(4):
			h_bound_lower = 0
			h_bound_upper = h_bounds[i+1]
			w_bound_lower = 0
			w_bound_upper = w_bounds[i+1]
			mono_bound = monochromatic_image[h_bound_lower:h_bound_upper, w_bound_lower:w_bound_upper,:]
			scribbled_bound = scribbled_image[h_bound_lower:h_bound_upper, w_bound_lower:w_bound_upper,:]
			chromo_yuv = colorize(mono_bound, scribbled_bound)
			chromo_image = cv2.cvtColor(chromo_yuv, cv2.COLOR_YUV2BGR)*255
			if verbose:
				im = cv2.cvtColor(scribbled_image, cv2.COLOR_BGR2RGB)
				plt.imshow(im)
				plt.show()
			scribbled_image[:h_bound_upper, :w_bound_upper,:] = chromo_image[:h_bound_upper, :w_bound_upper,:]
		cv2.imwrite(save_path, scribbled_image)
		chromo_image = cv2.cvtColor(scribbled_image, cv2.COLOR_BGR2RGB)
		plt.imshow(chromo_image)
		plt.show()

