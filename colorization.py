import cv2
import numpy as np
import matplotlib.pyplot as plt

#File path for images
monochromatic_image_path = "example.bmp"
scribbled_image_path = "example_marked.bmp"
window_size = 1
"""
###
#X#
###
for window size 1
"""

#Convert from filepath to image in np.ndarray
monochromatic_image = cv2.imread(monochromatic_image_path)
h, w, _ = monochromatic_image.shape
scribbled_image = cv2.imread(scribbled_image_path)
scribble_h, scribble_w, _ = scribbled_image.shape

assert h == scribble_w and w == scribble_w, "Monochromatic image and scribbled image must have same dims"


scribbled = np.sum(abs(scribbled_image - monochromatic_image),2) > 0.0001

#Convert BGR image to YUV colorspace
monochromatic_yuv = cv2.cvtColor(monochromatic_image, cv2.COLOR_BGR2YUV)
scribble_yuv = cv2.cvtColor(scribbled_image, cv2.COLOR_BGR2YUV)

#get the luminance layer of the monochromatic image
mono_y, _, _ = cv2.split(monochromatic_yuv)

#get the scribble u,v
_, scribble_u, scribble_v = cv2.split(scribble_yuv)

pixel_index = 0
constraints = np.zeros((h*w, h, w))
for x in range(w):
	for y in range(h):
		#if not scribbled
		if scribbled[y,x]:
			window = np.zeros((2*window_size+1)**2)
			window_index = 0

			#apply get y values for window
			for window_y in range(2*window_size+1):	
				for window_x in range(2*window_size+1):
					#accounts for bounds of image
					i = min(max(window_y - window_size,0), h)
					j = min(max(window_x - window_size,0), w)
					# we only count neighbors
					if i == x and j == y:
						continue
					window[window_index] = mono_y[i,j]
					window_index += 1
			mean = np.mean(window[:window_index+1])
			variance = np.mean((window[:window_index+1] - mean)**2)
			window[:window_index] = np.exp(-(window[:window_index] - mono_y[y,x])**2/(2*variance))
			window[:window_index] = window[:window_index]/np.sum(window)

			for window_y in range(2*window_size+1):	
				for window_x in range(2*window_size+1):
					#accounts for bounds of image
					i = min(max(window_y - window_size,0), h)
					j = min(max(window_x - window_size,0), w)
					if i == x and j == y:
						continue
					constraints[pixel_index,i,j] = window[window_index]
					window_index += 1

		constraints[pixel_index, y, x] = mono_y[y,x]
		pixel_index += 1

A = constraints.reshape((constraints.shape[0],-1))

b = np.zeros((h,w))
b[scribbled] = scribble_u[scribbled]
b = b.flatten('F')
chromo_u = np.linalg.lstsq(A,b).reshape((h,w),order='F')

b = np.zeros((h,w))
b[scribbled] = scribble_v[scribbled]
b = b.flatten('F')
chromo_v = np.linalg.lstsq(A,b).reshape((h,w),order='F')

def yiq_to_rgb(y, i, q):                                                        # the code from colorsys.yiq_to_rgb is modified to work for arrays
    r = y + 0.948262*i + 0.624013*q
    g = y - 0.276066*i - 0.639810*q
    b = y - 1.105450*i + 1.729860*q
    r[r < 0] = 0
    r[r > 1] = 1
    g[g < 0] = 0
    g[g > 1] = 1
    b[b < 0] = 0
    b[b > 1] = 1
    return (r, g, b)

(R, G, B) = yiq_to_rgb(mono_y,chromo_u,chromo_v)
colorizedRGB = np.zeros(colorized.shape)
colorizedRGB[:,:,0] = R                                                         # colorizedRGB as colorizedIm
colorizedRGB[:,:,1] = G
colorizedRGB[:,:,2] = B

plt.imshow(colorizedRGB)
plt.show()



