#%%
import cv2
import numpy as np
from scipy import misc
i = misc.ascent()

#%%
import matplotlib.pyplot as plt
plt.grid(False)
plt.gray()
plt.axis('off')
plt.imshow(i)
plt.show()

#%%
i_transformed = np.copy(i)
size_x = i_transformed.shape[0]
size_y = i_transformed.shape[1]

#%%
filter = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
weight = 1

#%%
for x in range(size_x-2):
  for y in range(size_y-2):
    conv = 0.0
    conv = conv + (i[x,y] * filter[0][0])
    conv = conv + (i[x+1,y] * filter[0][1])
    conv = conv + (i[x+2,y] * filter[0][2])
    conv = conv + (i[x,y+1] * filter[1][0])
    conv = conv + (i[x+1,y+1] * filter[1][1])
    conv = conv + (i[x,y+1] * filter[1][2])
    conv = conv + (i[x,y+2] * filter[2][0])
    conv = conv + (i[x+1,y+2] * filter[2][1])
    conv = conv + (i[x+2,y+2] * filter[2][2])
    conv = conv * weight
    if(conv < 0):
      conv = 0
    if(conv > 255):
      conv =255
    i_transformed[x+1,y+1] = conv

#%%
plt.gray()
plt.grid(False)
plt.imshow(i_transformed)
plt.show()

#%%
# Max Pooling
new_image = np.zeros((int(size_x/2), int(size_y/2)))
for x in range(0, size_x, 2):
  for y in range(0, size_y, 2):
    pixels = []
    pixels.append(i_transformed[x,y])
    pixels.append(i_transformed[x+1,y])
    pixels.append(i_transformed[x,y+1])
    pixels.append(i_transformed[x+1,y+1])
    new_image[int(x/2),int(y/2)] = max(pixels)

# Plot the image
plt.gray()
plt.grid(False)
plt.imshow(new_image)
plt.show()