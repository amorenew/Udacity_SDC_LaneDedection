import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

# Read in the image and print out some stats
image = mpimg.imread('test.jpg')
print('This image is: ', type(image),
      'with dimensions: Height>>', image.shape[0],
      'width>>', image.shape[1], 'RGB Color Layers>>', image.shape[2])

# Grab the x and y size and make a copy of the image
ysize = image.shape[0]
xsize = image.shape[1]

# Note: always make a copy rather than simply using "="
color_select = np.copy(image)

# Define our color selection criteria
# Note: if you run this code, you'll find these are not sensible values!!
# But you'll get a chance to play with them soon in a quiz
red_threshold = 0
green_threshold = 0
blue_threshold = 0
rgb_threshold = [red_threshold, green_threshold, blue_threshold]

# Identify pixels below the threshold in all color layers image [:, : , 0] mean all x and y points in red layer
thresholds = (image[:, :, 0] < rgb_threshold[0]) \
             | (image[:, :, 1] < rgb_threshold[1]) \
             | (image[:, :, 2] < rgb_threshold[2])
color_select[thresholds] = [0, 0, 0]

# Display the image
plt.imshow(color_select)
plt.show()
