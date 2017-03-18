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
region_select = np.copy(image)

# Define our color selection criteria
# Note: if you run this code, you'll find these are not sensible values!!
# But you'll get a chance to play with them soon in a quiz
red_threshold = 200
blue_threshold = 200
green_threshold = 200
blue_threshold = 200
rgb_threshold = [red_threshold, green_threshold, blue_threshold]

# Identify pixels below the threshold in all color layers image [:, : , 0] mean all x and y points in red layer
color_thresholds = (image[:, :, 0] < rgb_threshold[0]) \
                   | (image[:, :, 1] < rgb_threshold[1]) \
                   | (image[:, :, 2] < rgb_threshold[2])
color_select[color_thresholds] = [0, 0, 0]

# Define a triangle region of interest
# Keep in mind the origin (x=0, y=0) is in the upper left in image processing
# Note: if you run this code, you'll find these are not sensible values!!
# But you'll get a chance to play with them soon in a quiz
left_bottom = [0, ysize]
right_bottom = [xsize, ysize]
apex = [xsize / 2, ysize / 2]

# Fit lines (y=Ax+B) to identify the  3 sided region of interest
# np.polyfit() returns the coefficients [A, B] of the fit
fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)

# Find the region inside the lines
XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
region_thresholds = (YY > (XX * fit_left[0] + fit_left[1])) & \
                    (YY > (XX * fit_right[0] + fit_right[1])) & \
                    (YY < (XX * fit_bottom[0] + fit_bottom[1]))

# Color pixels red which are inside the region of interest
# region_select[region_thresholds] = [255, 0, 0]
# Find where image is both colored right and in the region
# combine color select and region of interest
region_select[~color_thresholds & region_thresholds] = [255, 0, 0]

# Display the image
plt.imshow(color_select)
plt.show()
plt.imshow(region_select)
plt.show()
# to save the image locally
mpimg.imsave("test-color-select.jpg", color_select)
mpimg.imsave("test-region-color-select.jpg", region_select)
