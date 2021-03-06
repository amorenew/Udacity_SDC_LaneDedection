import cv2
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
# canny_edge_select = np.copy(image)

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

# Read in the image and convert to grayscale
gray_select = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# Define a kernel size for Gaussian smoothing / blurring
# Note: this step is optional as cv2.Canny() applies a 5x5 Gaussian internally
kernel_size = 5  # Must be an odd number (3, 5, 7...)
blur_gray = cv2.GaussianBlur(gray_select, (kernel_size, kernel_size), 0)

# Define parameters for Canny and run it
# NOTE: if you try running this code you might want to change these!
low_threshold = 50
high_threshold = 150
edges_select = cv2.Canny(blur_gray, low_threshold, high_threshold)

# Next we'll create a masked edges image using cv2.fillPoly()
mask = np.zeros_like(edges_select)
ignore_mask_color = 255

# This time we are defining a four sided polygon to mask
vertices = np.array([[(0, ysize), (xsize * .30, ysize * .63), (xsize * .60, ysize * .64), (xsize, ysize)]],
                    dtype=np.int32)
cv2.fillPoly(mask, vertices, ignore_mask_color)
masked_edges = cv2.bitwise_and(edges_select, mask)

# Define the Hough transform parameters
# Make a blank the same size as our image to draw on
rho = 1  # distance resolution in pixels of the Hough grid
theta = np.pi / 180  # angular resolution in radians of the Hough grid
threshold = 1  # minimum number of votes (intersections in Hough grid cell)
min_line_length = 7  # minimum number of pixels making up a line
max_line_gap = 3  # maximum gap in pixels between connectible line segments
line_image = np.copy(image) * 0  # creating a blank to draw lines on

# Run Hough on edge detected image
lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

# Iterate over the output "lines" and draw lines on the blank
for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

# Create a "color" binary image to combine with line image
color_edges = np.dstack((edges_select, edges_select, edges_select))

# Draw the lines on the edge image
combine_edges_lines = cv2.addWeighted(image, 0.8, line_image, 1, 0)

# Display the image
# plt.imshow(masked_edges)
# plt.show()
plt.imshow(color_select)
# plt.show()
plt.imshow(region_select)
# plt.show()
plt.imshow(gray_select, cmap='gray')
# plt.show()
plt.imshow(edges_select, cmap='Greys_r')
# plt.show()
plt.imshow(line_image)
# plt.show()
plt.imshow(combine_edges_lines)
plt.show()


def process_image():
    return combine_edges_lines


# to save the image locally
mpimg.imsave("test-color-select.jpg", color_select)
mpimg.imsave("test-region-color-select.jpg", region_select)
mpimg.imsave("test-cv-gray.jpg", gray_select)
mpimg.imsave("test-cv-edges.jpg", edges_select)
mpimg.imsave("test-cv-line-image.jpg", line_image)
mpimg.imsave("test-cv-combine-edges-lines.jpg", combine_edges_lines)
