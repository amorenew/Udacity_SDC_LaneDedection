import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    # Read in the image and convert to grayscale
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    # Define a kernel size for Gaussian smoothing / blurring
    # Note: this step is optional as cv2.Canny() applies a 5x5 Gaussian internally
    # kernel_size = 5  # Must be an odd number (3, 5, 7...)
    gray_image = grayscale(img)
    return cv2.GaussianBlur(gray_image, (kernel_size, kernel_size), 0)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    # Define parameters for Canny and run it
    # NOTE: if you try running this code you might want to change these!
    kernel_size = 5  # Must be an odd number (3, 5, 7...)
    blur_image = gaussian_blur(img, kernel_size)
    return cv2.Canny(blur_image, low_threshold, high_threshold)


def make_vertices(img):
    # Grab the x and y size of the image
    ysize = img.shape[0]
    xsize = img.shape[1]

    # This time we are defining a four sided polygon to mask
    vertices = np.array([[(0, ysize), (xsize * .30, ysize * .63), (xsize * .60, ysize * .64), (xsize, ysize)]],
                        dtype=np.int32)
    return vertices


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    low_threshold = 50
    high_threshold = 150
    edges_image = canny(img, low_threshold, high_threshold)
    mask = np.zeros_like(edges_image)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(edges_image, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    # line_image = np.copy(img) * 0  # creating a blank to draw lines on
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 1  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 7  # minimum number of pixels making up a line
    max_line_gap = 3  # maximum gap in pixels between connectible line segments
    masked_image = region_of_interest(img, make_vertices(img))
    http: // opencv - users
    .1802565.n2.nabble.com / merge - lines - close - by - td5534403.html
    lines = cv2.HoughLinesP(masked_image, rho, theta, threshold, np.array([]), min_line_length,
                            max_line_gap)
    line_img = np.copy(image) * 0  # creating a blank to draw lines on
    draw_lines(line_img, lines)
    return line_img


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


image = mpimg.imread('test.jpg')
# plt.imshow(grayscale(image), cmap='gray')
# plt.show()
# plt.imshow(canny(image, 50, 150), cmap='Greys_r')
# plt.show()
# plt.imshow(region_of_interest(image, make_vertices(image)), cmap='Greys_r')
plt.imshow(weighted_img(hough_lines(image), image))
plt.show()
