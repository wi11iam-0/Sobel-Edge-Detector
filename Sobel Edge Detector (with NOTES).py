'''
Criteria:

- convert image to grayscale
- apply x kernel to grayscale image to get the x gradient
- apply y kernel to grayscale image to get the y gradient
- find the overall gradient
- if overall gradient at that point is above a certain threshold, consider edge
- display new image with 'edges' detected
'''

from PIL import Image
import numpy as np


def sobel_filter(image_path, threshold):
    #applyies the sobel filter using the previous function

    kernel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    kernel_y = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])

    # find the gradients at each pixel
    gradient_x = convolve_image(image_path, kernel_x)
    gradient_y = convolve_image(image_path, kernel_y)
    # both these variables are image arrays but instead of holding the instensity of each pixel, it holds the gradients at each pixel point in their repective direction

    # find overall graident, again this is just an image array holding the overall gradient at each pixel point
    overall_gradient = np.sqrt(np.square(gradient_x) + np.square(gradient_y))

    # create new image where if the overall gradient at a point exceedes the threshold, an edge is displayed
    filtered_image = (overall_gradient > threshold).astype(np.uint8) * 255
    # each element in filtered_image will either be 0 or 225
    # filtered_image is just a new image array with elements 0 or 225 depending on whether the overall_gradient array was greater than the threshold at that point
    
    return Image.fromarray(filtered_image)

def convolve_image(image_path, kernel):
    # takes the grayscale image and applies a kernel to each pixel point

    # convert OG image to grayscale and make it into an array
    gray_image = Image.open(image_path).convert('L')
    image_array = np.array(gray_image, dtype=float)

    # get image and kernel dimensions
    image_height, image_width = image_array.shape
    kernel_height, kernel_width = kernel.shape

    # create an output array with same dimensions as image_array but all elements 0
    convolved_image = np.zeros_like(image_array)

    # pad the image to handle edges, this adds 0s around the image array so that the kernel can be applied around the edges of the image (0 doesnt affect the output)
    padded_image = np.pad(image_array, ((kernel_height//2, kernel_height//2), (kernel_width, kernel_width)), mode='constant')
    #(kernel_height//2, kernel_height//2) - this section takes half the kernel's heigth and adds it to the top and bottom of the image_array
    # basically increasing the number of rows and these new rows all have elements 0 in them so as to not affect the output
    # mode = 'constant' specifies that the pads will be of constant value (which is by default 0)

    # convolve the image
    for i in range(image_height):
        for j in range(image_width):
            # apply the kernel to [i, j] 
            convolved_image[i, j] = np.sum(padded_image[i:i+kernel_height, j:j+kernel_width] * kernel)
            # i:i+kernel_height means from i to i+kernel_height
            # this means that only a section of the padded image is taken into account in this operation
            # np.sum takes all the resulting elements after they have been multiplied by the kernel and sums them
            # this result is stored in the array convolved_image at the point [i, j]
            
            # high value means large constrast and most likely an edge in the OG image

    return convolved_image

# main
image_path = 'Sobel Edge Detector/963852.png'
filtered_image = sobel_filter(image_path, threshold=100)

filtered_image.show()
