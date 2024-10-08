from PIL import Image
import numpy as np


def sobel_filter(image_path, threshold):

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
    
    overall_gradient = np.sqrt(np.square(gradient_x) + np.square(gradient_y))

    # create new image where if the overall gradient at a point exceedes the threshold, an edge is displayed
    filtered_image = (overall_gradient > threshold).astype(np.uint8) * 255
    
    return Image.fromarray(filtered_image)

def convolve_image(image_path, kernel):

    # convert original image to grayscale and turn it into an array
    gray_image = Image.open(image_path).convert('L')
    image_array = np.array(gray_image, dtype=float)

    # obtain image and kernel dimensions
    image_height, image_width = image_array.shape
    kernel_height, kernel_width = kernel.shape

    # create an output array with same dimensions as image_array
    convolved_image = np.zeros_like(image_array)

    # pad the image to handle edges
    padded_image = np.pad(image_array, ((kernel_height//2, kernel_height//2), (kernel_width, kernel_width)), mode='constant')
    
    # convolve the image
    for i in range(image_height):
        for j in range(image_width):
            # apply the kernel to [i, j] 
            convolved_image[i, j] = np.sum(padded_image[i:i+kernel_height, j:j+kernel_width] * kernel)
            
    return convolved_image

# main
image_path = 'Sobel Edge Detector/test image.jpeg'
filtered_image = sobel_filter(image_path, threshold=100)

filtered_image.show()
