# Image Filtering
A comprehensive tutorial towards 2D convolution and image filtering (The first step to understand Convolutional Neural 
Networks (CNNs)).

## Introduction
Convolution is one of the most important operations in signal and image processing. It could operate in 1D (e.g. speech 
processing), 2D (e.g. image processing) or 3D (video processing). Here, we will discuss convolution in 2D spatial which 
is mostly used in image processing for feature extraction and is also the core block of Convolutional Neural Networks (CNNs). 
Generally, we can consider an image as a matrix whose elements are numbers between 0 and 255. The size of this matrix is 
_(image height)_ x _(image width)_ x _(image channels)_. A grayscale image has 1 channel where a color image has 3 channels 
(for an RGB). 

![input_image](https://user-images.githubusercontent.com/35737777/68632461-8646d680-04e6-11ea-9106-774bfd96d0ad.jpg)

We can load and plot the image using opencv library in python:

```python
import cv2
def load_image(image_path):
    """
    Load the image using opencv
    :param image_path: <String> Path of input_image
    """
    coloured_image = cv2.imread(image_path)
    grey_image = cv2.cvtColor(coloured_image, cv2.COLOR_BGR2GRAY)
    print('image matrix size: ', grey_image.shape)
    print('\n First 5 columns and rows of the image matrix: \n', grey_image[:5, :5])
    cv2.imwrite('TopLeft5x5.jpg', grey_image[:5, :5])
    return grey_image

input_image = load_image('input_image.jpg')
```
![figure2](https://user-images.githubusercontent.com/35737777/68632478-95c61f80-04e6-11ea-9b07-aaa2c8aef1d3.jpg)

## Convolution
Each convolution operation has a kernel which could be a any matrix smaller than the original image in height and width. 
Each kernel is useful for a specific task, such as sharpening, blurring, edge detection, and more. Let's start with the 
sharpening kernel which is defined in [Types of Kernel](https://en.wikipedia.org/wiki/Kernel_(image_processing).

As previously mentioned, each kernel has a specific task to do and the sharpen kernel accentuate edges but with the cost 
of adding noise to those area of the image which colors are changing gradually. The output of image convolution is 
calculated as follows:

1. Flip the kernel both horizontally and vertically. As our selected kernel is symmetric, the flipped kernel is equal to 
the original.
2. Put the first element of the kernel at every pixel of the image (element of the image matrix). Then each element of the 
kernel will stand on top of an element of the image matrix.
![figure3](https://user-images.githubusercontent.com/35737777/68632479-95c61f80-04e6-11ea-80b2-2e86a4fcc258.jpg)
3. Multiply each element of the kernel with its corresponding element of the image matrix (the one which is overlapped 
with it)
4. Sum up all product outputs and put the result at the same position in the output matrix as the center of kernel in 
image matrix.
![figure4](https://user-images.githubusercontent.com/35737777/68632480-965eb600-04e6-11ea-8c0d-394e0e216e21.jpg)
5. For the pixels on the border of image matrix, some elements of the kernel might stands out of the image matrix and 
therefore does not have any corresponding element from the image matrix. In this case, we can eliminate the convolution 
operation for these position which end up an output matrix smaller than the input (image matrix) or we can apply padding 
to the input matrix (based on the size of the kernel we might need one or more pixels padding, in our example we just 
need 1 pixel padding):
![figure5](https://user-images.githubusercontent.com/35737777/68632482-965eb600-04e6-11ea-8924-9cf9514ad101.jpg)

As you can see in Figure 5, the output of convolution might violate the input range of [0-255]. Even though the python 
packages would take care of it by considering the maximum value of the image as the pure white (correspond to 255 in [0-255] 
scale) and the minimum value as the pure black (correspond to 0 in [0-255] scale), the values of the convolution output 
(filtered image) specially along the edges of the image (which are calculated based on the added zero padding) can cause 
a low contrast filtered image. Here, to overcome this loss of contrast issue, we can use Histogram Equalization technique. 
However, we might be able to end up with a better contrast neglecting the zero padding. The following python code convolves 
an image with the sharpen kernel and plots the result:

```python
def convolve2d(image, kernel):
    """
    This function which takes an image and a kernel and returns the convolution of them.

    :param image: a numpy array of size [image_height, image_width].
    :param kernel: a numpy array of size [kernel_height, kernel_width].
    :return: a numpy array of size [image_height, image_width] (convolution output).
    """
    # Flip the kernel
    kernel = np.flipud(np.fliplr(kernel))
    # convolution output
    output = np.zeros_like(image)

    # Add zero padding to the input image
    image_padded = np.zeros((image.shape[0] + 2, image.shape[1] + 2))
    image_padded[1:-1, 1:-1] = image

    # Loop over every pixel of the image
    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            # element-wise multiplication of the kernel and the image
            output[y, x]=(kernel * image_padded[y: y+3, x: x+3]).sum()

    return output

# kernel to be used to get sharpened image
KERNEL = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
image_sharpen = convolve2d(input_image, kernel=KERNEL)
cv2.imwrite('sharpened_image.jpg', image_sharpen)
```

and you can see the filtered image after applying sharpen filter below:
![sharpened_image](https://user-images.githubusercontent.com/35737777/68632484-965eb600-04e6-11ea-876a-215b5946dff9.jpg)

## More Filters
### Edge Detection
There are many other filters which are really useful in image processing and computer vision. One of the most important 
one is edge detection. Edge detection aims to identify pixels of an image at which the brightness changes drastically. 
Let's apply one of the simplest edge detection filters to our image and see the result.

```python
# kernel to be used for edge detection
image_edge1 = convolve2d(input_image,
                         kernel=np.array([[-1, -1, -1], [-1, 4, -1],[-1, -1, -1]]))
cv2.imwrite('edge_detection1.jpg', image_edge1)

image_edge2 = convolve2d(input_image,
                         kernel=np.array([[-1, 0, 1], [0, 0, 0], [1, 0, -1]]))
cv2.imwrite('edge_detection2.jpg', image_edge2)
```
![edge_detection1](https://user-images.githubusercontent.com/35737777/68632486-965eb600-04e6-11ea-8766-9abc2cb31001.jpg)
![edge_detection2](https://user-images.githubusercontent.com/35737777/68632477-95c61f80-04e6-11ea-9273-09831e904f52.jpg)

### Blur the Image
#### Box Blur
Now it iss time to apply a filter to the noisy image and reduce the noise. Blur filter could be a smart choice:

```python
# kernel to be used for box blur
imageboxblur = convolve2d(input_image,
                         kernel=np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])/9.0)
cv2.imwrite('box_blur.jpg', imageboxblur)

# kernel to be used for gaussian blur
imagegaussianblur = convolve2d(input_image,
                         kernel=np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])/16.0)
cv2.imwrite('gaussian_blur.jpg', imagegaussianblur)
```
![box_blur](https://user-images.githubusercontent.com/35737777/68632485-965eb600-04e6-11ea-848e-cd29c5682b42.jpg)
![gaussian_blur](https://user-images.githubusercontent.com/35737777/68632483-965eb600-04e6-11ea-8107-9c00eb3478f4.jpg)
