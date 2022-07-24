## What is Digital Image ?

A digital image can be interpreted as a rectangular array of numbers. In many cases it's easier to understand a gray-scale image, an image that is made up of different shades of gray. Image is comprised of a rectangular grid of blocks called pixels. Representation of these pixels with numbers called intensity values. Digital images have intensity values between zero and 255. 

- The pillow, or PIL library is a popular library for working with images in Python.
```
from PIL import Image
# load the image and create a PIL image object. 
image = Image.open(my_image)
import matplotlib.pyplot as plt
# plot images
plt.imshow(image)
# format image
image.format:PNG
# image size in pixel
image.size:(512,521)
# image color
image.mode:RGB

from PIL import ImageOps
# convert image into greyscale
image_gray = ImageOps.grayscale(image)
# mode =Luminous
image_gray.mode:L
# save image
image_gray.save("lenna.jpg")

#quantize image
image_gray.quantize(2)

#convert specific color in the image into gray
red,green,blue = image.split() 

# convert PIL image into numpy array
import numpy as np
array = np.array(image)
```

- OpenCV is a library used for computer vision. It has more functionality then PIL. But it is more difficult to use. Opencv always (AFAIK) uses BGR channel order.
```
import cv2
# openCV image is a numpy array,with intensity values as 8-bit unsigned
image = cv2.imread(my_image)
# numpy array result with intensity values as 8-bit, unassigned data types
type(image):numpy.ndarray
# shape of array
image.shape:(512,512,3)
import matplotlib.pyplot as plt
plt.imshow(image)

#change color space
new_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(new_image)

# convert the image in grey scale
new_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(new_gray)

# convert the image to gray-scale using cvtColor.
im_gray = cv2.imread('name_of_image.png', cv2.IMREAD_GRAYSCALE)

# load a color image and use slices to obtain the different color channels
baboon=cv2.imread('baboon.png')
blue,green,red=baboon[:,:,0],baboon[:,:,1],baboon[:,:,2]
```

#### Manupulating Image:

```
# Copying an Image
from PIL import Image
import cv2
import numpy as np
baboon = cv2.imread("color_image.png")
baboon = np.array(Image.open("baboon.png"))
id(baboon): 140555615378048
A=baboon
id(A): 140555615378048
B= baboon.copy()

# Fliping an image
from PIL import ImageOps
im_flip = ImageOps.flip(image)

# Mirro the image
im_mirror = ImageOps.mirror(image)

# Describe the flip upside-down
image.transpose(Image.FLIP_TOP_BOTTOM)

# Flip image using cv2
import cv2
im_flip = cv2.flip(image,0)

# Rotate Image 90 DEGREE
im_flip = cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE)
```

#### Manupulating Images One Pixel At a Time:

```
# cropping using slicing
from PIL import Image
# PIL array
import numpy as np
image = Image.open("cat_image.png")
image = np.array(image)
# PIL crop
crop_image=image.crop((left,upper,right,lower))
# Change Image Pixel and draw new one
from PIL import ImageDraw
image_draw=image.copy()
image_fn=ImageDraw.Draw(im=image_draw)
shape=[left,upper,right,lower]
image_fn.rectangle(xy=shape,fill="red")
# Insert text on image
from PIL import ImageFont
fnt=ImageFont.truetype('/Library/Fonts/Arial.ttf',100)
image_fn.text(xy=(0,0),text="box",font=fnt,fill(0,0,0))
# Impose other image on current image
image_lenna=Image.open("image1.png")
left=150
upper=150
image1.paste(new_image, box=(left,upper))

# cv array
import cv2
image=cv2.image("cat_image.png")
# crop from top
upper=150
lower=400
crop_top=image[upper:lower,:,:]
# crop horizontally
left=150
right=400
crop_horizontal= crop_top[:,left:right,:]
# image pixel manupulation or change image pixel
image_draw=np.copy(image)
left=150
upper=150
right=400
lower=400
start_point,end_point=(left,upper), (right,lower)
image_draw=np.copy(image)
cv2.rectangle(image_draw,pt1=start_point,pt=end_point,color=(0,255,0),thickness=3)
cv2.putText(img=image,text='Stuff,org=(10,500),color=(255,255,255),fontFace=4, fontScale=5, thickness=2)
```

#### Pixel Tranformation:
1. Histograms
2. Intensity Transformations
3. Thresholding and Simple Segmentation

*Histogram:* A histogram counts the number of occurrences of a pixel, and it's a useful tool for understanding and manipulating images.

*Intensity Transformations:* Intensity Transformation can change one pixel at a time.

*Thresholding adn Simple Segmentation:* 

```
# histogram count
import cv2
goldhill=cv2.imread("goldhill.bmp")
hist=cv2.calcHist([goldhill],[0],None,[256],[0,255])

# Image Negatives
image = cv2.imread("mammogram.png",cv2.IMREAD_GRAYSCALE)
# apply transformation as array operations
img_neg=-1*image+255

# Brightness and Contrast Adjustments
alpha=1 #Simple contrast control
beta=100 # Simple brightness control
new_image=cv2.convertScaleAbs(goldhill, alpha=alpha, beta=beta)
#change contrast
alpha=2 #Simple contrast control
beta=100 # Simple brightness control
new_image=cv2.convertScaleAbs(goldhill, alpha=alpha, beta=beta)

# Histogram Equalization is an algorithm that uses the image's histogram to adjust contrast.
# improve contrast
zelda=cv2.imread("zelda.png",cv2.IMREAD_GRAYSCALE)
new_image=cv2.equalizeHist(zelda)

# Thresholding and  Simple Segmentation
def thresholding(input_img,threshold,max_value=255,min_value=0):
    N,M=input_img.shape
    image_out=np.zeros((N,M),dtype=np.uint8)
    max_value = 255
    min_value = 0
    threshold = 1
    for i in range(N):
        for j in range(M):
            if input_img[i,j]>threshold:
                image_out[i,j]=max_value
            else:
                image_out[i,j]=min_value

return image_out

image=cv2.imread("cameraman.jpeg",cv2.IMREAD_GRAYSCALE)
max_value=255
threshold=87
ret,new-image=cv2.threshold(image,threshold,max_value,cv2.THRESH_BINARY)

# when difficult to select threshold then can use below method
ret,otsu = cv2.threshold(image,0,255,cv2.THRESH_OTSU)
ret:88.0
```

#### Geometric Tranformation:
Geometric Transformations called Affine transformations
1. Scaling: Scaling is where we reshape the image, we can shrink or expand the image in a horizontal and or vertical direction.
2. Translation: Translation is where we shift the image We can shift an image horizontally and vertically.
3. Rotation: Rotate an image counter-clockwise rotation or anti-clockwise.
4. Resize: Change the size of image.

```
# resize image PIL
from PIL import Image
image = Image.open("lenna.png")
width=512
height=512
new_width=2*width
new_height=height
new_image=image.resize((new_width, new_height)) ### if you want to you can also shrink the image but for that case the input must be an integer.

# Rotate Image
image = Image.open("lenna.png"0)
theta=45
new_image =image.rotate(theta)

# Resize image in Opencv
import cv2
image=cv2.imread("lenna.png")
new_image=cv2.resize(image, None,fx=2,fy=1,interpolation=cv2.INTER_CUBIC)

#tranlation
rows,cols,_=image,shape
tx=100
ty=0
M=np.float32([[1,0,tx],[0,1,ty]])
new_image =cv2.warpAffine(image,M,(cols,rows))

# rotation
theta=45.0
M=cv2.getRotationMatrix2D(center=(cols//2-1,rows//2-1),angle=theta,scale=1)
```

#### Spatial Operations in Image Processing:
1. Convolution: Linear Filtering: Convolution or linear filtering is a standard way to Filter an image using convolution. The filter is called the kernel. Different kernels perform different tasks.
- Mean Filtering
- Edge Detection
3. Median Filters

*Convoltuion:Linear Filtering:*
Convolution or linear filtering is a standard way to Filter an image the filter is called the kernel the different kernels perform different tasks. - Mean Filtering

*Edge Detection:* Edges in a digital image are where the image brightness changes sharply. Edge detection uses methods to approximate derivatives and gradients for identifying these areas.

*Median Filtering:*  Median Filters are another popular filter, they are better at removing some types of Noise but may distort the image.

```
# Filtering
import cv2
new_image= image+Noise
kernel =np.ones((6,6))/36
image_filtered=cv2.filter2D(src=new_image, ddepth=-1, kernel=kernel)

# Image sharpening: Enhancing or smoothing the image
kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
image_filtered=cv2.filter2D(src=image, ddepth=-1, kernel=kernel)

#Edge detection
img_gray=cv2,imread('barbara.png',cv2.IMREAD_GRAYSCALE)
# Smooth image using GaussianBlur low pass fiter and it is also called Gaussian smoothing
img_gray = cv2.GaussianBlur(img_gray,(3,3),sgimaX=0.1,sigmaY=0.1)

# Sobel filter
ddepth = cv2.CV_16S
grad_x=cv2.Sobel(src=img_gray,ddepth=ddepth,dx=1, dy=0,ksize=3)
grad_y=cv2.Sobel(src=img_gray,ddepth=ddepth,dx=0,dy=1,ksize=3)

# gradient
abs_grad_x=cv2.convertScaleAbs(grad_x)
abs_grad_y=sv2.convertScaleAbs(grad_y)
grad=cv2.addWeighted(abs_grad_x,0.5,abs_grad_y,0.5,0)



