import cv2
import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt

# Step 1: Load an image
image_path = 'your_image.jpg'  # Replace with your image path
image = cv2.imread(image_path)

# Convert BGR image (OpenCV default) to RGB for display
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Step 2: Resize the image
def resize_image(image, width=None, height=None):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        ratio = height / float(h)
        dim = (int(w * ratio), height)
    else:
        ratio = width / float(w)
        dim = (width, int(h * ratio))
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized

resized_image = resize_image(image_rgb, width=300)

# Step 3: Convert to grayscale
gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

# Step 4: Apply blurring and sharpening
# Blurring
blurred_image = cv2.GaussianBlur(image_rgb, (15, 15), 0)

# Sharpening
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
sharpened_image = cv2.filter2D(image_rgb, -1, kernel)

# Step 5: Edge detection using Canny
edges = cv2.Canny(gray_image, 100, 200)

# Step 6: Adjust contrast and brightness
def adjust_contrast_brightness(image, contrast=1.0, brightness=0):
    adjusted = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
    return adjusted

contrast_adjusted = adjust_contrast_brightness(image_rgb, contrast=1.5, brightness=10)

# Step 7: Histogram equalization
def histogram_equalization(image):
    if len(image.shape) == 2:  # Grayscale image
        equalized = cv2.equalizeHist(image)
    else:  # Color image
        ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        equalized = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    return equalized

equalized_image = histogram_equalization(image_rgb)

# Step 8: Display the results
plt.figure(figsize=(15, 10))

# Original Image
plt.subplot(3, 3, 1)
plt.imshow(image_rgb)
plt.title('Original Image')
plt.axis('off')

# Resized Image
plt.subplot(3, 3, 2)
plt.imshow(resized_image)
plt.title('Resized Image')
plt.axis('off')

# Grayscale Image
plt.subplot(3, 3, 3)
plt.imshow(gray_image, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')

# Blurred Image
plt.subplot(3, 3, 4)
plt.imshow(blurred_image)
plt.title('Blurred Image')
plt.axis('off')

# Sharpened Image
plt.subplot(3, 3, 5)
plt.imshow(sharpened_image)
plt.title('Sharpened Image')
plt.axis('off')

# Edge Detection
plt.subplot(3, 3, 6)
plt.imshow(edges, cmap='gray')
plt.title('Edge Detection')
plt.axis('off')

# Contrast Adjusted Image
plt.subplot(3, 3, 7)
plt.imshow(contrast_adjusted)
plt.title('Contrast Adjusted')
plt.axis('off')

# Histogram Equalization
plt.subplot(3, 3, 8)
plt.imshow(equalized_image)
plt.title('Histogram Equalization')
plt.axis('off')

plt.tight_layout()
plt.show()