import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load and resize image
img_path = r"C:\Users\Lenovo\Desktop\Dip Assignment\IMG_7450.JPG"  # Make sure this is correct
img_path = r"C:\Users\Lenovo\Desktop\Dip Assignment\IMG_7450.JPG"  # Remove the extra closing parenthesis
img = cv2.imread(img_path)
img = cv2.resize(img, (400, 400))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# 1. Negative
negative = 255 - gray

# 2. Contrast Stretching
def contrast_stretch(image):
    min_val = np.min(image)
    max_val = np.max(image)
    stretched = (image - min_val) * (255 / (max_val - min_val))
    return np.uint8(stretched)

contrast = contrast_stretch(gray)

# 3. Laplacian Filter
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
laplacian = np.uint8(np.absolute(laplacian))

# 4. Thresholding
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# 5. Gray Level Slicing
def gray_level_slicing(image, lower, upper):
    sliced = np.where((image >= lower) & (image <= upper), 255, image)
    return np.uint8(sliced)

sliced = gray_level_slicing(gray, 100, 180)

# 6. Log Transformation
def log_transform(image):
    image = image.astype(np.float32)
    c = 255 / np.log(1 + np.max(image))
    log_img = c * np.log(1 + image)
    return np.uint8(log_img)

log_img = log_transform(gray)

# 7. Gamma Transformation
def gamma_transform(image, gamma):
    image = image / 255.0
    gamma_img = np.power(image, gamma)
    gamma_img = gamma_img * 255
    return np.uint8(gamma_img)

gamma_img = gamma_transform(gray, 2.0)

# 8. Inverse Log Transformation
def inverse_log_transform(image):
    image = image.astype(np.float32) / 255
    inv_log = np.exp(image) - 1
    inv_log = inv_log / np.max(inv_log) * 255
    return np.uint8(inv_log)

inverse_log = inverse_log_transform(gray)

# 9. Bit Plane Slicing
def bit_plane_slicing(image):
    planes = []
    for i in range(8):
        plane = (image >> i) & 1
        planes.append(np.uint8(plane * 255))
    return planes

bit_planes = bit_plane_slicing(gray)

# 10. Histogram Equalization
hist_eq = cv2.equalizeHist(gray)

# ========== Display All Filters ========== #
titles = [
    'Original (Gray)', 'Negative', 'Contrast Stretching',
    'Laplacian', 'Thresholding', 'Gray Level Slicing',
    'Log Transform', 'Gamma Transform',
    'Inverse Log', 'Histogram Equalized'
]

images = [
    gray, negative, contrast,
    laplacian, thresh, sliced,
    log_img, gamma_img,
    inverse_log, hist_eq
]

plt.figure(figsize=(20, 12))
for i in range(len(images)):
    plt.subplot(3, 4, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()

# ========== Display Bit Planes ========== #
plt.figure(figsize=(16, 6))
for i in range(8):
    plt.subplot(2, 4, i+1)
    plt.imshow(bit_planes[i], cmap='gray')
    plt.title(f'Bit Plane {i}')
    plt.axis('off')
plt.tight_layout()
plt.show()

# ========== Save All Filters ========== #
save_names = [
    'gray.jpg', 'negative.jpg', 'contrast.jpg',
    'laplacian.jpg', 'threshold.jpg', 'sliced.jpg',
    'log.jpg', 'gamma.jpg', 'inverse_log.jpg', 'hist_eq.jpg'
]

for name, image in zip(save_names, images):
    cv2.imwrite(name, image)

# Save Bit Planes
for i, plane in enumerate(bit_planes):
    cv2.imwrite(f'bit_plane_{i}.jpg', plane)
