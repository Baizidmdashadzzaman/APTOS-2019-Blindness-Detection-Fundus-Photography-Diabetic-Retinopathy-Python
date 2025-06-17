import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the original image
img = cv2.imread("old/4.png")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Normalize grayscale to [0,1]
gray_norm = img_gray / 255.0

# Define multiple kernel sizes for multiscale filtering
kernel_sizes = [7, 15, 25]
bright_multiscale = np.zeros_like(gray_norm)
dark_multiscale = np.zeros_like(gray_norm)

# Apply Top-Hat and Black-Hat for each scale and accumulate results
for k in kernel_sizes:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bright = cv2.morphologyEx(gray_norm, cv2.MORPH_TOPHAT, kernel)
    dark = cv2.morphologyEx(gray_norm, cv2.MORPH_BLACKHAT, kernel)
    bright_multiscale += bright
    dark_multiscale += dark

# Average over number of scales
bright_multiscale /= len(kernel_sizes)
dark_multiscale /= len(kernel_sizes)

# Enhance visualization
bright_enhanced = np.clip(bright_multiscale * 3.0, 0, 1)
dark_enhanced = np.clip(dark_multiscale * 3.0, 0, 1)

# Convert to 3-channel for visualization
bright_rgb = cv2.merge([bright_enhanced]*3)
dark_rgb = cv2.merge([dark_enhanced]*3)

# Show side-by-side
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
titles = ['(a) Original RGB', '(b) Bright Region (Top-Hat)', '(c) Dark Region (Black-Hat)']
images = [img_rgb, bright_rgb, dark_rgb]

for ax, img, title in zip(axs, images, titles):
    ax.imshow(img, cmap='gray')
    ax.set_title(title, fontsize=14)
    ax.axis('off')

plt.tight_layout()
plt.savefig("retina_decomposition_output_multiscale.png", dpi=300)
# plt.show()  # Uncomment if you want to display the result interactively
