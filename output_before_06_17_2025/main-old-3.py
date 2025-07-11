import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the original image
img = cv2.imread("old/4.png")
if img is None:
    raise FileNotFoundError("Image not found. Check path and filename.")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Normalize grayscale for processing
gray_norm = img_gray / 255.0

# Morphological kernel
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

# Red lesion attention (simulate with black-hat)
red_att = cv2.morphologyEx(gray_norm, cv2.MORPH_BLACKHAT, kernel)
red_att_norm = np.clip(red_att * 5.0, 0, 1)

# Bright lesion attention (simulate with top-hat)
bright_att = cv2.morphologyEx(gray_norm, cv2.MORPH_TOPHAT, kernel)
bright_att_norm = np.clip(bright_att * 5.0, 0, 1)

# Apply colormap overlays for heatmaps
def apply_colormap_overlay(base_img, heatmap, alpha=0.6):
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(base_img, 1 - alpha, heatmap_color, alpha, 0)
    return overlay

# Resize attention maps to match original image (in case dimensions changed)
if red_att_norm.shape[:2] != img_rgb.shape[:2]:
    red_att_norm = cv2.resize(red_att_norm, (img_rgb.shape[1], img_rgb.shape[0]))
    bright_att_norm = cv2.resize(bright_att_norm, (img_rgb.shape[1], img_rgb.shape[0]))

# Overlay attention maps on original RGB
red_overlay = apply_colormap_overlay(img_rgb, red_att_norm)
bright_overlay = apply_colormap_overlay(img_rgb, bright_att_norm)

# Plot the result similar to paper style
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
titles = ['(a) Original', '(b) Attention: Red Lesions', '(c) Attention: Bright Lesions']
images = [img_rgb, red_overlay, bright_overlay]

for ax, im, title in zip(axs, images, titles):
    ax.imshow(im)
    ax.set_title(title, fontsize=14)
    ax.axis('off')

plt.tight_layout()
plt.savefig("attention_map_output.png", dpi=300)

