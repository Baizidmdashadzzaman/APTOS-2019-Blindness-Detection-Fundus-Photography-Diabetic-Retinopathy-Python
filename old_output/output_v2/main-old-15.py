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
# We can make the bright_enhanced image more visible here before further processing
# Let's try scaling it up even more, or applying a non-linear transform if needed
bright_enhanced = np.clip(bright_multiscale * 5.0, 0, 1) # Increased multiplication factor
dark_enhanced = np.clip(dark_multiscale * 3.0, 0, 1)

# Convert to 3-channel for visualization (initial conversion)
# This will be the base for our colored image
bright_rgb_base = cv2.merge([bright_enhanced]*3)
dark_rgb = cv2.merge([dark_enhanced]*3)

# --- Coloring the brightest regions in bright_rgb_base with parrot green ---
# Define parrot green color in RGB (normalized to [0, 1])
PARROT_GREEN = np.array([50/255, 205/255, 50/255])

# Create a mask for the brightest regions
# Adjust this threshold based on what you consider "brightest" in the bright_enhanced image
# A lower threshold will include more areas. Let's start with a more lenient one if the image is generally dark.
threshold = 0.15 # Adjusted threshold - you'll likely need to fine-tune this!
bright_mask = (bright_enhanced > threshold).astype(np.float32)

# Create an empty image for the green overlay
green_overlay = np.zeros_like(bright_rgb_base)

# Apply parrot green color to the green_overlay where the mask is active
# We multiply the mask by the green color. This creates an image that's green
# only in the regions where bright_mask is 1.
green_overlay[:,:,0] = bright_mask * PARROT_GREEN[0] # Red channel
green_overlay[:,:,1] = bright_mask * PARROT_GREEN[1] # Green channel
green_overlay[:,:,2] = bright_mask * PARROT_GREEN[2] # Blue channel

# Blend the original bright_rgb_base with the green_overlay
# We can use cv2.addWeighted for blending
# The alpha parameter (e.g., 0.5) controls the opacity of the green overlay.
# A higher alpha will make the green more dominant.
# This ensures that the underlying structure is still somewhat visible.
alpha = 0.7 # Opacity of the green overlay (0 to 1)
colored_bright_rgb = cv2.addWeighted(bright_rgb_base, 1 - alpha, green_overlay, alpha, 0)

# If you just want to completely replace the bright areas with solid green:
# colored_bright_rgb = bright_rgb_base.copy()
# for c in range(3):
#     colored_bright_rgb[bright_mask > 0, c] = PARROT_GREEN[c]


# Show side-by-side
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
titles = ['(a) Original RGB', '(b) Bright Region (Top-Hat) - Green Highlighted', '(c) Dark Region (Black-Hat)']
images = [img_rgb, colored_bright_rgb, dark_rgb] # Use colored_bright_rgb here

for ax, img_to_show, title in zip(axs, images, titles):
    ax.imshow(img_to_show) # No need for cmap='gray' for RGB images
    ax.set_title(title, fontsize=14)
    ax.axis('off')

plt.tight_layout()
plt.savefig("retina_decomposition_output_multiscale_green_highlighted_v2.png", dpi=300)
plt.show()