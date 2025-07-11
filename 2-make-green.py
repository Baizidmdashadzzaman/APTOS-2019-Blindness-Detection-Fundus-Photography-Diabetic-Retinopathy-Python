import cv2
import numpy as np

# Load the image you want to transform (the 'source' image)
img_source = cv2.imread('sharpened_retina2.jpg')

# --- Step 1: Conceptual Removal of Vessel Overlay ---
# THIS IS THE HARDEST PART WITHOUT THE ORIGINAL IMAGE.
# You might need advanced techniques or have access to a version without the overlay.
# For demonstration, we'll assume we work on the image *with* the overlay and accept potential artifacts.
# A sophisticated approach would involve pixel analysis around the pink lines or an inpainting algorithm.
# For now, we'll proceed directly to color/contrast on the image as is.

# --- Step 2: Color Correction (Introduce Cool/Greenish Tint and Desaturation) ---
# This will be highly experimental and require tuning.

# Method 1: Adjusting individual channels (simple, but requires careful tuning)
b, g, r = cv2.split(img_source)

# Decrease Red (to remove warmth)
r_adjusted = cv2.addWeighted(r, 0.7, np.zeros_like(r), 0, 0) # Reduce red component

# Increase Green and Blue (to add cool tint)
g_adjusted = cv2.addWeighted(g, 1.2, np.zeros_like(g), 0, 0) # Increase green component
b_adjusted = cv2.addWeighted(b, 1.1, np.zeros_like(b), 0, 0) # Increase blue component

img_color_adjusted = cv2.merge([b_adjusted, g_adjusted, r_adjusted])

# Method 2: Gamma correction to manipulate mid-tones (can influence overall brightness and contrast)
# A gamma > 1.0 will darken mid-tones, which can contribute to the higher contrast/blown out highlights look.
# However, for a desaturated, lower contrast look, you might need a different approach.
# Let's try to increase overall exposure in bright areas and reduce contrast.

# --- Step 3: Contrast and Brightness Adjustment ---

# Convert to LAB for better luminance control
lab_img = cv2.cvtColor(img_color_adjusted, cv2.COLOR_BGR2LAB)
L, A, B = cv2.split(lab_img)

# Lower Contrast on L channel
# Simple linear contrast adjustment: L_new = alpha * L + beta
# alpha < 1 to reduce contrast
# beta to adjust brightness
alpha = 0.7 # Reduce contrast
beta = -30 # Make it a bit darker overall to match the darker parts of image.jpeg
L_adjusted = cv2.convertScaleAbs(L, alpha=alpha, beta=beta) # Use convertScaleAbs for alpha/beta
# You might also want to clamp values if they go out of range

# Merge back and convert to BGR
img_contrast_adjusted = cv2.cvtColor(cv2.merge([L_adjusted, A, B]), cv2.COLOR_LAB2BGR)

# Apply a global brightness adjustment if needed
# img_final = cv2.addWeighted(img_contrast_adjusted, 1, np.zeros(img_contrast_adjusted.shape, img_contrast_adjusted.dtype), 0, -20) # Decrease brightness by 20


# --- Step 4: Slight Blurring (Optional) ---
# To simulate less sharpness
img_final = cv2.GaussianBlur(img_contrast_adjusted, (3, 3), 0)


# Display the results
# Display the original image in a mid-sized window
cv2.namedWindow('Original Source', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Original Source', 600, 600)
cv2.imshow('Original Source', img_source)

# Display the converted image in a mid-sized window
cv2.namedWindow('Converted Attempt', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Converted Attempt', 600, 600)
cv2.imshow('Converted Attempt', img_final)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the result
cv2.imwrite('converted_to_image_jpeg_style2.png', img_final)