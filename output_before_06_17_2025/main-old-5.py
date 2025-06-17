from PIL import Image
import numpy as np
from skimage.morphology import disk, opening, closing
from skimage.color import rgb2gray  # To easily convert to grayscale for morphological ops


def preprocess_fundus_image(image_path):
    """
    Preprocesses a fundus image by loading, resizing, and normalizing.
    This function is carried over from our previous discussion.
    """
    img = Image.open(image_path).convert("RGB")
    resized_img = img.resize((640, 640), Image.LANCZOS)
    img_array = np.array(resized_img, dtype=np.float32)
    # Normalize to [-1, 1] for CNN input later
    normalized_array = (img_array / 255.0 * 2) - 1
    return normalized_array


def decompose_image_colorized_approximation(preprocessed_image_array, se_radius_small=3, se_radius_large=7,
                                            bright_threshold_factor=0.02, dark_threshold_factor=0.02):
    """
    Approximates the image decomposition into colorized dark (Idark) and bright (Ibri) regions
    to visually match the paper's figures.
    THIS IS STILL AN APPROXIMATION AND NOT THE EXACT ALGORITHM FROM THE REFERENCED PAPER [34].

    Args:
        preprocessed_image_array (np.ndarray): The preprocessed image
                                               (normalized to [-1, 1], shape HxWx3).
        se_radius_small (int): Radius for the smaller structuring element.
        se_radius_large (int): Radius for the larger structuring element.
        bright_threshold_factor (float): Factor to threshold bright features.
        dark_threshold_factor (float): Factor to threshold dark features.

    Returns:
        tuple: (Idark_color, Ibri_color) - numpy arrays representing colorized dark and bright
               enhanced regions, with background black, normalized to [-1, 1].
               Shape HxWx3.
    """
    # Convert back to [0, 255] uint8 for skimage morphological operations
    img_display_scale = (((preprocessed_image_array + 1) / 2) * 255).astype(np.uint8)

    # Convert to grayscale for morphological operations
    # Green channel often works well for fundus, or a general grayscale conversion.
    # Let's try the green channel first, as it's common for vessel extraction.
    gray_img = img_display_scale[:, :, 1]
    # Alternatively, use rgb2gray for a weighted grayscale:
    # gray_img = (rgb2gray(img_display_scale) * 255).astype(np.uint8)

    # Define structuring elements
    se_small = disk(se_radius_small)
    se_large = disk(se_radius_large)  # Not used in this simplified colorized version, but kept for context.

    # --- Process for Ibri (Bright Regions) ---
    # Highlight bright regions using white top-hat transform (image - opening)
    # Opening removes bright features smaller than SE. Subtracting them recovers them.
    bright_features_raw = np.maximum(0, gray_img - opening(gray_img, se_small))

    # Normalize bright features to [0, 1] for easier thresholding
    bright_features_norm = bright_features_raw.astype(np.float32)
    if bright_features_norm.max() > 0:
        bright_features_norm /= bright_features_norm.max()

    # Create a mask for significant bright features
    bright_mask = bright_features_norm > bright_threshold_factor

    # Now, to colorize Ibri like in the paper's (b) figure:
    # We want to keep the original color of the bright features and make the rest black.
    Ibri_color = np.zeros_like(img_display_scale, dtype=np.float32)  # Initialize with black
    # Apply the bright mask to the original preprocessed image (scaled to 0-1 for easier multiplication)
    original_img_0_1 = (preprocessed_image_array + 1) / 2  # Scale back to [0, 1]

    # We'll apply the mask to each channel of the original image
    # For a yellowish/greenish tint, we can emphasize Red and Green channels
    # The paper's image (b) seems to emphasize them directly from original color.
    for c in range(3):  # Iterate over R, G, B channels
        Ibri_color[:, :, c] = original_img_0_1[:, :, c] * bright_mask

    # Normalize Ibri_color to [-1, 1]
    Ibri_color = (Ibri_color * 2) - 1

    # --- Process for Idark (Dark Regions - Vessels/Red Lesions) ---
    # Highlight dark regions using black top-hat transform (closing - image)
    # Closing fills in dark features smaller than SE. Subtracting original from it recovers them.
    dark_features_raw = np.maximum(0, closing(gray_img, se_small) - gray_img)

    # Normalize dark features to [0, 1]
    dark_features_norm = dark_features_raw.astype(np.float32)
    if dark_features_norm.max() > 0:
        dark_features_norm /= dark_features_norm.max()

    # Create a mask for significant dark features
    dark_mask = dark_features_norm > dark_threshold_factor

    # Now, to colorize Idark like in the paper's (c) figure:
    # Vessels appear green. Other dark lesions might have some red.
    Idark_color = np.zeros_like(img_display_scale, dtype=np.float32)  # Initialize with black

    # We'll apply the dark mask. For green vessels, put features primarily in the green channel.
    # The paper's image (c) also has some red and faint blue for the background or other features.
    # Let's try to emphasize green for vessels, and allow some original color for other dark areas.

    # Simple approach for green vessels:
    # Use the dark_mask to extract features. For visualization, map these to the green channel.
    # (Preprocessed image is -1 to 1. Mask is boolean. Convert original image to 0-1 for clarity)
    original_img_0_1 = (preprocessed_image_array + 1) / 2

    # Option 1: Strictly green vessels on black background (like the clearest parts of 5.jpg (c))
    Idark_color[:, :, 1] = original_img_0_1[:, :, 1] * dark_mask  # Apply mask to green channel
    # Optionally, add some residual red/blue from other channels for more realism, if the mask isn't too strict
    # Idark_color[:, :, 0] = original_img_0_1[:, :, 0] * (dark_mask * 0.5) # Faint red
    # Idark_color[:, :, 2] = original_img_0_1[:, :, 2] * (dark_mask * 0.5) # Faint blue

    # Option 2: More nuanced, trying to retain some original dark lesion color but emphasize green vessels
    # This is harder to get exact without knowing their precise color mapping logic.
    # Let's stick with making the strong features green and other channels very dim.

    # Normalize Idark_color to [-1, 1]
    Idark_color = (Idark_color * 2) - 1

    return Idark_color, Ibri_color


# --- Example Usage ---
image_path = "old/4.png"  # Make sure this file is accessible

# 1. Preprocessing stage
preprocessed_image_array = preprocess_fundus_image(image_path)
print("Preprocessing complete. Shape:", preprocessed_image_array.shape,
      "Min/Max:", preprocessed_image_array.min(), preprocessed_image_array.max())

# 2. Image Decomposition stage with colorization approximation
idark_colorized_image, ibri_colorized_image = decompose_image_colorized_approximation(
    preprocessed_image_array,
    se_radius_small=2,  # Adjusted for potentially finer details
    se_radius_large=5,
    bright_threshold_factor=0.01,  # Lower threshold to capture more details
    dark_threshold_factor=0.01  # Lower threshold
)

print("\nColorized Image Decomposition complete.")
print("Idark (color) shape:", idark_colorized_image.shape, "Min/Max:", idark_colorized_image.min(),
      idark_colorized_image.max())
print("Ibri (color) shape:", ibri_colorized_image.shape, "Min/Max:", ibri_colorized_image.min(),
      ibri_colorized_image.max())

# --- Visualization (Optional - for checking the output) ---
# Convert back to displayable 0-255 uint8 for saving/showing.
# These are the images you'd compare to 5.jpg (b) and (c)

# For Idark visualization:
idark_display = (((idark_colorized_image + 1) / 2) * 255).astype(np.uint8)
idark_pil_image = Image.fromarray(idark_display)
idark_pil_image.save("Idark_colorized_approximation.jpg")
print("\nSaved Idark_colorized_approximation.jpg")
# idark_pil_image.show() # Uncomment to display the image

# For Ibri visualization:
ibri_display = (((ibri_colorized_image + 1) / 2) * 255).astype(np.uint8)
ibri_pil_image = Image.fromarray(ibri_display)
ibri_pil_image.save("Ibri_colorized_approximation.jpg")
print("Saved Ibri_colorized_approximation.jpg")
ibri_pil_image.show() # Uncomment to display the image