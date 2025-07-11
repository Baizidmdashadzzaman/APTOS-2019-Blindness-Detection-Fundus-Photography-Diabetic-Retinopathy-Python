from PIL import Image
import numpy as np
from skimage.morphology import disk, opening, closing
from skimage.filters import rank # Although not directly used for the decomposition itself here,
                                 # it's a useful skimage module for local filters.

def preprocess_fundus_image(image_path):
    """
    Preprocesses a fundus image by loading, resizing, and normalizing.
    This function is carried over from our previous discussion.
    """
    img = Image.open(image_path).convert("RGB")
    resized_img = img.resize((640, 640), Image.LANCZOS)
    img_array = np.array(resized_img, dtype=np.float32)
    normalized_array = (img_array / 255.0 * 2) - 1
    return normalized_array

def decompose_image_morphological_approximation(preprocessed_image_array, se_radius_small=3, se_radius_large=7, threshold_factor=0.2):
    """
    Approximates the image decomposition into dark (Idark) and bright (Ibri) regions.
    THIS IS NOT THE EXACT ALGORITHM FROM THE REFERENCED PAPER [34]
    (Morales et al., 2018), as its details are not publicly available here.

    This implementation uses common morphological operations (opening and closing)
    to highlight features and then isolates them.

    Args:
        preprocessed_image_array (np.ndarray): The preprocessed image
                                               (normalized to [-1, 1]).
                                               Expected shape (H, W, C) or (H, W).
        se_radius_small (int): Radius for the smaller structuring element.
        se_radius_large (int): Radius for the larger structuring element, used for background estimation.
        threshold_factor (float): Factor to threshold the enhanced images,
                                  to make most of the non-highlighted areas black.

    Returns:
        tuple: (Idark, Ibri) - numpy arrays representing dark and bright
               enhanced regions, with background black, normalized to [-1, 1].
    """
    # Convert back to [0, 255] uint8 for skimage morphological operations
    # and work with a single channel for simplicity, typically green or grayscale.
    img_display_scale = (((preprocessed_image_array + 1) / 2) * 255).astype(np.uint8)

    if img_display_scale.ndim == 3:
        # Assuming RGB, take the green channel. Green channel often provides good contrast
        # for both vessels and exudates in fundus images.
        gray_img = img_display_scale[:, :, 1]
    else:
        gray_img = img_display_scale # Already grayscale or single channel

    # Define structuring elements (disk shape is common for retinal analysis)
    se_small = disk(se_radius_small)
    se_large = disk(se_radius_large)

    # --- Approximation for Ibri (Bright Regions) ---
    # Ibri aims to highlight bright structures (e.g., hard exudates).
    # A common way to do this is using a "white top-hat" transform or by subtracting
    # an 'opened' version of the image from the original. Opening removes bright
    # features smaller than the structuring element.
    # By subtracting the image where small bright features are removed, we get those features.
    opened_for_bright = opening(gray_img, se_small)
    # The difference highlights bright regions
    Ibri_raw = np.maximum(0, gray_img - opened_for_bright)

    # Further process to enhance contrast and make the background truly black
    # Convert to float for manipulation
    Ibri_float = Ibri_raw.astype(np.float32)
    # Normalize to [0, 1] based on its own max
    if Ibri_float.max() > 0:
        Ibri_float /= Ibri_float.max()
    # Thresholding: values below a certain threshold are set to 0 (black)
    Ibri_float[Ibri_float < threshold_factor] = 0
    # Normalize to [-1, 1] as required for CNN input
    Ibri = (Ibri_float * 2) - 1


    # --- Approximation for Idark (Dark Regions) ---
    # Idark aims to highlight dark structures (e.g., vessels, red lesions).
    # A common way to do this is using a "black top-hat" transform or by subtracting
    # the original image from a 'closed' version. Closing fills in small dark holes.
    # The difference (closed - original) highlights these filled-in dark regions.
    closed_for_dark = closing(gray_img, se_small)
    # The difference highlights dark regions
    Idark_raw = np.maximum(0, closed_for_dark - gray_img)

    # Further process to enhance contrast and make the background truly black
    # Convert to float for manipulation
    Idark_float = Idark_raw.astype(np.float32)
    # Normalize to [0, 1] based on its own max
    if Idark_float.max() > 0:
        Idark_float /= Idark_float.max()
    # Thresholding: values below a certain threshold are set to 0 (black)
    Idark_float[Idark_float < threshold_factor] = 0
    # Normalize to [-1, 1] as required for CNN input
    Idark = (Idark_float * 2) - 1

    return Idark, Ibri

# --- Example Usage ---
# Make sure '4.jpg' is in the same directory as this script, or provide its full path.
image_path = "test_images/4.jpeg"

# 1. Preprocessing stage
preprocessed_image_array = preprocess_fundus_image(image_path)
print("Preprocessing complete. Shape:", preprocessed_image_array.shape,
      "Min/Max:", preprocessed_image_array.min(), preprocessed_image_array.max())

# 2. Image Decomposition stage (using our approximation)
idark_image, ibri_image = decompose_image_morphological_approximation(
    preprocessed_image_array, se_radius_small=3, se_radius_large=7, threshold_factor=0.05
) # Adjusted threshold_factor for potentially better visualization of features

print("\nImage Decomposition complete.")
print("Idark image shape:", idark_image.shape, "Min/Max:", idark_image.min(), idark_image.max())
print("Ibri image shape:", ibri_image.shape, "Min/Max:", ibri_image.min(), ibri_image.max())


# --- Visualization (Optional - for checking the output) ---
# To save or display these images, they need to be converted back to 0-255 uint8.
# Remember that the actual CNN input would be the normalized [-1, 1] numpy arrays.

# For Idark visualization:
idark_display = (((idark_image + 1) / 2) * 255).astype(np.uint8)
idark_pil_image = Image.fromarray(idark_display)
idark_pil_image.save("Idark_approximation.jpg")
print("\nSaved Idark_approximation.jpg")
idark_pil_image.show() # Uncomment to display the image

# For Ibri visualization:
ibri_display = (((ibri_image + 1) / 2) * 255).astype(np.uint8)
ibri_pil_image = Image.fromarray(ibri_display)
ibri_pil_image.save("Ibri_approximation.jpg")
print("Saved Ibri_approximation.jpg")
ibri_pil_image.show() # Uncomment to display the image