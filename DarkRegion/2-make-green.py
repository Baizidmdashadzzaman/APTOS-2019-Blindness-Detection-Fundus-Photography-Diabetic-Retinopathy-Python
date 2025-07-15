import os
import numpy as np
from PIL import Image
import cv2
from skimage.morphology import disk, opening, closing, black_tophat
import matplotlib.pyplot as plt

# --- Configuration ---
TARGET_IMAGE_SIZE = (640, 640)  # Image size for processing and display
IMAGE_PATH = "output/sharpened_retina.jpg"  # Path to the input image.


# Make sure this path is correct for your environment.

# --- Image Quality Assessment (Simplified for this context) ---
def is_good_quality(image_path):
    """
    Checks if an image is of good quality based on simple intensity thresholds.
    This is a simplified version for demonstration purposes and might need
    more robust checks for real-world applications.
    """
    try:
        img_pil = Image.open(image_path).convert("L")  # Open in grayscale for quick checks
        img_array = np.array(img_pil)

        # Simple check: avoid totally black or white images
        if np.mean(img_array) < 10 or np.mean(img_array) > 245:
            return False
        return True
    except Exception as e:
        print(f"Error checking image quality for {image_path}: {e}")
        return False


# --- Image Decomposition Functions ---
def suppress_vessels(image, kernel_length):
    """
    Suppresses vessels in the retinal image using Gabor filters.
    This helps in isolating lesions by reducing the influence of blood vessels.
    """
    if kernel_length % 2 == 0:
        kernel_length += 1
    angles = np.arange(0, 180, 15)  # Angles for Gabor filters
    vessel_response = np.zeros_like(image, dtype=np.float32)

    for angle in angles:
        # Create a Gabor kernel
        kernel = cv2.getGaborKernel((kernel_length, kernel_length), sigma=kernel_length / 4.0,
                                    theta=np.deg2rad(angle), lambd=kernel_length / 2.0,
                                    gamma=0.5, psi=0)
        kernel -= kernel.mean()  # Center the kernel
        # Apply the filter
        filtered = cv2.filter2D(image.astype(np.float32), cv2.CV_32F, kernel)
        vessel_response = np.maximum(vessel_response, filtered)  # Take maximum response across angles

    # Normalize vessel response to [0, 1]
    if vessel_response.max() > 0:
        vessel_response = (vessel_response - vessel_response.min()) / (vessel_response.max() - vessel_response.min())
    return vessel_response


def decompose_lesions(image):
    """
    Decomposes the input image into bright and dark lesion maps.
    It assumes the input `image` is already normalized to [-1, 1] and is in RGB format.
    """
    # Convert image to [0, 255] range for OpenCV operations, as it expects uint8
    image_for_cv = ((image + 1.0) * 127.5).astype(np.uint8)

    # Use the green channel, which often provides the best contrast for retinal features.
    green_channel = image_for_cv[:, :, 1]

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for contrast enhancement.
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    green_proc = clahe.apply(green_channel)

    # Suppress vessels to better isolate lesions.
    vessel_map = suppress_vessels(green_proc, 15)
    vessel_scaled = vessel_map * 0.8  # Scale down vessel contribution
    lesion_input = np.clip(green_proc * (1 - vessel_scaled), 0, 255)  # Remove vessel influence

    # --- Bright Lesion Detection (e.g., Exudates) ---
    # Morphological opening helps to remove small bright structures (like vessels)
    # while preserving larger bright lesions.
    se_bright = disk(7)  # Structuring element for bright lesions
    opened = opening(lesion_input, se_bright)
    bright_map = np.maximum(0, lesion_input - opened).astype(np.float32)  # Calculate bright lesions
    bright_map = bright_map / bright_map.max() if bright_map.max() > 0 else bright_map  # Normalize to [0, 1]
    bright_map[bright_map < 0.015] = 0  # Thresholding to remove minor noise

    # --- Dark Lesion Detection (e.g., Hemorrhages, Microaneurysms) ---
    # Black top-hat transform is effective for highlighting dark features on a bright background.
    # Using multiple radii helps detect lesions of various sizes.
    dark_maps = []
    for radius in [3, 5, 10, 15, 20]:
        se_dark = disk(radius)
        top_hat = black_tophat(lesion_input, se_dark).astype(np.float32)
        if top_hat.max() > 0:
            top_hat /= top_hat.max()  # Normalize
        dark_maps.append(top_hat)
    dark_map = np.maximum.reduce(dark_maps)  # Combine maps from different radii
    dark_map[dark_map < 0.005] = 0  # Thresholding to remove minor noise

    # Further refine dark map by closing small gaps to make lesions more cohesive.
    closed = closing((dark_map > 0).astype(np.uint8) * 255, disk(5))
    dark_map = dark_map * (closed / 255.0)

    # Normalize bright_map and dark_map to [-1, 1] range, as this is often expected by models.
    bright_map_norm = (bright_map * 2.0) - 1.0
    dark_map_norm = (dark_map * 2.0) - 1.0

    return bright_map_norm, dark_map_norm


def isolate_green_from_array(img_array):
    """
    Isolates green color components from an image array and makes
    all other parts black.
    This function expects an image array in RGB format (0-255 range).
    """
    if img_array is None:
        print("Error: Input image array for green isolation is None.")
        return None

    # Ensure the image is in the 0-255 range and uint8 type for OpenCV
    img_array_uint8 = img_array.astype(np.uint8)

    # Convert the image from RGB to HSV color space for better color segmentation
    # OpenCV's cvtColor expects BGR by default, but since our PIL load is RGB,
    # we need to be careful. If img_array came from PIL.Image.open().convert("RGB"),
    # it's RGB. If it came from cv2.imread(), it's BGR. Assuming RGB from PIL.
    hsv_img = cv2.cvtColor(img_array_uint8, cv2.COLOR_RGB2HSV)

    # --- Define color ranges for Green ---
    # These values might need slight tuning based on your specific image's green shades.
    # Lower green: [Hue, Saturation, Value]
    # Upper green: [Hue, Saturation, Value]
    lower_green = np.array([40, 50, 50])  # Adjusted for a broader range of greens
    upper_green = np.array([80, 255, 255])

    # Create a mask for green color
    mask_green = cv2.inRange(hsv_img, lower_green, upper_green)

    # Apply the mask to the original image to get only green parts on a black background
    result_img = cv2.bitwise_and(img_array_uint8, img_array_uint8, mask=mask_green)

    return result_img


# --- Main execution for decomposition and visualization ---
def run_decomposition_and_visualize():
    """
    Loads the specified image, performs decomposition, and visualizes the results.
    It includes adjustable thresholds for emphasizing specific features in the maps.
    """
    print(f"Checking image path: {IMAGE_PATH}")
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Image file not found at {IMAGE_PATH}. Please ensure the path is correct.")
        return

    print("\n--- Visualizing a sample image and its decomposition ---")

    # Perform quality assessment (optional, but good practice)
    if not is_good_quality(IMAGE_PATH):
        print(f"Warning: Image at {IMAGE_PATH} is of poor quality. Decomposition results might not be optimal.")

    # Load the original image and resize it for consistent processing
    original_display_img = Image.open(IMAGE_PATH).convert("RGB")
    original_display_img = np.array(original_display_img.resize(TARGET_IMAGE_SIZE, Image.LANCZOS))

    # Convert to float and normalize to [-1, 1] for the decomposition functions
    img_array_for_decompose = (original_display_img.astype(np.float32) / 127.5) - 1.0

    # Perform the lesion decomposition
    bright_map_decomp, dark_map_decomp = decompose_lesions(img_array_for_decompose)

    # --- Adjusting maps for visualization ---
    # Normalize decomposed maps to [0, 1] for display purposes (0=black, 1=white)
    bright_map_display = (bright_map_decomp + 1.0) / 2.0
    dark_map_display = (dark_map_decomp + 1.0) / 2.0

    # --- IMPORTANT: Adjust this threshold to control visible bright spots ---
    # A higher value (e.g., 0.7, 0.8, 0.9) will only show the *very brightest* spots.
    # A lower value (e.g., 0.1, 0.2, 0.3, 0.4) will show more bright areas, including less intense ones.
    bright_map_threshold = 0.4  # Changed from 0.5 to 0.4 to show more bright spots
    bright_map_display[bright_map_display < bright_map_threshold] = 0
    # Re-normalize the bright map after thresholding to ensure the brightest parts are still 1
    if bright_map_display.max() > 0:
        bright_map_display = bright_map_display / bright_map_display.max()

    # Apply a threshold to make non-lesion areas of the dark map very dark (close to zero opacity)
    dark_map_threshold = 0.05
    dark_map_display[dark_map_display < dark_map_threshold] = 0
    # Re-normalize the dark map after thresholding
    if dark_map_display.max() > 0:
        dark_map_display = dark_map_display / dark_map_display.max()

    # --- Generate Green Isolated Map ---
    # Use the original_display_img (0-255 RGB) for color isolation
    green_isolated_map = isolate_green_from_array(original_display_img)

    # --- Plotting the results ---
    plt.figure(figsize=(24, 6))  # Increased figure size to accommodate 4 subplots

    plt.subplot(1, 4, 1)
    plt.imshow(original_display_img)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(bright_map_display, cmap='gray')  # Use grayscale for single-channel maps
    plt.title(f'Bright Lesion Map (Threshold: {bright_map_threshold})')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(dark_map_display, cmap='gray')  # Use grayscale for single-channel maps
    plt.title('Dark Lesion Map (Low Opacity Background)')
    plt.axis('off')

    plt.subplot(1, 4, 4)  # New subplot for green isolated map
    plt.imshow(green_isolated_map)  # Display as color image
    plt.title('Green Isolated Map')
    plt.axis('off')

    plt.suptitle("Image Decomposition and Feature Isolation Results")
    plt.tight_layout()
    plt.show()

    print("--- End of decomposition visualization ---")


if __name__ == "__main__":
    run_decomposition_and_visualize()
