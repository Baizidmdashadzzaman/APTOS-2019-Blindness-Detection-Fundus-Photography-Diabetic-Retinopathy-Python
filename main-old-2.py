import os
from PIL import Image
import numpy as np
from skimage.morphology import disk, opening, closing
import cv2
import matplotlib.pyplot as plt # Optional, only if you want matplotlib for custom plots

# --- Configuration for Parameters (Tune these!) ---
# These parameters significantly affect lesion detection.
# Experiment with different values to see how they impact the output.

# Parameters for Dark Lesion (Red Lesion like Microaneurysms, Hemorrhages) Detection
DARK_LESION_SE_RADIUS = 5     # Structuring element radius for dark lesions (e.g., 5-15)
DARK_LESION_THRESHOLD = 0.01  # Normalized intensity threshold for dark lesions (e.g., 0.005-0.02)
USE_ALTERNATIVE_DARK_METHOD = False # Set to True to try the inverted green channel method

# Parameters for Bright Lesion (Exudates) Detection
BRIGHT_LESION_SE_RADIUS = 5   # Structuring element radius for bright lesions (e.g., 5-15)
BRIGHT_LESION_THRESHOLD = 0.01 # Normalized intensity threshold for bright lesions (e.g., 0.005-0.02)

IMAGE_DISPLAY_SIZE = (640, 640) # Standardize image size for processing and display

# --- Core Functions ---

def preprocess_fundus_image(image_path: str, target_size: tuple = IMAGE_DISPLAY_SIZE) -> (np.ndarray, np.ndarray):
    """
    Loads, converts, resizes, and normalizes a fundus image.

    Args:
        image_path (str): Path to the input image.
        target_size (tuple): Desired (width, height) for the resized image.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Normalized image array (float32, pixel values in [-1, 1]).
            - np.ndarray: Original resized image array (uint8, pixel values in [0, 255]).
    """
    try:
        img = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None, None
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return None, None

    resized_img = img.resize(target_size, Image.LANCZOS)
    img_array_float = np.array(resized_img, dtype=np.float32)

    # Normalize to [-1, 1] for potential future use (e.g., deep learning models expect this range)
    # Note: For morphological ops, we convert back to [0, 255] internally in the next function.
    normalized_array = (img_array_float / 255.0 * 2) - 1

    return normalized_array, np.array(resized_img, dtype=np.uint8)


def decompose_image_morphological_approximation(
    pre_img_normalized: np.ndarray,
    dark_se_radius: int = DARK_LESION_SE_RADIUS,
    dark_threshold_factor: float = DARK_LESION_THRESHOLD,
    use_alternative_dark_detection: bool = USE_ALTERNATIVE_DARK_METHOD,
    bright_se_radius: int = BRIGHT_LESION_SE_RADIUS,
    bright_threshold_factor: float = BRIGHT_LESION_THRESHOLD,
) -> (np.ndarray, np.ndarray):
    """
    Decomposes the fundus image into dark (red lesions) and bright (exudates) maps
    using morphological operations on the green channel.

    Args:
        pre_img_normalized (np.ndarray): Pre-processed image array normalized to [-1, 1].
        dark_se_radius (int): Structuring element radius for dark lesion detection.
        dark_threshold_factor (float): Threshold for dark lesion map normalization.
        use_alternative_dark_detection (bool): If True, uses an alternative method for dark spot detection.
        bright_se_radius (int): Structuring element radius for bright lesion detection.
        bright_threshold_factor (float): Threshold for bright lesion map normalization.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Normalized map of dark regions (float32, [0, 1]).
            - np.ndarray: Normalized map of bright regions (float32, [0, 1]).
    """
    # Convert normalized image back to 0-255 range for skimage morphological operations
    img_display_scale = (((pre_img_normalized + 1) / 2) * 255).astype(np.uint8)
    green_channel = img_display_scale[:, :, 1] # Green channel provides best contrast for lesions

    # --- Bright Regions (Exudates, Cotton Wool Spots, Drusen) Detection ---
    se_bright = disk(bright_se_radius)
    opened_bright = opening(green_channel, se_bright)
    # Exudates appear as bright structures. Opening removes small bright objects.
    # Subtracting opened image from original highlights these bright structures.
    ibri_raw = np.maximum(0, green_channel - opened_bright)
    ibri = ibri_raw.astype(np.float32)
    if ibri.max() > 0:
        ibri /= ibri.max() # Normalize to [0, 1]
    # Apply threshold to remove noise or weak responses
    ibri[ibri < bright_threshold_factor] = 0

    # --- Dark Regions (Red Lesions like Microaneurysms, Hemorrhages) Detection ---
    idark = np.zeros_like(green_channel, dtype=np.float32) # Initialize idark map

    if not use_alternative_dark_detection:
        # Default method: Closing fills dark spots, then subtract original
        se_dark = disk(dark_se_radius)
        closed_dark = closing(green_channel, se_dark)
        # Red lesions appear as dark spots. Closing fills these dark spots.
        # Subtracting original from closed highlights these dark structures.
        idark_raw = np.maximum(0, closed_dark - green_channel)
    else:
        # Alternative method: Invert green channel and apply opening
        # This seeks 'bright' spots in the inverted image, which are 'dark' in original
        se_dark = disk(dark_se_radius) # Use the dark lesion specific radius
        inv_green = 255 - green_channel
        opened_inv_dark = opening(inv_green, se_dark)
        idark_raw = np.maximum(0, opened_inv_dark - inv_green)

    idark = idark_raw.astype(np.float32)
    if idark.max() > 0:
        idark /= idark.max() # Normalize to [0, 1]
    # Apply threshold to remove noise or weak responses
    idark[idark < dark_threshold_factor] = 0

    return idark, ibri


def create_heatmap_overlay(original_img_np: np.ndarray, attention_map: np.ndarray,
                           alpha: float = 0.6, colormap=cv2.COLORMAP_JET) -> np.ndarray:
    """
    Overlays a grayscale attention map as a heatmap onto the original image.

    Args:
        original_img_np (np.ndarray): Original image array (uint8, [0, 255]).
        attention_map (np.ndarray): Grayscale attention map (float32, [0, 1]).
        alpha (float): Transparency factor for the heatmap (0.0 to 1.0).
        colormap: OpenCV colormap to use (e.g., cv2.COLORMAP_JET).

    Returns:
        np.ndarray: Blended image with heatmap overlay (BGR format).
    """
    # Ensure original image is uint8 for OpenCV operations
    original_img_uint8 = original_img_np.astype(np.uint8)
    # Convert attention map to 0-255 uint8 for colormap application
    attention_map_uint8 = (attention_map * 255).astype(np.uint8)

    # Apply colormap to the attention map
    heatmap_bgr = cv2.applyColorMap(attention_map_uint8, colormap)

    # Convert original image to BGR as OpenCV uses BGR by default
    original_bgr = cv2.cvtColor(original_img_uint8, cv2.COLOR_RGB2BGR)

    # Blend the original image and the heatmap
    overlay_img = cv2.addWeighted(original_bgr, 1 - alpha, heatmap_bgr, alpha, 0)
    return overlay_img


def generate_analysis_figure_columns(
    image_path: str,
    dark_se_radius: int,
    dark_threshold: float,
    use_alternative_dark: bool,
    bright_se_radius: int,
    bright_threshold: float,
    img_size: tuple = IMAGE_DISPLAY_SIZE,
) -> list:
    """
    Processes a single fundus image and generates a list of images for display
    (original, red lesion overlay, bright lesion overlay).

    Args:
        image_path (str): Path to the input image.
        dark_se_radius (int): Structuring element radius for dark lesions.
        dark_threshold (float): Threshold for dark lesion map.
        use_alternative_dark (bool): Flag for alternative dark detection method.
        bright_se_radius (int): Structuring element radius for bright lesions.
        bright_threshold (float): Threshold for bright lesion map.
        img_size (tuple): Target size for image processing.

    Returns:
        list: A list of OpenCV BGR images: [original_display_bgr, red_lesion_overlay, bright_lesion_overlay].
              Returns None if image processing fails.
    """
    pre_img, original_resized_img_np = preprocess_fundus_image(image_path, target_size=img_size)

    if pre_img is None: # Handle cases where preprocess_fundus_image failed
        return None

    idark, ibri = decompose_image_morphological_approximation(
        pre_img,
        dark_se_radius=dark_se_radius,
        dark_threshold_factor=dark_threshold,
        use_alternative_dark_detection=use_alternative_dark,
        bright_se_radius=bright_se_radius,
        bright_threshold_factor=bright_threshold,
    )

    original_display_bgr = cv2.cvtColor(original_resized_img_np, cv2.COLOR_RGB2BGR)
    red_lesion_overlay = create_heatmap_overlay(original_resized_img_np, idark)
    bright_lesion_overlay = create_heatmap_overlay(original_resized_img_np, ibri)

    return [original_display_bgr, red_lesion_overlay, bright_lesion_overlay]


def create_color_bar(height: int, width: int, colormap=cv2.COLORMAP_JET,
                     ticks: list = None, labels: list = None) -> np.ndarray:
    """
    Creates a vertical color bar with optional ticks and labels for heatmap legend.

    Args:
        height (int): Height of the color bar image.
        width (int): Width of the color bar image.
        colormap: OpenCV colormap.
        ticks (list): List of normalized tick positions (0.0 to 1.0).
        labels (list): List of string labels for ticks.

    Returns:
        np.ndarray: The color bar image (BGR format).
    """
    colorbar_img = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        # Map vertical position to intensity for colormap (inverted so 1.0 is top)
        intensity = int(255 * (height - 1 - i) / (height - 1))
        # Apply colormap to a single pixel and get its BGR color
        color_bgr = cv2.applyColorMap(np.array([[intensity]], dtype=np.uint8), colormap)[0, 0]
        colorbar_img[i, :] = color_bgr

    if ticks is not None and labels is not None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 1
        text_color = (0, 0, 0) # Black text for contrast

        for i, tick_val in enumerate(ticks):
            # Calculate y position based on normalized tick value
            y_pos = int((1 - tick_val) * (height - 1))
            # Draw a small white line for the tick
            cv2.line(colorbar_img, (0, y_pos), (width // 4, y_pos), (255, 255, 255), 1)
            label = str(labels[i])
            # Get text size to position it
            text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
            text_x = width // 3 + 5
            text_y = y_pos + text_size[1] // 2
            # Draw the label
            cv2.putText(colorbar_img, label, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    return colorbar_img

# --- Main Execution Block ---
if __name__ == "__main__":
    # Define your test images here. Make sure the paths are correct!
    test_image_paths = [
        # Example 1: Original test image
        "test_before_train/60f15dd68d30.png",
        # Example 2: Add another image if you have one
        # "path/to/your/second_fundus_image.png",
    ]

    for p in test_image_paths:
        if not os.path.isfile(p):
            print(f"Error: Image not found at '{p}'. Please update `test_image_paths` with valid file paths.")
            print("Exiting. Make sure to download an example fundus image or use your own.")
            exit(1) # Exit if essential images are missing

    all_rows_images = []

    # --- Experiment with Parameters Here ---
    # To demonstrate flexibility, you can modify these or create loops to test combinations.
    # For the problem of the large red lesion, focus on DARK_LESION_SE_RADIUS
    # and DARK_LESION_THRESHOLD.
    current_dark_se_radius = 15   # Increased radius to capture larger features
    current_dark_threshold = 0.005 # Lowered threshold to include weaker signals
    current_use_alternative_dark = False # Try True here as well

    print(f"\n--- Processing with Parameters ---")
    print(f"Dark Lesion Radius: {current_dark_se_radius}, Threshold: {current_dark_threshold}, Alt Method: {current_use_alternative_dark}")
    print(f"Bright Lesion Radius: {BRIGHT_LESION_SE_RADIUS}, Threshold: {BRIGHT_LESION_THRESHOLD}")

    for img_path in test_image_paths:
        print(f"Processing {os.path.basename(img_path)}...")
        row_images = generate_analysis_figure_columns(
            img_path,
            dark_se_radius=current_dark_se_radius,
            dark_threshold=current_dark_threshold,
            use_alternative_dark=current_use_alternative_dark,
            bright_se_radius=BRIGHT_LESION_SE_RADIUS, # Using global config for bright
            bright_threshold=BRIGHT_LESION_THRESHOLD, # Using global config for bright
            img_size=IMAGE_DISPLAY_SIZE,
        )
        if row_images: # Only add if processing was successful
            all_rows_images.append(row_images)

    if all_rows_images:
        # Stack images horizontally for each row, then vertically for multiple rows
        combined_rows = [np.hstack(row) for row in all_rows_images]
        final_figure_main = np.vstack(combined_rows)

        # Create and append color bar
        colorbar_height = final_figure_main.shape[0]
        colorbar_width = 80
        # Ticks and labels for the color bar (0.0 to 1.0, 1.0 is highest intensity)
        colorbar_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        colorbar_labels = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"] # Match labels
        colorbar_labels.reverse() # For display: 1.0 at top, 0.0 at bottom
        colorbar_ticks.reverse() # For display: 1.0 at top, 0.0 at bottom

        custom_colorbar = create_color_bar(colorbar_height, colorbar_width,
                                           colormap=cv2.COLORMAP_JET,
                                           ticks=colorbar_ticks,
                                           labels=colorbar_labels)

        final_figure_with_colorbar = np.hstack((final_figure_main, custom_colorbar))

        # Display the result
        window_name = "Fundus Image Lesion Analysis"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name,
                         final_figure_with_colorbar.shape[1],
                         final_figure_with_colorbar.shape[0])
        cv2.imshow(window_name, final_figure_with_colorbar)

        print("\nDisplaying results. Press any key to close the window.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No images were successfully processed to create the figure.")