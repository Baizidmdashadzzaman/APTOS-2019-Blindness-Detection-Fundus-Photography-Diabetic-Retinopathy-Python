import os
from PIL import Image
import numpy as np
from skimage.morphology import disk, opening, closing
import cv2
import matplotlib.pyplot as plt # Optional, for custom plotting if needed

# --- Global Configuration Parameters (Tune these!) ---
# Experiment with different values to see how they impact the output.

# Image Processing Parameters
IMAGE_DISPLAY_SIZE = (640, 640) # Standardize image size for processing and display

# CLAHE (Contrast Limited Adaptive Histogram Equalization) Parameters
# Set APPLY_CLAHE_GREEN to True to enable this powerful pre-processing step.
APPLY_CLAHE_GREEN = True
CLAHE_CLIP_LIMIT = 2.0    # Threshold for contrast limiting (e.g., 1.0 to 4.0)
CLAHE_TILE_GRID_SIZE = (8, 8) # Size of the grid for histogram equalization (e.g., (8,8) or (16,16))

# Parameters for Dark Lesion (Red Lesion like Microaneurysms, Hemorrhages) Detection
# Consider increasing SE_RADIUS for larger lesions and lowering THRESHOLD.
DARK_LESION_SE_RADIUS = 15    # Structuring element radius for dark lesions (e.g., 5-25)
DARK_LESION_THRESHOLD = 0.005 # Normalized intensity threshold for dark lesions (e.g., 0.001-0.02)
USE_ALTERNATIVE_DARK_METHOD = False # Set to True to try the inverted green channel method

# Parameters for Bright Lesion (Exudates) Detection
BRIGHT_LESION_SE_RADIUS = 5   # Structuring element radius for bright lesions (e.g., 5-15)
BRIGHT_LESION_THRESHOLD = 0.01 # Normalized intensity threshold for bright lesions (e.g., 0.005-0.02)

# Output and Display Parameters
SAVE_OUTPUT_IMAGE = True
OUTPUT_FILENAME = "Fundus_Lesion_Analysis_Output.png"
OVERLAY_ALPHA = 0.6 # Transparency of the heatmap overlay (0.0 to 1.0)
HEATMAP_COLORMAP = cv2.COLORMAP_JET # Colormap for heatmaps (e.g., cv2.COLORMAP_JET, cv2.COLORMAP_VIRIDIS)

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
            Returns (None, None) if image loading fails.
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

    # Normalize to [-1, 1] range
    normalized_array = (img_array_float / 255.0 * 2) - 1

    return normalized_array, np.array(resized_img, dtype=np.uint8)


def decompose_image_morphological_approximation(
    pre_img_normalized: np.ndarray,
    dark_se_radius: int,
    dark_threshold_factor: float,
    use_alternative_dark_detection: bool,
    bright_se_radius: int,
    bright_threshold_factor: float,
    apply_clahe: bool,
    clahe_clip_limit: float,
    clahe_tile_grid_size: tuple
) -> (np.ndarray, np.ndarray):
    """
    Decomposes the fundus image into dark (red lesions) and bright (exudates) maps
    using morphological operations on the green channel. Optionally applies CLAHE.

    Args:
        pre_img_normalized (np.ndarray): Pre-processed image array normalized to [-1, 1].
        dark_se_radius (int): Structuring element radius for dark lesion detection.
        dark_threshold_factor (float): Threshold for dark lesion map normalization.
        use_alternative_dark_detection (bool): If True, uses an alternative method for dark spot detection.
        bright_se_radius (int): Structuring element radius for bright lesion detection.
        bright_threshold_factor (float): Threshold for bright lesion map normalization.
        apply_clahe (bool): If True, applies CLAHE to the green channel.
        clahe_clip_limit (float): CLAHE clip limit.
        clahe_tile_grid_size (tuple): CLAHE tile grid size.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Normalized map of dark regions (float32, [0, 1]).
            - np.ndarray: Normalized map of bright regions (float32, [0, 1]).
    """
    # Convert normalized image back to 0-255 range for skimage morphological operations
    img_255_scale = (((pre_img_normalized + 1) / 2) * 255).astype(np.uint8)
    green_channel = img_255_scale[:, :, 1] # Green channel provides best contrast for lesions

    # --- Optional CLAHE Pre-processing ---
    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid_size)
        green_channel_enhanced = clahe.apply(green_channel)
    else:
        green_channel_enhanced = green_channel

    # --- Bright Regions (Exudates, Cotton Wool Spots, Drusen) Detection ---
    se_bright = disk(bright_se_radius)
    opened_bright = opening(green_channel_enhanced, se_bright)
    # Exudates appear as bright structures. Opening removes small bright objects.
    # Subtracting opened image from original highlights these bright structures.
    ibri_raw = np.maximum(0, green_channel_enhanced - opened_bright)
    ibri = ibri_raw.astype(np.float32)
    if ibri.max() > 0:
        ibri /= ibri.max() # Normalize to [0, 1]
    # Apply threshold to remove noise or weak responses
    ibri[ibri < bright_threshold_factor] = 0

    # --- Dark Regions (Red Lesions like Microaneurysms, Hemorrhages) Detection ---
    idark = np.zeros_like(green_channel_enhanced, dtype=np.float32) # Initialize idark map

    if not use_alternative_dark_detection:
        # Default method: Closing fills dark spots, then subtract original
        se_dark = disk(dark_se_radius)
        closed_dark = closing(green_channel_enhanced, se_dark)
        # Red lesions appear as dark spots. Closing fills these dark spots.
        # Subtracting original from closed highlights these dark structures.
        idark_raw = np.maximum(0, closed_dark - green_channel_enhanced)
    else:
        # Alternative method: Invert green channel and apply opening
        # This seeks 'bright' spots in the inverted image, which are 'dark' in original
        se_dark = disk(dark_se_radius)
        inv_green_enhanced = 255 - green_channel_enhanced
        opened_inv_dark = opening(inv_green_enhanced, se_dark)
        idark_raw = np.maximum(0, opened_inv_dark - inv_green_enhanced)

    idark = idark_raw.astype(np.float32)
    if idark.max() > 0:
        idark /= idark.max() # Normalize to [0, 1]
    # Apply threshold to remove noise or weak responses
    idark[idark < dark_threshold_factor] = 0

    return idark, ibri


def create_heatmap_overlay(original_img_np: np.ndarray, attention_map: np.ndarray,
                           alpha: float = OVERLAY_ALPHA, colormap=HEATMAP_COLORMAP) -> np.ndarray:
    """
    Overlays a grayscale attention map as a heatmap onto the original image.

    Args:
        original_img_np (np.ndarray): Original image array (uint8, [0, 255]).
        attention_map (np.ndarray): Grayscale attention map (float32, [0, 1]).
        alpha (float): Transparency factor for the heatmap (0.0 to 1.0).
        colormap: OpenCV colormap to use.

    Returns:
        np.ndarray: Blended image with heatmap overlay (BGR format).
    """
    original_img_uint8 = original_img_np.astype(np.uint8)
    attention_map_uint8 = (attention_map * 255).astype(np.uint8)

    heatmap_bgr = cv2.applyColorMap(attention_map_uint8, colormap)
    original_bgr = cv2.cvtColor(original_img_uint8, cv2.COLOR_RGB2BGR)

    overlay_img = cv2.addWeighted(original_bgr, 1 - alpha, heatmap_bgr, alpha, 0)
    return overlay_img


def generate_analysis_figure_columns(
    image_path: str,
    img_size: tuple = IMAGE_DISPLAY_SIZE,
    dark_se_radius: int = DARK_LESION_SE_RADIUS,
    dark_threshold: float = DARK_LESION_THRESHOLD,
    use_alternative_dark: bool = USE_ALTERNATIVE_DARK_METHOD,
    bright_se_radius: int = BRIGHT_LESION_SE_RADIUS,
    bright_threshold: float = BRIGHT_LESION_THRESHOLD,
    apply_clahe: bool = APPLY_CLAHE_GREEN,
    clahe_clip_limit: float = CLAHE_CLIP_LIMIT,
    clahe_tile_grid_size: tuple = CLAHE_TILE_GRID_SIZE
) -> list:
    """
    Processes a single fundus image and generates a list of images for display
    (original, red lesion overlay, bright lesion overlay).

    Args:
        image_path (str): Path to the input image.
        img_size (tuple): Target size for image processing.
        (Other parameters are passed directly to morphological decomposition)

    Returns:
        list: A list of OpenCV BGR images: [original_display_bgr, red_lesion_overlay, bright_lesion_overlay].
              Returns None if image processing fails.
    """
    pre_img, original_resized_img_np = preprocess_fundus_image(image_path, target_size=img_size)

    if pre_img is None:
        return None

    idark, ibri = decompose_image_morphological_approximation(
        pre_img,
        dark_se_radius=dark_se_radius,
        dark_threshold_factor=dark_threshold,
        use_alternative_dark_detection=use_alternative_dark,
        bright_se_radius=bright_se_radius,
        bright_threshold_factor=bright_threshold,
        apply_clahe=apply_clahe,
        clahe_clip_limit=clahe_clip_limit,
        clahe_tile_grid_size=clahe_tile_grid_size
    )

    original_display_bgr = cv2.cvtColor(original_resized_img_np, cv2.COLOR_RGB2BGR)
    red_lesion_overlay = create_heatmap_overlay(original_resized_img_np, idark)
    bright_lesion_overlay = create_heatmap_overlay(original_resized_img_np, ibri)

    return [original_display_bgr, red_lesion_overlay, bright_lesion_overlay]


def create_color_bar(height: int, width: int, colormap=HEATMAP_COLORMAP,
                     ticks: list = None, labels: list = None) -> np.ndarray:
    """
    Creates a vertical color bar with optional ticks and labels for heatmap legend.
    """
    colorbar_img = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        intensity = int(255 * (height - 1 - i) / (height - 1))
        color_bgr = cv2.applyColorMap(np.array([[intensity]], dtype=np.uint8), colormap)[0, 0]
        colorbar_img[i, :] = color_bgr

    if ticks is not None and labels is not None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 1
        text_color = (0, 0, 0) # Black text for contrast

        for i, tick_val in enumerate(ticks):
            y_pos = int((1 - tick_val) * (height - 1))
            cv2.line(colorbar_img, (0, y_pos), (width // 4, y_pos), (255, 255, 255), 1)
            label = str(labels[i])
            text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
            text_x = width // 3 + 5
            text_y = y_pos + text_size[1] // 2
            cv2.putText(colorbar_img, label, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    return colorbar_img

# --- Main Execution Block ---
if __name__ == "__main__":
    # Define your test images here. Make sure the paths are correct!
    test_image_paths = [
        # Place your test image paths here, e.g.:
        "test_before_train/60f15dd68d30.png", # Example from previous runs
        # "path/to/your/second_fundus_image.png",
    ]

    for p in test_image_paths:
        if not os.path.isfile(p):
            print(f"Error: Image not found at '{p}'. Please update `test_image_paths` with valid file paths.")
            print("Exiting. Make sure to download an example fundus image or use your own.")
            exit(1)

    all_rows_images = []

    # --- Run Analysis with Configured Parameters ---
    print(f"\n--- Processing with Current Configuration ---")
    print(f"CLAHE Enabled: {APPLY_CLAHE_GREEN}, Clip Limit: {CLAHE_CLIP_LIMIT}, Tile Size: {CLAHE_TILE_GRID_SIZE}")
    print(f"Dark Lesion Radius: {DARK_LESION_SE_RADIUS}, Threshold: {DARK_LESION_THRESHOLD}, Alt Method: {USE_ALTERNATIVE_DARK_METHOD}")
    print(f"Bright Lesion Radius: {BRIGHT_LESION_SE_RADIUS}, Threshold: {BRIGHT_LESION_THRESHOLD}")

    for img_path in test_image_paths:
        print(f"Processing {os.path.basename(img_path)}...")
        row_images = generate_analysis_figure_columns(
            img_path,
            img_size=IMAGE_DISPLAY_SIZE,
            dark_se_radius=DARK_LESION_SE_RADIUS,
            dark_threshold=DARK_LESION_THRESHOLD,
            use_alternative_dark=USE_ALTERNATIVE_DARK_METHOD,
            bright_se_radius=BRIGHT_LESION_SE_RADIUS,
            bright_threshold=BRIGHT_LESION_THRESHOLD,
            apply_clahe=APPLY_CLAHE_GREEN,
            clahe_clip_limit=CLAHE_CLIP_LIMIT,
            clahe_tile_grid_size=CLAHE_TILE_GRID_SIZE
        )
        if row_images:
            all_rows_images.append(row_images)

    if all_rows_images:
        # Prepare for display: add titles to each column
        column_titles = ["Original Image", "Red Lesions (Dark)", "Bright Lesions (Exudates)"]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        font_thickness = 2
        text_color = (255, 255, 255) # White text for titles

        titled_rows = []
        for row_idx, row_imgs in enumerate(all_rows_images):
            titled_img_row = []
            for col_idx, img in enumerate(row_imgs):
                # Make a copy to draw on, so original image data isn't modified
                img_with_title = img.copy()
                title = column_titles[col_idx]
                text_size = cv2.getTextSize(title, font, font_scale, font_thickness)[0]
                text_x = (img_with_title.shape[1] - text_size[0]) // 2
                text_y = text_size[1] + 10 # 10 pixels padding from top
                cv2.putText(img_with_title, title, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
                titled_img_row.append(img_with_title)
            titled_rows.append(np.hstack(titled_img_row))

        final_figure_main = np.vstack(titled_rows)

        # Create and append color bar
        colorbar_height = final_figure_main.shape[0]
        colorbar_width = 80
        colorbar_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        colorbar_labels = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
        colorbar_labels.reverse()
        colorbar_ticks.reverse() # For display: 1.0 at top, 0.0 at bottom

        custom_colorbar = create_color_bar(colorbar_height, colorbar_width,
                                           colormap=HEATMAP_COLORMAP,
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

        if SAVE_OUTPUT_IMAGE:
            output_path = OUTPUT_FILENAME
            cv2.imwrite(output_path, final_figure_with_colorbar)
            print(f"Output image saved as '{output_path}'")

        print("\nDisplaying results. Press any key to close the window.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No images were successfully processed to create the figure.")