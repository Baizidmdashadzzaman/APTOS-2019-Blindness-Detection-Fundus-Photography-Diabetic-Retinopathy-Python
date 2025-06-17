import os
from PIL import Image
import numpy as np
from skimage.morphology import disk, opening, closing, black_tophat  # Added black_tophat for clarity
import cv2

# --- Global Configuration Parameters (Tune these!) ---
# Experiment with different values to see how they impact the output.

# Image Processing & Display Parameters
IMAGE_DISPLAY_SIZE = (640, 640)  # Standardize image size for processing and display
OVERLAY_ALPHA = 0.6  # Transparency of the heatmap overlay (0.0 to 1.0)
HEATMAP_COLORMAP = cv2.COLORMAP_VIRIDIS  # Recommended for better perception (e.g., cv2.COLORMAP_JET, cv2.COLORMAP_MAGMA)

# CLAHE (Contrast Limited Adaptive Histogram Equalization) Parameters
# Set APPLY_CLAHE_GREEN to True to enable this powerful pre-processing step.
APPLY_CLAHE_GREEN = True
CLAHE_CLIP_LIMIT = 2.5  # Threshold for contrast limiting (e.g., 1.0 to 4.0)
CLAHE_TILE_GRID_SIZE = (8, 8)  # Size of the grid for histogram equalization (e.g., (8,8) or (16,16))

# Dark Lesion (Red Lesion like Microaneurysms, Hemorrhages) Detection Parameters
# DARK_LESION_SE_RADII: List of structuring element radii for multi-scale detection.
# Larger radii help detect broader, more diffuse lesions.
DARK_LESION_SE_RADII = [5, 15, 25]  # Try adding more or larger values, e.g., [5, 10, 20, 30]
DARK_LESION_THRESHOLD = 0.003  # Normalized intensity threshold for dark lesions (e.g., 0.001-0.02). Lower for more sensitivity.
USE_ALTERNATIVE_DARK_METHOD = False  # Set to True to try the inverted green channel method for all scales

# Bright Lesion (Exudates) Detection Parameters
BRIGHT_LESION_SE_RADIUS = 5  # Structuring element radius for bright lesions (e.g., 5-15)
BRIGHT_LESION_THRESHOLD = 0.01  # Normalized intensity threshold for bright lesions (e.g., 0.005-0.02)

# Output Options
SAVE_OUTPUT_IMAGE = True
OUTPUT_FILENAME = "Fundus_Lesion_Analysis_Output.png"


# --- Core Functions ---

def preprocess_fundus_image(image_path: str, target_size: tuple = IMAGE_DISPLAY_SIZE) -> (np.ndarray, np.ndarray):
    """
    Loads, converts, resizes, and normalizes a fundus image.
    Returns normalized image (-1 to 1) and original resized image (0-255 uint8).
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
    normalized_array = (img_array_float / 255.0 * 2) - 1  # Normalize to [-1, 1]

    return normalized_array, np.array(resized_img, dtype=np.uint8)


def decompose_image_morphological_approximation(
        pre_img_normalized: np.ndarray,
        dark_se_radii: list,  # Now expects a list of radii
        dark_threshold_factor: float,
        use_alternative_dark_detection: bool,
        bright_se_radius: int,
        bright_threshold_factor: float,
        apply_clahe: bool,
        clahe_clip_limit: float,
        clahe_tile_grid_size: tuple
) -> (np.ndarray, np.ndarray):
    """
    Decomposes the fundus image into multi-scale dark (red lesions) and bright (exudates) maps
    using morphological operations on the green channel. Optionally applies CLAHE.
    """
    img_255_scale = (((pre_img_normalized + 1) / 2) * 255).astype(np.uint8)
    green_channel = img_255_scale[:, :, 1]  # Green channel provides best contrast for lesions

    # --- Optional CLAHE Pre-processing ---
    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid_size)
        green_channel_processed = clahe.apply(green_channel)
    else:
        green_channel_processed = green_channel

    # --- Bright Regions (Exudates, Cotton Wool Spots, Drusen) Detection ---
    se_bright = disk(bright_se_radius)
    opened_bright = opening(green_channel_processed, se_bright)
    ibri_raw = np.maximum(0, green_channel_processed - opened_bright)
    ibri = ibri_raw.astype(np.float32)
    if ibri.max() > 0:
        ibri /= ibri.max()
    ibri[ibri < bright_threshold_factor] = 0

    # --- Dark Regions (Red Lesions like Microaneurysms, Hemorrhages) Detection (Multi-scale) ---
    all_scale_idark_maps = []
    for current_se_radius in dark_se_radii:
        if current_se_radius <= 0: continue  # Skip invalid radii

        se_dark = disk(current_se_radius)

        if not use_alternative_dark_detection:
            # Default method (morphological black top-hat): closing fills dark spots, then subtract original
            # This is effective for distinct dark features on a brighter background.
            # black_tophat function from skimage directly computes closing(image, se) - image
            current_idark_raw = black_tophat(green_channel_processed, se_dark)
        else:
            # Alternative method: Invert green channel and apply opening
            # This seeks 'bright' spots in the inverted image, which are 'dark' in original
            inv_green_processed = 255 - green_channel_processed
            opened_inv_dark = opening(inv_green_processed, se_dark)
            current_idark_raw = np.maximum(0, opened_inv_dark - inv_green_processed)

        # Normalize and add to list (thresholding will be applied after combining all scales)
        temp_idark = current_idark_raw.astype(np.float32)
        if temp_idark.max() > 0:
            temp_idark /= temp_idark.max()
        all_scale_idark_maps.append(temp_idark)

    # Combine all multi-scale dark maps by taking the maximum value at each pixel
    if all_scale_idark_maps:
        idark = np.maximum.reduce(all_scale_idark_maps)
    else:
        idark = np.zeros_like(green_channel_processed, dtype=np.float32)  # Fallback if no radii provided

    # Apply global threshold after combining all scales
    idark[idark < dark_threshold_factor] = 0

    return idark, ibri


def create_heatmap_overlay(original_img_np: np.ndarray, attention_map: np.ndarray,
                           alpha: float = OVERLAY_ALPHA, colormap=HEATMAP_COLORMAP) -> np.ndarray:
    """
    Overlays a grayscale attention map as a heatmap onto the original image.
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
        dark_se_radii: list = DARK_LESION_SE_RADII,  # Now a list
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
    Returns None if image processing fails.
    """
    pre_img, original_resized_img_np = preprocess_fundus_image(image_path, target_size=img_size)

    if pre_img is None:
        return None

    idark, ibri = decompose_image_morphological_approximation(
        pre_img,
        dark_se_radii=dark_se_radii,  # Pass list of radii
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
        text_color = (0, 0, 0)  # Black text for contrast

        for i, tick_val in enumerate(ticks):
            y_pos = int((1 - tick_val) * (height - 1))
            cv2.line(colorbar_img, (0, y_pos), (width // 4, y_pos), (255, 255, 255), 1)
            label = str(labels[i])
            text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
            text_x = width // 3 + 5
            text_y = y_pos + text_size[1] // 2
            cv2.putText(colorbar_img, label, (text_x, text_y), font, font_scale, text_color, font_thickness,
                        cv2.LINE_AA)

    return colorbar_img


# --- Main Execution Block ---
if __name__ == "__main__":
    # Define your test images here. Make sure the paths are correct!
    test_image_paths = [
        # Use the image that had the large unhighlighted lesion:
        "test_before_train/60f15dd68d30.png",  # Example from previous runs
        # "path/to/your/another_challenging_image.png", # Add more if you have them
    ]

    for p in test_image_paths:
        if not os.path.isfile(p):
            print(f"Error: Image not found at '{p}'. Please update `test_image_paths` with valid file paths.")
            print("Exiting. Make sure to download an example fundus image or use your own.")
            exit(1)

    all_rows_images = []

    print(f"\n--- Processing with Current Configuration ---")
    print(f"CLAHE Enabled: {APPLY_CLAHE_GREEN}, Clip Limit: {CLAHE_CLIP_LIMIT}, Tile Size: {CLAHE_TILE_GRID_SIZE}")
    print(
        f"Dark Lesion Radii (Multi-scale): {DARK_LESION_SE_RADII}, Threshold: {DARK_LESION_THRESHOLD}, Alt Method: {USE_ALTERNATIVE_DARK_METHOD}")
    print(f"Bright Lesion Radius: {BRIGHT_LESION_SE_RADIUS}, Threshold: {BRIGHT_LESION_THRESHOLD}")

    for img_path in test_image_paths:
        print(f"Processing {os.path.basename(img_path)}...")
        row_images = generate_analysis_figure_columns(
            img_path,
            img_size=IMAGE_DISPLAY_SIZE,
            dark_se_radii=DARK_LESION_SE_RADII,  # Pass list
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
        text_color = (255, 255, 255)  # White text for titles

        titled_rows = []
        for row_idx, row_imgs in enumerate(all_rows_images):
            titled_img_row = []
            for col_idx, img in enumerate(row_imgs):
                img_with_title = img.copy()
                title = column_titles[col_idx]
                text_size = cv2.getTextSize(title, font, font_scale, font_thickness)[0]
                text_x = (img_with_title.shape[1] - text_size[0]) // 2
                text_y = text_size[1] + 10
                cv2.putText(img_with_title, title, (text_x, text_y), font, font_scale, text_color, font_thickness,
                            cv2.LINE_AA)
                titled_img_row.append(img_with_title)
            titled_rows.append(np.hstack(titled_img_row))

        final_figure_main = np.vstack(titled_rows)

        # Create and append color bar
        colorbar_height = final_figure_main.shape[0]
        colorbar_width = 80
        colorbar_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        colorbar_labels = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
        colorbar_labels.reverse()
        colorbar_ticks.reverse()

        custom_colorbar = create_color_bar(colorbar_height, colorbar_width,
                                           colormap=HEATMAP_COLORMAP,
                                           ticks=colorbar_ticks,
                                           labels=colorbar_labels)

        final_figure_with_colorbar = np.hstack((final_figure_main, custom_colorbar))

        # Display the result
        window_name = "Fundus Image Lesion Analysis (Multi-scale)"
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