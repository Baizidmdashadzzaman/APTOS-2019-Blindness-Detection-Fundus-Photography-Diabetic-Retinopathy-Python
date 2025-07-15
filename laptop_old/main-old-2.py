import os
from PIL import Image
import numpy as np
from skimage.morphology import disk, opening, closing, black_tophat
import cv2
import matplotlib.pyplot as plt

# --- Global Configuration Parameters (Tune these for best results!) ---
# These parameters significantly influence lesion detection. Experiment with different values.

# Image Preprocessing & Output Display Parameters
TARGET_IMAGE_SIZE = (640, 640)  # Standardize image size for processing (paper uses 640x640)
OVERLAY_TRANSPARENCY = 0.6  # Transparency of the heatmap overlay (0.0 to 1.0, 0.6 is good)
HEATMAP_COLORMAP = cv2.COLORMAP_VIRIDIS  # Colormap for heatmap (e.g., cv2.COLORMAP_JET, cv2.COLORMAP_MAGMA, cv2.COLORMAP_PARULA)

# CLAHE (Contrast Limited Adaptive Histogram Equalization) Parameters
# Applied to the green channel to enhance local contrast, as fundus images often have uneven illumination.
APPLY_CLAHE_PREPROCESSING = True
CLAHE_CLIP_LIMIT = 2.5  # Controls contrast limiting. Higher values enhance more, but can amplify noise. (e.g., 1.0 - 4.0)
CLAHE_TILE_GRID_SIZE = (
8, 8)  # Size of the grid for histogram equalization. Smaller for fine details, larger for broader areas.

# Dark Lesion (e.g., Microaneurysms, Hemorrhages) Detection Parameters
# These lesions appear darker than the surrounding retina.
# Multi-scale detection uses different structuring element sizes to capture varied lesion sizes.
DARK_LESION_SE_RADII = [3, 5, 10, 15,
                        20]  # Radii of structuring elements (disk shape). Adjust based on expected lesion sizes.
DARK_LESION_THRESHOLD = 0.005  # Normalized intensity threshold. Pixels below this are discarded. Higher = less sensitive (less noise).
POST_PROCESS_DARK_CLOSING_RADIUS = 5  # Morphological closing to fill small holes in detected dark lesions. 0 to disable.

# Bright Lesion (e.g., Exudates, Cotton Wool Spots) Detection Parameters
# These lesions appear brighter than the surrounding retina.
BRIGHT_LESION_SE_RADIUS = 7  # Radius of structuring element (disk shape).
BRIGHT_LESION_THRESHOLD = 0.015  # Normalized intensity threshold. Pixels below this are discarded. Higher = less sensitive (less noise).

# Output File Settings
SAVE_OUTPUT_TO_FILE = True  # Set to True to save the generated plots as PNG images
OUTPUT_FILENAME_PREFIX = "Fundus_Lesion_Decomposition_Output_"  # Prefix for output image filenames


# --- Core Image Processing Functions ---

def load_and_preprocess_image(image_path: str, target_size: tuple) -> (np.ndarray, np.ndarray):
    """
    Loads an image, converts it to RGB, resizes, and normalizes pixel values to [-1, 1].
    Returns the normalized float array and the original resized 0-255 uint8 array.
    """
    print(f"  Loading and preprocessing: {os.path.basename(image_path)}...")
    try:
        # Open image using PIL for broader format support and consistent resizing
        img_pil = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"ERROR: Image file not found at '{image_path}'. Skipping.")
        return None, None
    except Exception as e:
        print(f"ERROR: Could not open/process image '{image_path}': {e}. Skipping.")
        return None, None

    # Resize the image
    resized_img_pil = img_pil.resize(target_size, Image.LANCZOS)
    original_resized_np_uint8 = np.array(resized_img_pil, dtype=np.uint8)

    # Convert to float and normalize to [-1, 1] as described in the paper (Section 3.3)
    img_array_float = np.array(resized_img_pil, dtype=np.float32)
    normalized_array = (img_array_float / 255.0 * 2) - 1
    print(f"  Image resized to {target_size} and normalized to [-1, 1].")
    return normalized_array, original_resized_np_uint8


def decompose_and_highlight_lesions(
        normalized_image: np.ndarray,
        apply_clahe: bool, clahe_clip_limit: float, clahe_tile_grid_size: tuple,
        dark_se_radii: list, dark_threshold: float, post_process_dark_closing_radius: int,
        bright_se_radius: int, bright_threshold: float
) -> (np.ndarray, np.ndarray):
    """
    Implements the image decomposition described in Section 3.4 of the paper.
    It extracts the green channel, optionally applies CLAHE, and uses morphological
    operations to highlight dark (red) and bright (exudate) lesions.
    """
    # Convert normalized image [-1, 1] back to [0, 255] for OpenCV/skimage processing
    image_255_scale = (((normalized_image + 1) / 2) * 255).astype(np.uint8)

    # The green channel typically offers the best contrast for fundus lesions
    green_channel = image_255_scale[:, :, 1]
    print(f"  Green channel extracted for processing. Shape: {green_channel.shape}")

    # --- Apply CLAHE Pre-processing (Optional, based on global param) ---
    if apply_clahe:
        clahe_enhancer = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid_size)
        processed_green_channel = clahe_enhancer.apply(green_channel)
        print(f"  CLAHE applied to green channel (Clip: {clahe_clip_limit}, Tile: {clahe_tile_grid_size}).")
    else:
        processed_green_channel = green_channel
        print("  CLAHE preprocessing skipped.")

    # --- 1. Detect Bright Lesions (Exudates, Cotton Wool Spots) ---
    # These are typically brighter than the background.
    # Use morphological opening (erosion followed by dilation) to estimate the background,
    # then subtract it from the image to highlight bright features (similar to white top-hat).
    print(f"  Detecting bright lesions (SE Radius: {bright_se_radius})...")
    se_bright = disk(bright_se_radius)
    # Ensure input to opening is float for consistency in subtraction before normalizing
    opened_bright = opening(processed_green_channel.astype(np.float32), se_bright)
    bright_lesion_map_raw = np.maximum(0, processed_green_channel.astype(np.float32) - opened_bright)

    # Normalize the bright lesion map to [0, 1]
    bright_lesion_map = bright_lesion_map_raw.astype(np.float32)
    if bright_lesion_map.max() > 0:
        bright_lesion_map /= bright_lesion_map.max()
        print(f"  Bright lesion map normalized. Raw max: {bright_lesion_map_raw.max()}.")
    else:
        print("  WARNING: No significant bright features detected before thresholding.")

    # Apply global threshold to remove weak responses/noise
    bright_lesion_map[bright_lesion_map < bright_threshold] = 0
    print(
        f"  Bright lesion map thresholded (Threshold: {bright_threshold}). Max after threshold: {bright_lesion_map.max()}")

    # --- 2. Detect Dark Lesions (Microaneurysms, Hemorrhages) ---
    # These are typically darker than the background.
    # Use multi-scale black top-hat (closing followed by subtraction) to highlight dark features.
    print(f"  Detecting dark lesions (Multi-scale SE Radii: {dark_se_radii})...")
    all_scale_dark_maps = []
    for current_radius in dark_se_radii:
        if current_radius <= 0: continue  # Skip invalid radii
        se_dark = disk(current_radius)
        # Black top-hat highlights dark objects smaller than the structuring element on a bright background.
        current_dark_map_raw = black_tophat(processed_green_channel.astype(np.float32), se_dark)

        temp_dark_map = current_dark_map_raw.astype(np.float32)
        if temp_dark_map.max() > 0:
            temp_dark_map /= temp_dark_map.max()  # Normalize each scale
        all_scale_dark_maps.append(temp_dark_map)
        print(f"    - Scale {current_radius} processed. Temp dark map max: {temp_dark_map.max()}")

    # Combine all multi-scale dark maps by taking the maximum at each pixel
    dark_lesion_map = np.zeros_like(processed_green_channel, dtype=np.float32)
    if all_scale_dark_maps:
        dark_lesion_map = np.maximum.reduce(all_scale_dark_maps)
        print(f"  All dark scale maps combined. Combined dark map max: {dark_lesion_map.max()}")
    else:
        print("  WARNING: No dark scale maps generated.")

    # Apply global threshold after combining all scales
    dark_lesion_map[dark_lesion_map < dark_threshold] = 0
    print(f"  Dark lesion map thresholded (Threshold: {dark_threshold}). Max after threshold: {dark_lesion_map.max()}")

    # --- Post-processing: Morphological Closing on the Dark Lesion Map (Optional) ---
    # Helps fill small gaps and connect nearby dark lesion fragments.
    if post_process_dark_closing_radius > 0:
        print(f"  Applying post-processing (closing radius: {post_process_dark_closing_radius}) to dark map...")
        # Convert to binary for morphological operation
        binary_dark_map = (dark_lesion_map > 0).astype(np.uint8) * 255
        se_post_process = disk(post_process_dark_closing_radius)
        closed_binary_dark_map = closing(binary_dark_map, se_post_process)

        # Apply the closed mask back to the original intensity map
        dark_lesion_map_filled = dark_lesion_map * (closed_binary_dark_map / 255.0)
        dark_lesion_map = dark_lesion_map_filled
        print(f"  Dark lesion map post-processed. Max after post-processing: {dark_lesion_map.max()}")
    else:
        print("  No post-processing applied to dark map.")

    return dark_lesion_map, bright_lesion_map


def create_heatmap_overlay(original_img_uint8: np.ndarray, attention_map: np.ndarray,
                           alpha: float, colormap) -> np.ndarray:
    """
    Overlays a grayscale attention map (0-1 float) as a heatmap onto the original RGB image.
    """
    # Convert attention map to 0-255 uint8 for colormapping
    attention_map_uint8 = (attention_map * 255).astype(np.uint8)

    # Apply colormap to the attention map
    heatmap_bgr = cv2.applyColorMap(attention_map_uint8, colormap)

    # Convert original image to BGR for overlay operation with OpenCV
    original_bgr = cv2.cvtColor(original_img_uint8, cv2.COLOR_RGB2BGR)

    # Blend the original image and the heatmap
    overlay_img = cv2.addWeighted(original_bgr, 1 - alpha, heatmap_bgr, alpha, 0)
    return overlay_img


def create_dummy_fundus_image(filename: str, has_dark_lesions: bool, has_bright_lesions: bool):
    """
    Creates a synthetic fundus-like image with optional dark/bright lesions for testing.
    """
    print(f"Creating dummy image: {filename}")
    img_size = (640, 640)

    # Create a base image with a gradient to simulate fundus background
    base_img = np.zeros(img_size + (3,), dtype=np.uint8)
    center_x, center_y = img_size[0] // 2, img_size[1] // 2

    for y in range(img_size[1]):
        for x in range(img_size[0]):
            # Simulate a circular fundus view with color variation
            dist_to_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            normalized_dist = dist_to_center / (min(img_size) / 2)

            r_val = int(max(0, 150 - normalized_dist * 80))
            g_val = int(max(0, 100 - normalized_dist * 50))
            b_val = int(max(0, 50 - normalized_dist * 20))

            # Simulate optic disc area
            if 0.1 < normalized_dist < 0.2 and abs(x - center_x) < 50 and abs(y - center_y + 100) < 50:
                r_val = int(max(0, r_val + 50))
                g_val = int(max(0, g_val + 30))
                b_val = int(max(0, b_val + 10))

            base_img[y, x] = [r_val, g_val, b_val]

    # Add a main vessel network (simplified)
    cv2.line(base_img, (center_x - 100, center_y - 100), (center_x + 100, center_y + 100), (10, 10, 10), 3)
    cv2.line(base_img, (center_x + 100, center_y - 100), (center_x - 100, center_y + 100), (10, 10, 10), 3)
    cv2.line(base_img, (center_x, center_y - 150), (center_x, center_y + 150), (15, 15, 15), 2)

    # Add lesions based on flags
    if has_dark_lesions:
        # Microaneurysms
        cv2.circle(base_img, (center_x - 50, center_y + 30), 3, (0, 0, 0), -1)
        cv2.circle(base_img, (center_x + 60, center_y - 40), 4, (10, 0, 0), -1)
        # Hemorrhage
        cv2.ellipse(base_img, (center_x - 80, center_y - 80), (15, 25), 45, 0, 360, (20, 0, 0), -1)
        cv2.circle(base_img, (center_x + 150, center_y + 10), 8, (15, 0, 0), -1)  # Another larger dark spot

    if has_bright_lesions:
        # Hard Exudates (small, bright, yellowish spots)
        cv2.circle(base_img, (center_x + 50, center_y + 20), 5, (255, 255, 150), -1)
        cv2.circle(base_img, (center_x + 70, center_y + 35), 4, (255, 255, 120), -1)
        cv2.rectangle(base_img, (center_x - 10, center_y + 80), (center_x + 20, center_y + 95), (255, 255, 100), -1)
        # Cotton Wool Spot (larger, fuzzy bright area)
        cv2.ellipse(base_img, (center_x - 100, center_y + 120), (20, 10), 30, 0, 360, (250, 250, 250), -1)
        cv2.circle(base_img, (center_x - 150, center_y - 50), 10, (255, 255, 200), -1)  # Another large bright spot

    # Add subtle noise to make it more realistic
    noise = np.random.normal(0, 5, base_img.shape).astype(np.int16)
    noisy_img = np.clip(base_img + noise, 0, 255).astype(np.uint8)

    cv2.imwrite(filename, noisy_img)
    print(f"  Dummy image '{filename}' created successfully.")


# --- Main Execution Block ---
if __name__ == "__main__":
    # Ensure a directory for output exists
    output_dir = "output_lesion_analysis"
    os.makedirs(output_dir, exist_ok=True)

    # --- Dummy Image Creation (for easy testing without real fundus images) ---
    # These images will be created if they don't exist in the current directory.
    dummy_img_path_1 = "dummy_fundus_1_lesions.png"
    dummy_img_path_2 = "dummy_fundus_2_no_lesions.png"
    dummy_img_path_3 = "dummy_fundus_3_bright_only.png"
    dummy_img_path_4 = "dummy_fundus_4_dark_only.png"

    if not os.path.exists(dummy_img_path_1):
        create_dummy_fundus_image(dummy_img_path_1, has_dark_lesions=True, has_bright_lesions=True)
    if not os.path.exists(dummy_img_path_2):
        create_dummy_fundus_image(dummy_img_path_2, has_dark_lesions=False, has_bright_lesions=False)
    if not os.path.exists(dummy_img_path_3):
        create_dummy_fundus_image(dummy_img_path_3, has_dark_lesions=False, has_bright_lesions=True)
    if not os.path.exists(dummy_img_path_4):
        create_dummy_fundus_image(dummy_img_path_4, has_dark_lesions=True, has_bright_lesions=False)

    # --- Define Images to Process ---
    # Add your actual fundus image paths here.
    # The dummy images are included by default for immediate testing.
    images_to_process = [
        "test_images/5.jpeg",  # Uncomment and replace with your image paths
        "test_images/6.jpeg",
        "test_images/7.jpeg",
    ]

    # Filter out paths that do not exist
    valid_images = [img_path for img_path in images_to_process if os.path.exists(img_path)]

    if not valid_images:
        print("\nNo valid images found to process. Please ensure image paths are correct.")
        print("Dummy images have been created for testing if you don't have real ones.")
        exit(1)

    print(f"\n--- Starting Lesion Analysis with Current Configuration ---")
    print(
        f"  CLAHE Preprocessing: {'Enabled' if APPLY_CLAHE_PREPROCESSING else 'Disabled'} (Clip: {CLAHE_CLIP_LIMIT}, Tile: {CLAHE_TILE_GRID_SIZE})")
    print(
        f"  Dark Lesion Params: Radii={DARK_LESION_SE_RADII}, Threshold={DARK_LESION_THRESHOLD}, Post-Closing={POST_PROCESS_DARK_CLOSING_RADIUS}")
    print(f"  Bright Lesion Params: Radius={BRIGHT_LESION_SE_RADIUS}, Threshold={BRIGHT_LESION_THRESHOLD}")
    print(f"  Output images will be saved in '{output_dir}' with prefix '{OUTPUT_FILENAME_PREFIX}'.")

    for img_file_path in valid_images:
        print(f"\nProcessing image: {os.path.basename(img_file_path)}")
        normalized_img, original_resized_img_uint8 = load_and_preprocess_image(img_file_path, TARGET_IMAGE_SIZE)

        if normalized_img is None:
            continue  # Skip to next image if loading failed

        dark_lesion_map, bright_lesion_map = decompose_and_highlight_lesions(
            normalized_img,
            APPLY_CLAHE_PREPROCESSING, CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID_SIZE,
            DARK_LESION_SE_RADII, DARK_LESION_THRESHOLD, POST_PROCESS_DARK_CLOSING_RADIUS,
            BRIGHT_LESION_SE_RADIUS, BRIGHT_LESION_THRESHOLD
        )

        # Create heatmap overlays
        red_lesion_overlay_bgr = create_heatmap_overlay(
            original_resized_img_uint8, dark_lesion_map, OVERLAY_TRANSPARENCY, HEATMAP_COLORMAP
        )
        bright_lesion_overlay_bgr = create_heatmap_overlay(
            original_resized_img_uint8, bright_lesion_map, OVERLAY_TRANSPARENCY, HEATMAP_COLORMAP
        )

        # Prepare a single figure for output with 5 subplots
        # (Original, Dark Overlay, Bright Overlay, Pure Dark Mask, Pure Bright Mask)
        fig, axes = plt.subplots(1, 5, figsize=(25, 6))  # Increased figure size for 5 plots
        fig.suptitle(f"Lesion Analysis for: {os.path.basename(img_file_path)}", fontsize=16)

        # 1. Original Image
        axes[0].imshow(cv2.cvtColor(original_resized_img_uint8, cv2.COLOR_BGR2RGB))
        axes[0].set_title("1. Original Image")
        axes[0].axis('off')

        # 2. Dark Lesions (Heatmap Overlay)
        axes[1].imshow(cv2.cvtColor(red_lesion_overlay_bgr, cv2.COLOR_BGR2RGB))
        axes[1].set_title("2. Dark Lesions (Overlay)")
        axes[1].axis('off')

        # 3. Bright Lesions (Heatmap Overlay)
        axes[2].imshow(cv2.cvtColor(bright_lesion_overlay_bgr, cv2.COLOR_BGR2RGB))
        axes[2].set_title("3. Bright Lesions (Overlay)")
        axes[2].axis('off')

        # 4. Pure Dark Lesion Mask (Similar to Paper's Figure 5(c))
        axes[3].imshow(dark_lesion_map, cmap='gray', vmin=0, vmax=1)  # Use grayscale colormap
        axes[3].set_title("4. Pure Dark Lesion Mask")
        axes[3].axis('off')

        # 5. Pure Bright Lesion Mask (Similar to Paper's Figure 5(d))
        axes[4].imshow(bright_lesion_map, cmap='gray', vmin=0, vmax=1)  # Use grayscale colormap
        axes[4].set_title("5. Pure Bright Lesion Mask")
        axes[4].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to prevent title overlap

        # Save the figure
        if SAVE_OUTPUT_TO_FILE:
            output_basename = f"{OUTPUT_FILENAME_PREFIX}{os.path.splitext(os.path.basename(img_file_path))[0]}.png"
            output_full_path = os.path.join(output_dir, output_basename)
            plt.savefig(output_full_path, bbox_inches='tight', pad_inches=0.1)
            print(f"  Output saved to: {output_full_path}")
        else:
            plt.show()  # Show interactive plot if not saving (for quick checks)

        plt.close(fig)  # Close the figure to free up memory

    print("\nProcessing complete. Check the 'output_lesion_analysis' directory for results.")