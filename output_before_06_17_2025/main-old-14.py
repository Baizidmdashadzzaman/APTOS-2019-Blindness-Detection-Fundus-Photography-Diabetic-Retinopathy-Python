import os
from PIL import Image
import numpy as np
from skimage.morphology import disk, opening, closing
import cv2
import matplotlib.pyplot as plt  # Still useful if we want to custom create a colormap or its bar


def preprocess_fundus_image(image_path, size=(640, 640)):
    img = Image.open(image_path).convert("RGB")
    resized_img = img.resize(size, Image.LANCZOS)
    img_array_float = np.array(resized_img, dtype=np.float32)

    # Normalize to [-1, 1] for the 'pre_img' as used in decompose_image_morphological_approximation
    normalized_array = (img_array_float / 255.0 * 2) - 1

    # Return the normalized array for morphological ops, and the 0-255 original for overlay
    return normalized_array, np.array(resized_img)  # original_img is uint8 [0, 255]


def decompose_image_morphological_approximation(pre_img, se_radius_small=3, threshold_factor=0.05):
    # This part was correct, it converts pre_img ([-1,1]) back to (0,255) uint8 to get green channel
    img_display_scale = (((pre_img + 1) / 2) * 255).astype(np.uint8)
    green_channel = img_display_scale[:, :, 1]  # Use green channel

    se = disk(se_radius_small)

    # Bright regions (e.g. exudates)
    opened = opening(green_channel, se)
    ibri_raw = np.maximum(0, green_channel - opened)
    ibri = ibri_raw.astype(np.float32)
    if ibri.max() > 0:
        ibri /= ibri.max()  # Normalize to [0, 1]
    # Apply thresholding: values below threshold become 0
    ibri[ibri < threshold_factor] = 0

    # Dark regions (e.g. red lesions)
    closed = closing(green_channel, se)
    idark_raw = np.maximum(0, closed - green_channel)
    idark = idark_raw.astype(np.float32)
    if idark.max() > 0:
        idark /= idark.max()  # Normalize to [0, 1]
    # Apply thresholding: values below threshold become 0
    idark[idark < threshold_factor] = 0

    return idark, ibri  # idark and ibri are now normalized to [0, 1] for colormapping


def create_heatmap_overlay(original_img_np, attention_map, alpha=0.6, colormap=cv2.COLORMAP_JET):
    """
    Creates a heatmap overlay on the original image, matching the paper's style.
    original_img_np: NumPy array of the original image (H, W, 3) in RGB (0-255).
    attention_map: NumPy array (H, W) normalized to [0, 1].
    alpha: Blending factor for the heatmap (controls how much the heatmap color shows).
           Lower alpha makes original image more prominent, higher makes heatmap stronger.
    colormap: OpenCV colormap (e.g., cv2.COLORMAP_JET).
    Returns: BGR image (0-255) with heatmap overlay.
    """
    # Ensure original_img_np is uint8
    original_img_uint8 = original_img_np.astype(np.uint8)

    # Convert attention map from [0, 1] to [0, 255] uint8 for colormap application
    # The map will be darker (blue) for low values, brighter (red) for high values
    attention_map_uint8 = (attention_map * 255).astype(np.uint8)

    # Apply colormap to get heatmap (BGR format)
    heatmap_bgr = cv2.applyColorMap(attention_map_uint8, colormap)

    # Convert original RGB to BGR for OpenCV blending
    original_bgr = cv2.cvtColor(original_img_uint8, cv2.COLOR_RGB2BGR)

    # Blend heatmap and original image
    # original_bgr * (1 - alpha) + heatmap_bgr * alpha
    # The paper's heatmaps seem to have a strong presence, so alpha can be relatively high.
    # We want the 'hot' regions to dominate.
    overlay_img = cv2.addWeighted(original_bgr, 1 - alpha, heatmap_bgr, alpha, 0)
    return overlay_img


def generate_figure_columns(image_path, se_radius=3, threshold=0.05, img_size=(640, 640)):
    """
    Generates the three columns for a single row of the figure:
    Original image, Red Lesion Attention Map overlay, Bright Lesion Attention Map overlay.
    Returns a list of 3 BGR images.
    """
    # preprocess_fundus_image returns pre_img ([-1,1]) and original_resized_img_np (0-255)
    pre_img, original_resized_img_np = preprocess_fundus_image(image_path, size=img_size)

    # Decompose to get attention maps (normalized to [0,1])
    idark, ibri = decompose_image_morphological_approximation(pre_img, se_radius_small=se_radius,
                                                              threshold_factor=threshold)

    # Convert original resized image to BGR for the first column
    original_display_bgr = cv2.cvtColor(original_resized_img_np, cv2.COLOR_RGB2BGR)

    # Create heatmap overlays. Adjust alpha if needed for desired transparency.
    red_lesion_overlay = create_heatmap_overlay(original_resized_img_np, idark,
                                                alpha=0.6)  # Alpha 0.6 is a good starting point
    bright_lesion_overlay = create_heatmap_overlay(original_resized_img_np, ibri, alpha=0.6)

    return [original_display_bgr, red_lesion_overlay, bright_lesion_overlay]


def create_color_bar(height, width, colormap=cv2.COLORMAP_JET, ticks=None, labels=None):
    """
    Creates a vertical color bar image.
    height: Height of the color bar in pixels.
    width: Width of the color bar in pixels.
    colormap: OpenCV colormap to use.
    ticks: List of normalized values [0,1] for tick marks (e.g., [0, 0.5, 1]).
    labels: List of strings for tick labels.
    Returns: BGR image of the color bar.
    """
    colorbar_img = np.zeros((height, width, 3), dtype=np.uint8)

    # Generate the gradient
    for i in range(height):
        # Map vertical position (from top to bottom) to 0-255 intensity
        # To match the paper, high values (1.0) are at the top (red), low values (0.0) at the bottom (blue).
        intensity = int(255 * (height - 1 - i) / (height - 1))
        color_bgr = cv2.applyColorMap(np.array([[intensity]], dtype=np.uint8), colormap)[0, 0]
        colorbar_img[i, :] = color_bgr

    # Optional: Add tick marks and labels
    if ticks and labels:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 1
        text_color = (0, 0, 0)  # Black color for labels

        for i, tick_val in enumerate(ticks):
            # Calculate y position for the tick
            # y_pos = height - 1 - int(tick_val * (height - 1))
            y_pos = int((1 - tick_val) * (height - 1))  # Inverse mapping for top=1, bottom=0

            # Draw a small white tick mark
            cv2.line(colorbar_img, (0, y_pos), (width // 4, y_pos), (255, 255, 255), 1)

            # Add label
            label = str(labels[i])
            text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
            text_x = width // 3 + 5  # Offset from the tick mark
            text_y = y_pos + text_size[1] // 2  # Center vertically
            cv2.putText(colorbar_img, label, (text_x, text_y), font, font_scale, text_color, font_thickness,
                        cv2.LINE_AA)

    return colorbar_img


if __name__ == "__main__":
    # List your test images here (change paths as needed)
    test_images = [
        "old/4.png",
        "test_images/2.jpeg",
        "test_images/3.jpeg"
    ]

    # Check images exist before running
    for p in test_images:
        if not os.path.isfile(p):
            print(f"Error: Image not found at {p}. Please update `test_images` paths.")
            exit(1)

    all_rows_images = []
    # Use a fixed image size that matches the example
    image_display_size = (640, 640)

    for img_path in test_images:
        print(f"Processing {os.path.basename(img_path)}...")
        row_images = generate_figure_columns(img_path, se_radius=3, threshold=0.05, img_size=image_display_size)
        all_rows_images.append(row_images)

    # Stack all rows vertically
    if all_rows_images:
        # Each row_images is a list of [original, red_overlay, bright_overlay]
        # First, hstack each row
        combined_rows = [np.hstack(row) for row in all_rows_images]

        # Then, vstack all combined rows
        final_figure_main = np.vstack(combined_rows)

        # Create the color bar to match the height of the entire figure
        colorbar_height = final_figure_main.shape[0]
        colorbar_width = 80  # Adjust width as needed
        colorbar_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  # Example ticks
        colorbar_labels = ["0", "0.2", "0.4", "0.6", "0.8", "1"]  # Example labels
        # Reverse labels to match top-to-bottom display of 1 to 0
        colorbar_labels.reverse()
        colorbar_ticks.reverse()  # Reverse ticks as well to match labels

        custom_colorbar = create_color_bar(colorbar_height, colorbar_width,
                                           colormap=cv2.COLORMAP_JET,
                                           ticks=colorbar_ticks,
                                           labels=colorbar_labels)

        # Concatenate the main figure with the color bar
        final_figure_with_colorbar = np.hstack((final_figure_main, custom_colorbar))

        # Display the final composite figure
        cv2.namedWindow("Figure 5 Recreation", cv2.WINDOW_NORMAL)
        # Resize window to fit, considering 3 images wide + colorbar
        cv2.resizeWindow("Figure 5 Recreation",
                         final_figure_with_colorbar.shape[1],
                         final_figure_with_colorbar.shape[0])
        cv2.imshow("Figure 5 Recreation", final_figure_with_colorbar)

        # Save the final figure
        output_filename = "recreated_figure5_heatmap_style.jpg"
        cv2.imwrite(output_filename, final_figure_with_colorbar)
        print(f"\nFinal figure saved as: {output_filename}")

        print("Press any key to close the window.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No images were processed to create the figure.")