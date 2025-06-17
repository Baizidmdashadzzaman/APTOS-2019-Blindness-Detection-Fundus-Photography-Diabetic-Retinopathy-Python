import os
from PIL import Image
import numpy as np
from skimage.morphology import disk, opening, closing
import cv2
import matplotlib.pyplot as plt # Keep for general colormap options, though cv2.COLORMAP_JET is probably best here

def preprocess_fundus_image(image_path, size=(640, 640)):
    img = Image.open(image_path).convert("RGB")
    resized_img = img.resize(size, Image.LANCZOS)
    img_array = np.array(resized_img, dtype=np.float32)
    # No normalization to [-1, 1] for morphological ops in this case,
    # as we want to work with 0-255 range initially for green channel extraction.
    # The output from this function will be the raw resized image and its array.
    return img_array, np.array(resized_img)

def decompose_image_morphological_approximation(img_array_0_255, se_radius_small=3, threshold_factor=0.05):
    # Ensure input is 0-255 for green channel extraction
    img_display_scale = img_array_0_255.astype(np.uint8)
    green_channel = img_display_scale[:, :, 1]  # Use green channel

    se = disk(se_radius_small)

    # Bright regions (e.g. exudates)
    opened = opening(green_channel, se)
    ibri_raw = np.maximum(0, green_channel - opened)
    ibri = ibri_raw.astype(np.float32)
    if ibri.max() > 0:
        ibri /= ibri.max() # Normalize to [0, 1]
    ibri[ibri < threshold_factor] = 0


    # Dark regions (e.g. red lesions)
    closed = closing(green_channel, se)
    idark_raw = np.maximum(0, closed - green_channel)
    idark = idark_raw.astype(np.float32)
    if idark.max() > 0:
        idark /= idark.max() # Normalize to [0, 1]
    idark[idark < threshold_factor] = 0

    return idark, ibri # idark and ibri are now normalized to [0, 1]

def create_heatmap_overlay(original_img_np, attention_map, alpha=0.6, colormap=cv2.COLORMAP_JET):
    """
    Creates a heatmap overlay on the original image.
    original_img_np: NumPy array of the original image (H, W, 3) in RGB (0-255).
    attention_map: NumPy array (H, W) normalized to [0, 1].
    alpha: Blending factor for the heatmap.
    colormap: OpenCV colormap (e.g., cv2.COLORMAP_JET).
    Returns: BGR image (0-255) with heatmap overlay.
    """
    # Ensure original_img_np is uint8
    original_img_uint8 = original_img_np.astype(np.uint8)

    # Convert attention map from [0, 1] to [0, 255] uint8 for colormap application
    attention_map_uint8 = (attention_map * 255).astype(np.uint8)

    # Apply colormap to get heatmap
    heatmap = cv2.applyColorMap(attention_map_uint8, colormap)

    # Convert original RGB to BGR for OpenCV blending
    original_bgr = cv2.cvtColor(original_img_uint8, cv2.COLOR_RGB2BGR)

    # Blend heatmap and original image
    # Use 1.0 - alpha for the original image to make the heatmap more visible.
    # The example figure seems to have a dominant heatmap with original image detail visible.
    overlay_img = cv2.addWeighted(heatmap, alpha, original_bgr, 1 - alpha, 0)
    return overlay_img

def generate_figure_columns(image_path, se_radius=3, threshold=0.05, img_size=(640, 640)):
    """
    Generates the three columns for a single row of the figure:
    Original image, Red Lesion Attention Map overlay, Bright Lesion Attention Map overlay.
    Returns a list of 3 BGR images.
    """
    # preprocess_fundus_image returns img_array_0_255 and original_resized_img_np_0_255
    pre_img_arr_0_255, original_resized_img_np = preprocess_fundus_image(image_path, size=img_size)

    # Decompose to get attention maps (normalized to [0,1])
    idark, ibri = decompose_image_morphological_approximation(pre_img_arr_0_255, se_radius_small=se_radius, threshold_factor=threshold)

    # Convert original resized image to BGR for display if needed later
    original_display = cv2.cvtColor(original_resized_img_np, cv2.COLOR_RGB2BGR)

    # Create heatmap overlays
    red_lesion_overlay = create_heatmap_overlay(original_resized_img_np, idark, alpha=0.6, colormap=cv2.COLORMAP_JET)
    bright_lesion_overlay = create_heatmap_overlay(original_resized_img_np, ibri, alpha=0.6, colormap=cv2.COLORMAP_JET)

    return [original_display, red_lesion_overlay, bright_lesion_overlay]


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
    for img_path in test_images:
        print(f"Processing {os.path.basename(img_path)}...")
        row_images = generate_figure_columns(img_path, se_radius=3, threshold=0.05, img_size=(640, 640))
        all_rows_images.append(row_images)

    # Stack all rows vertically
    if all_rows_images:
        # Each row_images is a list of [original, red_overlay, bright_overlay]
        # First, hstack each row
        combined_rows = [np.hstack(row) for row in all_rows_images]

        # Then, vstack all combined rows
        final_figure = np.vstack(combined_rows)

        # Create a dummy colormap bar for visualization, similar to the example figure
        # This part is a bit more involved to get exactly like the example,
        # but here's a basic idea using matplotlib.
        # For perfect reproduction, consider creating this separately and adding it in an image editor.
        colorbar_height = final_figure.shape[0]
        colorbar_width = 50 # Width of the color bar
        colorbar_img = np.zeros((colorbar_height, colorbar_width, 3), dtype=np.uint8)

        for i in range(colorbar_height):
            # Map vertical position to 0-255 range
            intensity = int(255 * (colorbar_height - 1 - i) / (colorbar_height - 1))
            color_bgr = cv2.applyColorMap(np.array([[intensity]], dtype=np.uint8), cv2.COLORMAP_JET)[0,0]
            colorbar_img[i, :] = color_bgr

        # Concatenate the main figure with the color bar
        final_figure_with_colorbar = np.hstack((final_figure, colorbar_img))


        # Display the final composite figure
        cv2.namedWindow("Figure 5 Recreation", cv2.WINDOW_NORMAL)
        cv2.imshow("Figure 5 Recreation", final_figure_with_colorbar)

        # Save the final figure
        output_filename = "recreated_figure5.jpg"
        cv2.imwrite(output_filename, final_figure_with_colorbar)
        print(f"\nFinal figure saved as: {output_filename}")

        print("Press any key to close the window.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No images were processed to create the figure.")