import os
from PIL import Image
import numpy as np
from skimage.morphology import disk, opening, closing
import cv2
import matplotlib.pyplot as plt

def preprocess_fundus_image(image_path, size=(640, 640)):
    img = Image.open(image_path).convert("RGB")
    resized_img = img.resize(size, Image.LANCZOS)
    img_array = np.array(resized_img, dtype=np.float32)
    normalized_array = (img_array / 255.0 * 2) - 1  # Normalize to [-1, 1]
    return normalized_array, np.array(resized_img)

def decompose_image_morphological_approximation(pre_img, se_radius_small=3, threshold_factor=0.05):
    img_display_scale = (((pre_img + 1) / 2) * 255).astype(np.uint8)
    green_channel = img_display_scale[:, :, 1]  # Use green channel

    se = disk(se_radius_small)

    # Bright regions (e.g. exudates)
    opened = opening(green_channel, se)
    ibri_raw = np.maximum(0, green_channel - opened)
    ibri = ibri_raw.astype(np.float32)
    if ibri.max() > 0:
        ibri /= ibri.max()
    ibri[ibri < threshold_factor] = 0
    # Keep normalized to [0,1] for heatmap visualization
    ibri = ibri

    # Dark regions (e.g. red lesions)
    closed = closing(green_channel, se)
    idark_raw = np.maximum(0, closed - green_channel)
    idark = idark_raw.astype(np.float32)
    if idark.max() > 0:
        idark /= idark.max()
    idark[idark < threshold_factor] = 0
    idark = idark

    return idark, ibri

def apply_colormap_to_attention(attention_map, colormap=plt.cm.hot):
    # Normalize attention to [0,255]
    if attention_map.max() == 0:
        attention_map_norm = attention_map
    else:
        attention_map_norm = attention_map / attention_map.max()
    attention_norm = np.uint8(attention_map_norm * 255)
    # Apply matplotlib colormap (RGBA)
    colored = colormap(attention_norm)
    # Convert to 8-bit RGB
    colored_rgb = np.delete(colored, 3, 2)  # remove alpha channel
    colored_rgb = np.uint8(colored_rgb * 255)
    # Convert RGB to BGR for OpenCV
    colored_bgr = cv2.cvtColor(colored_rgb, cv2.COLOR_RGB2BGR)
    return colored_bgr

def plot_attention_maps_cv(images_paths, se_radius=3, threshold=0.05):
    for img_path in images_paths:
        pre_img, orig_img = preprocess_fundus_image(img_path)
        idark, ibri = decompose_image_morphological_approximation(pre_img, se_radius_small=se_radius, threshold_factor=threshold)

        # Convert original image (RGB) to BGR for OpenCV
        orig_bgr = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)

        # Get attention heatmaps with colormap applied
        heat_red = apply_colormap_to_attention(idark)
        heat_bright = apply_colormap_to_attention(ibri)

        # Resize heatmaps to match original if needed
        heat_red = cv2.resize(heat_red, (orig_bgr.shape[1], orig_bgr.shape[0]))
        heat_bright = cv2.resize(heat_bright, (orig_bgr.shape[1], orig_bgr.shape[0]))

        # Blend heatmaps on original image for visualization (alpha blending)
        blend_red = cv2.addWeighted(orig_bgr, 0.7, heat_red, 0.3, 0)
        blend_bright = cv2.addWeighted(orig_bgr, 0.7, heat_bright, 0.3, 0)

        # Stack images horizontally: original | red lesion attention | bright lesion attention
        combined = np.hstack((orig_bgr, blend_red, blend_bright))

        # Show in OpenCV window (resize window if too large)
        window_name = f"Attention Maps - {os.path.basename(img_path)}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 3 * 640, 640)  # 3 images wide, height 640
        cv2.imshow(window_name, combined)

        print(f"Showing {img_path}. Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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
            print(f"Image not found: {p}")
            exit(1)

    plot_attention_maps_cv(test_images)
