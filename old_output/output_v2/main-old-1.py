import os
from PIL import Image
import numpy as np
from skimage.morphology import disk, opening, closing
import cv2
import matplotlib.pyplot as plt  # Optional, only if you want matplotlib


def preprocess_fundus_image(image_path, size=(640, 640)):
    img = Image.open(image_path).convert("RGB")
    resized_img = img.resize(size, Image.LANCZOS)
    img_array_float = np.array(resized_img, dtype=np.float32)

    # Normalize to [-1, 1] for morphological ops
    normalized_array = (img_array_float / 255.0 * 2) - 1

    return normalized_array, np.array(resized_img)  # preprocessed and original


def decompose_image_morphological_approximation(pre_img, se_radius_small=5, threshold_factor=0.01,
                                                use_alternative_dark=False):
    img_display_scale = (((pre_img + 1) / 2) * 255).astype(np.uint8)
    green_channel = img_display_scale[:, :, 1]

    se = disk(se_radius_small)

    # Bright regions (exudates)
    opened = opening(green_channel, se)
    ibri_raw = np.maximum(0, green_channel - opened)
    ibri = ibri_raw.astype(np.float32)
    if ibri.max() > 0:
        ibri /= ibri.max()
    ibri[ibri < threshold_factor] = 0

    if not use_alternative_dark:
        # Dark regions (red lesions) - original method
        closed = closing(green_channel, se)
        idark_raw = np.maximum(0, closed - green_channel)
        idark = idark_raw.astype(np.float32)
        if idark.max() > 0:
            idark /= idark.max()
        idark[idark < threshold_factor] = 0
    else:
        # Alternative dark spot detection method
        inv_green = 255 - green_channel
        opened_inv = opening(inv_green, se)
        idark_raw = np.maximum(0, opened_inv - inv_green)
        idark = idark_raw.astype(np.float32)
        if idark.max() > 0:
            idark /= idark.max()
        idark[idark < threshold_factor] = 0

    return idark, ibri


def create_heatmap_overlay(original_img_np, attention_map, alpha=0.6, colormap=cv2.COLORMAP_JET):
    original_img_uint8 = original_img_np.astype(np.uint8)
    attention_map_uint8 = (attention_map * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(attention_map_uint8, colormap)
    original_bgr = cv2.cvtColor(original_img_uint8, cv2.COLOR_RGB2BGR)
    overlay_img = cv2.addWeighted(original_bgr, 1 - alpha, heatmap_bgr, alpha, 0)
    return overlay_img


def generate_figure_columns(image_path, se_radius=5, threshold=0.01, img_size=(640, 640),
                            use_alternative_dark=False):
    pre_img, original_resized_img_np = preprocess_fundus_image(image_path, size=img_size)
    idark, ibri = decompose_image_morphological_approximation(pre_img, se_radius_small=se_radius,
                                                              threshold_factor=threshold,
                                                              use_alternative_dark=use_alternative_dark)

    original_display_bgr = cv2.cvtColor(original_resized_img_np, cv2.COLOR_RGB2BGR)
    red_lesion_overlay = create_heatmap_overlay(original_resized_img_np, idark, alpha=0.6)
    bright_lesion_overlay = create_heatmap_overlay(original_resized_img_np, ibri, alpha=0.6)

    return [original_display_bgr, red_lesion_overlay, bright_lesion_overlay]


def create_color_bar(height, width, colormap=cv2.COLORMAP_JET, ticks=None, labels=None):
    colorbar_img = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        intensity = int(255 * (height - 1 - i) / (height - 1))
        color_bgr = cv2.applyColorMap(np.array([[intensity]], dtype=np.uint8), colormap)[0, 0]
        colorbar_img[i, :] = color_bgr

    if ticks and labels:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 1
        text_color = (0, 0, 0)
        for i, tick_val in enumerate(ticks):
            y_pos = int((1 - tick_val) * (height - 1))
            cv2.line(colorbar_img, (0, y_pos), (width // 4, y_pos), (255, 255, 255), 1)
            label = str(labels[i])
            text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
            text_x = width // 3 + 5
            text_y = y_pos + text_size[1] // 2
            cv2.putText(colorbar_img, label, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    return colorbar_img


# Optional debug visualization to help you check intermediate results
def debug_show_images(images_dict):
    for name, img in images_dict.items():
        cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_images = [
        "test_before_train/60f15dd68d30.png",
        "old/4.png"
    ]

    for p in test_images:
        if not os.path.isfile(p):
            print(f"Error: Image not found at {p}. Please update `test_images` paths.")
            exit(1)

    all_rows_images = []
    image_display_size = (640, 640)

    # Toggle alternative dark spot detection here:
    use_alternative_dark_spot_method = False

    for img_path in test_images:
        print(f"Processing {os.path.basename(img_path)}...")
        row_images = generate_figure_columns(img_path, se_radius=5, threshold=0.01,
                                             img_size=image_display_size,
                                             use_alternative_dark=use_alternative_dark_spot_method)
        all_rows_images.append(row_images)

    if all_rows_images:
        combined_rows = [np.hstack(row) for row in all_rows_images]
        final_figure_main = np.vstack(combined_rows)

        colorbar_height = final_figure_main.shape[0]
        colorbar_width = 80
        colorbar_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        colorbar_labels = ["0", "0.2", "0.4", "0.6", "0.8", "1"]
        colorbar_labels.reverse()
        colorbar_ticks.reverse()

        custom_colorbar = create_color_bar(colorbar_height, colorbar_width,
                                           colormap=cv2.COLORMAP_JET,
                                           ticks=colorbar_ticks,
                                           labels=colorbar_labels)

        final_figure_with_colorbar = np.hstack((final_figure_main, custom_colorbar))

        cv2.namedWindow("Figure 5 Recreation", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Figure 5 Recreation",
                         final_figure_with_colorbar.shape[1],
                         final_figure_with_colorbar.shape[0])
        cv2.imshow("Figure 5 Recreation", final_figure_with_colorbar)

        print("Press any key to close the window.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No images were processed to create the figure.")
