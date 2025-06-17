from PIL import Image
import numpy as np
from skimage.morphology import disk, opening, closing
import cv2

def preprocess_fundus_image(image_path, size=(640, 640)):
    img = Image.open(image_path).convert("RGB")
    resized_img = img.resize(size, Image.LANCZOS)
    img_array = np.array(resized_img, dtype=np.float32)
    normalized_array = (img_array / 255.0 * 2) - 1  # Normalize to [-1, 1]
    return normalized_array, resized_img

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
    ibri = (ibri * 2) - 1  # Normalize to [-1, 1]

    # Dark regions (e.g. red lesions)
    closed = closing(green_channel, se)
    idark_raw = np.maximum(0, closed - green_channel)
    idark = idark_raw.astype(np.float32)
    if idark.max() > 0:
        idark /= idark.max()
    idark[idark < threshold_factor] = 0
    idark = (idark * 2) - 1  # Normalize to [-1, 1]

    return idark, ibri

def save_and_show(image_array, filename, window_title):
    # Convert normalized [-1,1] image to uint8 [0,255]
    out_img = (((image_array + 1) / 2) * 255).astype(np.uint8)
    Image.fromarray(out_img).save(filename)
    print(f"Saved: {filename}")

    cv2.imshow(window_title, out_img)
    key = cv2.waitKey(0)
    if key == 27:  # ESC key to close window
        cv2.destroyAllWindows()

def overlay_on_green(image_array, green_channel, filename):
    mask = (((image_array + 1) / 2) * 255).astype(np.uint8)
    overlay = np.stack([green_channel, green_channel, green_channel], axis=-1)

    # Add red overlay on red channel (index 0 in RGB)
    overlay[..., 0] = np.maximum(overlay[..., 0], mask)

    Image.fromarray(overlay).save(filename)
    print(f"Saved overlay: {filename}")

    # Convert RGB to BGR for OpenCV display
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imshow("Overlay", overlay_bgr)
    key = cv2.waitKey(0)
    if key == 27:
        cv2.destroyAllWindows()

def generate_attention_heatmap_overlay(original_img, attention_map, alpha=0.6, colormap=cv2.COLORMAP_JET, save_path=None, window_title=None):
    """
    original_img: np.array (H,W,3) uint8 RGB image
    attention_map: np.array (H,W), float in [-1,1]
    alpha: blending factor for overlay
    colormap: OpenCV colormap type
    save_path: filename to save overlay image (optional)
    window_title: if provided, show overlay in OpenCV window (optional)
    """

    # Normalize attention map from [-1,1] to [0,255] uint8
    att_norm = np.clip((attention_map + 1) / 2, 0, 1)  # [0,1]
    att_uint8 = (att_norm * 255).astype(np.uint8)

    # Apply colormap to attention map to get heatmap (BGR)
    heatmap = cv2.applyColorMap(att_uint8, colormap)

    # Convert original RGB to BGR for OpenCV
    original_bgr = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)

    # Blend heatmap and original image
    overlay = cv2.addWeighted(heatmap, alpha, original_bgr, 1 - alpha, 0)

    # Save image if filename provided
    if save_path:
        cv2.imwrite(save_path, overlay)
        print(f"Saved heatmap overlay: {save_path}")

    # Show window if requested
    if window_title:
        cv2.imshow(window_title, overlay)
        key = cv2.waitKey(0)
        if key == 27:
            cv2.destroyAllWindows()

    return overlay


# === MAIN ===
if __name__ == "__main__":
    image_path = "old/4.png"  # Change to your image path
    pre_img, original_img = preprocess_fundus_image(image_path)
    idark, ibri = decompose_image_morphological_approximation(pre_img, se_radius_small=3, threshold_factor=0.05)

    # Save & Show lesion approximations
    save_and_show(idark, "Idark_approximation.jpg", "Red Lesions (Dark Features)")
    save_and_show(ibri, "Ibri_approximation.jpg", "Bright Lesions")

    green = np.array(original_img)[:, :, 1]
    overlay_on_green(idark, green, "Idark_overlay_on_green.jpg")
    overlay_on_green(ibri, green, "Ibri_overlay_on_green.jpg")

    # Generate and show/save attention heatmaps overlaid on original fundus image
    original_np = np.array(original_img)
    generate_attention_heatmap_overlay(original_np, idark, alpha=0.6, save_path="Idark_heatmap_overlay.jpg", window_title="Red Lesions Heatmap")
    generate_attention_heatmap_overlay(original_np, ibri, alpha=0.6, save_path="Ibri_heatmap_overlay.jpg", window_title="Bright Lesions Heatmap")
