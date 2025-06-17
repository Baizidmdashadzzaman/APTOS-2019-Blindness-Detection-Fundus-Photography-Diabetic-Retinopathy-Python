from PIL import Image
import numpy as np
from skimage.morphology import disk, opening, closing
import matplotlib
matplotlib.use('Agg')  # Avoid plt.show() error in PyCharm
import matplotlib.pyplot as plt
import os

def preprocess_fundus_image(image_path):
    """
    Load, resize, normalize the fundus image to [-1, 1]
    """
    img = Image.open(image_path).convert("RGB")
    resized_img = img.resize((640, 640), Image.LANCZOS)
    img_array = np.array(resized_img, dtype=np.float32)
    normalized_array = (img_array / 255.0 * 2) - 1
    return normalized_array

def decompose_image_morphological_approximation(preprocessed_image_array, se_radius_small=3, threshold_factor=0.05):
    """
    Morphological approximation to extract dark (Idark) and bright (Ibri) regions
    """
    img_display_scale = (((preprocessed_image_array + 1) / 2) * 255).astype(np.uint8)

    if img_display_scale.ndim == 3:
        gray_img = img_display_scale[:, :, 1]  # Use green channel
    else:
        gray_img = img_display_scale

    se_small = disk(se_radius_small)

    # Bright lesions
    opened = opening(gray_img, se_small)
    Ibri_raw = np.maximum(0, gray_img - opened)
    Ibri_float = Ibri_raw.astype(np.float32)
    if Ibri_float.max() > 0:
        Ibri_float /= Ibri_float.max()
    Ibri_float[Ibri_float < threshold_factor] = 0
    Ibri = (Ibri_float * 2) - 1

    # Dark lesions
    closed = closing(gray_img, se_small)
    Idark_raw = np.maximum(0, closed - gray_img)
    Idark_float = Idark_raw.astype(np.float32)
    if Idark_float.max() > 0:
        Idark_float /= Idark_float.max()
    Idark_float[Idark_float < threshold_factor] = 0
    Idark = (Idark_float * 2) - 1

    return Idark, Ibri

def save_result(image_array, filename):
    """ Save [-1, 1] image array to a file in [0, 255] scale """
    img_uint8 = (((image_array + 1) / 2) * 255).astype(np.uint8)
    Image.fromarray(img_uint8).save(filename)
    print("Saved:", filename)
    return img_uint8

def overlay_on_green_channel(original_array, mask_array, filename):
    """
    Overlay white mask on green channel of original image.
    """
    green = (((original_array + 1) / 2) * 255).astype(np.uint8)[:, :, 1]
    mask = (((mask_array + 1) / 2) * 255).astype(np.uint8)
    overlay = np.stack([green]*3, axis=-1)
    overlay[mask > 50] = [255, 255, 255]
    Image.fromarray(overlay).save(filename)
    print("Saved overlay:", filename)
    return overlay

# ==== Main ====
image_path = "old/4.png"  # Set your image path
preprocessed = preprocess_fundus_image(image_path)
print("Preprocessing done. Shape:", preprocessed.shape)

Idark, Ibri = decompose_image_morphological_approximation(
    preprocessed, se_radius_small=3, threshold_factor=0.05
)

# Save outputs
idark_display = save_result(Idark, "Idark_approximation.jpg")
ibri_display = save_result(Ibri, "Ibri_approximation.jpg")

# Save overlay (optional)
overlay_on_green_channel(preprocessed, Idark, "Idark_overlay_on_green.jpg")
overlay_on_green_channel(preprocessed, Ibri, "Ibri_overlay_on_green.jpg")

# Save matplotlib plots
plt.imsave("Idark_plot.jpg", idark_display, cmap="gray")
plt.imsave("Ibri_plot.jpg", ibri_display, cmap="gray")

print("All done.")
