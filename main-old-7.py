from PIL import Image
import numpy as np
from skimage.morphology import disk, opening, closing
import os

def preprocess_fundus_image(image_path):
    """
    Preprocesses a fundus image by loading, resizing, and normalizing.
    """
    img = Image.open(image_path).convert("RGB")
    resized_img = img.resize((640, 640), Image.LANCZOS)
    img_array = np.array(resized_img, dtype=np.float32)
    normalized_array = (img_array / 255.0 * 2) - 1
    return normalized_array

def apply_circular_mask(image, center, radius):
    """
    Masks a circular region in the image by setting it to 0.
    """
    yy, xx = np.ogrid[:image.shape[0], :image.shape[1]]
    mask = (xx - center[0])**2 + (yy - center[1])**2 <= radius**2
    image[mask] = 0
    return image

def decompose_image_morphological_approximation(preprocessed_image_array, se_radius_small=3, se_radius_large=7, threshold_factor=0.2):
    """
    Approximate morphological decomposition to enhance dark (Idark) and bright (Ibri) regions.
    """
    img_display_scale = (((preprocessed_image_array + 1) / 2) * 255).astype(np.uint8)

    if img_display_scale.ndim == 3:
        gray_img = img_display_scale[:, :, 1]  # green channel
    else:
        gray_img = img_display_scale

    se_small = disk(se_radius_small)
    se_large = disk(se_radius_large)

    # --- Bright Region (Ibri) ---
    opened_for_bright = opening(gray_img, se_small)
    Ibri_raw = np.maximum(0, gray_img - opened_for_bright)
    Ibri_float = Ibri_raw.astype(np.float32)
    if Ibri_float.max() > 0:
        Ibri_float /= Ibri_float.max()
    Ibri_float[Ibri_float < threshold_factor] = 0
    Ibri = (Ibri_float * 2) - 1

    # --- Dark Region (Idark) ---
    closed_for_dark = closing(gray_img, se_small)
    Idark_raw = np.maximum(0, closed_for_dark - gray_img)
    Idark_float = Idark_raw.astype(np.float32)
    if Idark_float.max() > 0:
        Idark_float /= Idark_float.max()
    Idark_float[Idark_float < threshold_factor] = 0
    Idark = (Idark_float * 2) - 1

    # --- Mask optic disc region (optional) ---
    # Example center and radius — adjust based on your fundus image layout
    optic_disc_center = (450, 330)  # x, y coordinates
    optic_disc_radius = 60          # in pixels
    Idark = apply_circular_mask(Idark, optic_disc_center, optic_disc_radius)

    return Idark, Ibri

# ------------------- MAIN EXECUTION ------------------- #
if __name__ == "__main__":
    image_path = "old/4.png"  # Replace with your image path

    # 1. Preprocessing stage
    preprocessed_image_array = preprocess_fundus_image(image_path)
    print("Preprocessing complete. Shape:", preprocessed_image_array.shape,
          "Min/Max:", preprocessed_image_array.min(), preprocessed_image_array.max())

    # 2. Decomposition
    idark_image, ibri_image = decompose_image_morphological_approximation(
        preprocessed_image_array,
        se_radius_small=3,
        se_radius_large=7,
        threshold_factor=0.05
    )

    print("\nDecomposition complete.")
    print("Idark shape:", idark_image.shape, "Min/Max:", idark_image.min(), idark_image.max())
    print("Ibri shape:", ibri_image.shape, "Min/Max:", ibri_image.min(), ibri_image.max())

    # 3. Save and show results
    idark_display = (((idark_image + 1) / 2) * 255).astype(np.uint8)
    ibri_display = (((ibri_image + 1) / 2) * 255).astype(np.uint8)

    idark_pil_image = Image.fromarray(idark_display)
    ibri_pil_image = Image.fromarray(ibri_display)

    os.makedirs("output", exist_ok=True)
    idark_pil_image.save("output/Idark_approximation.jpg")
    ibri_pil_image.save("output/Ibri_approximation.jpg")

    print("Saved images to 'output/' directory.")
    idark_pil_image.show()
    ibri_pil_image.show()
