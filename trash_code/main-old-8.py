from PIL import Image
import numpy as np
from skimage.morphology import disk, opening, closing
from skimage.filters import gaussian
from skimage.exposure import equalize_adapthist
import os

def preprocess_fundus_image(image_path):
    """
    Load, resize, and normalize the fundus image.
    """
    img = Image.open(image_path).convert("RGB")
    resized_img = img.resize((640, 640), Image.LANCZOS)
    img_array = np.array(resized_img, dtype=np.float32)
    normalized_array = (img_array / 255.0 * 2) - 1  # normalize to [-1, 1]
    return normalized_array

def decompose_image_morphological_approximation(preprocessed_image_array, se_radius_small=3, threshold_factor=0.05):
    """
    Approximate morphological decomposition to extract:
    - Idark: red lesions (e.g., microaneurysms, hemorrhages)
    - Ibri: bright lesions (e.g., exudates)

    Uses CLAHE + smoothing + morphological filtering.
    """
    img_display_scale = (((preprocessed_image_array + 1) / 2) * 255).astype(np.uint8)

    if img_display_scale.ndim == 3:
        green_channel = img_display_scale[:, :, 1]
    else:
        green_channel = img_display_scale

    # --- Step 1: CLAHE (Contrast enhancement for lesions) ---
    clahe_img = equalize_adapthist(green_channel, clip_limit=0.03)  # [0,1] float
    clahe_img = (clahe_img * 255).astype(np.uint8)

    # --- Step 2: Gaussian smoothing ---
    smoothed = gaussian(clahe_img, sigma=1)  # float64 in [0,1]

    # --- Step 3: Morphological closing to detect dark spots ---
    se = disk(se_radius_small)
    closed = closing(smoothed, se)
    Idark_raw = np.maximum(0, closed - smoothed)

    Idark_float = Idark_raw.astype(np.float32)
    if Idark_float.max() > 0:
        Idark_float /= Idark_float.max()
    Idark_float[Idark_float < threshold_factor] = 0
    Idark = (Idark_float * 2) - 1

    # --- Step 4: Morphological opening to detect bright lesions ---
    opened = opening(smoothed, se)
    Ibri_raw = np.maximum(0, smoothed - opened)

    Ibri_float = Ibri_raw.astype(np.float32)
    if Ibri_float.max() > 0:
        Ibri_float /= Ibri_float.max()
    Ibri_float[Ibri_float < threshold_factor] = 0
    Ibri = (Ibri_float * 2) - 1

    return Idark, Ibri

# ------------------------ MAIN ------------------------ #
if __name__ == "__main__":
    image_path = "old/4.png"  # Path to fundus image

    # Step 1: Preprocess the image
    preprocessed_image_array = preprocess_fundus_image(image_path)
    print("Preprocessing complete. Shape:", preprocessed_image_array.shape,
          "Min/Max:", preprocessed_image_array.min(), preprocessed_image_array.max())

    # Step 2: Decompose image into dark and bright lesion maps
    idark_image, ibri_image = decompose_image_morphological_approximation(
        preprocessed_image_array,
        se_radius_small=3,
        threshold_factor=0.05
    )

    print("\nDecomposition complete.")
    print("Idark shape:", idark_image.shape, "Min/Max:", idark_image.min(), idark_image.max())
    print("Ibri shape:", ibri_image.shape, "Min/Max:", ibri_image.min(), ibri_image.max())

    # Step 3: Save outputs as images
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    idark_display = (((idark_image + 1) / 2) * 255).astype(np.uint8)
    ibri_display = (((ibri_image + 1) / 2) * 255).astype(np.uint8)

    idark_pil_image = Image.fromarray(idark_display)
    ibri_pil_image = Image.fromarray(ibri_display)

    idark_path = os.path.join(output_dir, "Idark_approximation.jpg")
    ibri_path = os.path.join(output_dir, "Ibri_approximation.jpg")

    idark_pil_image.save(idark_path)
    ibri_pil_image.save(ibri_path)

    print(f"Saved: {idark_path}")
    print(f"Saved: {ibri_path}")

    idark_pil_image.show()
    ibri_pil_image.show()
