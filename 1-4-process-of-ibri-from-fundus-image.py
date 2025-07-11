import cv2
import numpy as np

def sharpen_image(image_path):
    """
    Loads an image from the given path and applies a sharpening filter.

    Args:
        image_path (str): The path to the input image.

    Returns:
        numpy.ndarray: The sharpened image, or None if the image cannot be loaded.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image for sharpening from {image_path}")
        return None

    # Define a sharpening kernel
    sharpening_kernel = np.array([[-1, -1, -1],
                                  [-1,  10, -1],
                                  [-1, -1, -1]])

    # Apply the sharpening kernel to the image
    sharpened_image = cv2.filter2D(img, -1, sharpening_kernel)
    return sharpened_image

def make_green_and_adjust(img_source):
    """
    Applies color correction (greenish tint) and contrast/brightness adjustments
    to an input image array.

    Args:
        img_source (numpy.ndarray): The input image array.

    Returns:
        numpy.ndarray: The color-adjusted and contrast-modified image,
                       or None if the input array is None.
    """
    if img_source is None:
        print("Error: Input image for green adjustment is None.")
        return None

    # --- Color Correction (Introduce Cool/Greenish Tint and Desaturation) ---
    b, g, r = cv2.split(img_source)

    # Decrease Red (to remove warmth)
    r_adjusted = cv2.addWeighted(r, 0.7, np.zeros_like(r), 0, 0)

    # Increase Green and Blue (to add cool tint)
    g_adjusted = cv2.addWeighted(g, 1.2, np.zeros_like(g), 0, 0)
    b_adjusted = cv2.addWeighted(b, 1.1, np.zeros_like(b), 0, 0)

    img_color_adjusted = cv2.merge([b_adjusted, g_adjusted, r_adjusted])

    # --- Contrast and Brightness Adjustment (using LAB color space for luminance control) ---
    lab_img = cv2.cvtColor(img_color_adjusted, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab_img)

    alpha = 0.7 # Reduce contrast
    beta = -30 # Make it a bit darker overall
    L_adjusted = cv2.convertScaleAbs(L, alpha=alpha, beta=beta)

    img_contrast_adjusted = cv2.cvtColor(cv2.merge([L_adjusted, A, B]), cv2.COLOR_LAB2BGR)

    # --- Slight Blurring (Optional, to simulate less sharpness) ---
    img_final = cv2.GaussianBlur(img_contrast_adjusted, (3, 3), 0)

    return img_final

def isolate_green_from_array(img_array):
    """
    Isolates green color components from an image array and makes
    all other parts black.

    Args:
        img_array (numpy.ndarray): The input image array.

    Returns:
        numpy.ndarray: The processed image with only green parts,
                       or None if the input array is None.
    """
    if img_array is None:
        print("Error: Input image array for green isolation is None.")
        return None

    # Convert the image from BGR to HSV color space for better color segmentation
    hsv_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2HSV)

    # --- Define color ranges for Green ---
    # Adjusted lower hue to be more restrictive to remove yellowish tones.
    # These values might need slight tuning based on your specific image's green shades.
    lower_green = np.array([42, 90, 95])
    upper_green = np.array([95, 255, 255])

    # Create a mask for green color
    mask_green = cv2.inRange(hsv_img, lower_green, upper_green)

    # Apply the mask to the original image to get only green parts on a black background
    result_img = cv2.bitwise_and(img_array, img_array, mask=mask_green)

    return result_img

def overlay_and_darken(base_image_path, overlay_image, darken_factor=0.6):
    """
    Overlays an image (assumed to have a black background for non-overlay areas)
    onto another base image, after darkening the base image.

    Args:
        base_image_path (str): Path to the base image (e.g., '4.jpg').
        overlay_image (numpy.ndarray): The image to overlay (e.g., green isolated image).
                                       It's assumed that non-target pixels in this
                                       image are black (0,0,0).
        darken_factor (float): Factor by which to darken the base image (0.0 to 1.0).
                                1.0 means no darkening, 0.0 means completely black.
                                A value like 0.6 will make it 60% of its original brightness.

    Returns:
        numpy.ndarray: The final overlaid image, or None if base image cannot be loaded.
    """
    base_img = cv2.imread(base_image_path)

    if base_img is None:
        print(f"Error: Could not load base image from {base_image_path}")
        return None

    # Resize the base image to match the dimensions of the overlay_image
    # This is crucial for ensuring both images align correctly for overlaying.
    h, w, _ = overlay_image.shape
    base_img_resized = cv2.resize(base_img, (w, h))

    # Darken the resized base image
    # We convert the image to float for accurate multiplication, then back to uint8.
    darkened_base_img = (base_img_resized * darken_factor).astype(np.uint8)

    # Create a mask from the overlay_image:
    # Where the overlay_image has non-black pixels (i.e., the green parts),
    # the mask will be white (255). Otherwise, it will be black (0).
    gray_overlay = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_overlay, 1, 255, cv2.THRESH_BINARY)

    # Invert the mask. This inverted mask will be white where the overlay_image
    # is black (i.e., the background area where we want to keep the darkened base image).
    mask_inv = cv2.bitwise_not(mask)

    # Use the inverted mask to get the background parts from the darkened base image.
    # This effectively "cuts out" the area where the green overlay will go.
    base_background = cv2.bitwise_and(darkened_base_img, darkened_base_img, mask=mask_inv)

    # Use the original mask to get only the green foreground parts from the overlay image.
    overlay_foreground = cv2.bitwise_and(overlay_image, overlay_image, mask=mask)

    # Combine the extracted background from the darkened base image and the
    # green foreground from the overlay image.
    final_image = cv2.add(base_background, overlay_foreground)

    return final_image

# --- Main execution block for the full pipeline ---
# Define the paths to your input images
# 'test_before_train/7.png' is used as the initial image for sharpening (from 1-sharp-image.py)
initial_image_for_sharpening = 'test_before_train/4.png'
# 'test_before_train/4.png' is used as the base image for the final overlay
base_image_for_final_overlay = 'test_before_train/4.png'

# 0. Display original initial images (for reference)
print("Displaying original input images...")
original_initial_sharpen_img = cv2.imread(initial_image_for_sharpening)
if original_initial_sharpen_img is not None:
    cv2.namedWindow('Original Image for Sharpening', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Original Image for Sharpening', 600, 600)
    cv2.imshow('Original Image for Sharpening', original_initial_sharpen_img)
else:
    print(f"Error: Could not load original image for sharpening from {initial_image_for_sharpening}")

original_base_for_overlay_img = cv2.imread(base_image_for_final_overlay)
if original_base_for_overlay_img is not None:
    cv2.namedWindow('Original Base Image for Overlay (4.png)', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Original Base Image for Overlay (4.png)', 600, 600)
    cv2.imshow('Original Base Image for Overlay (4.png)', original_base_for_overlay_img)
else:
    print(f"Error: Could not load original base image for overlay from {base_image_for_final_overlay}")


# 1. Sharpen the initial image
print(f"Step 1: Sharpening {initial_image_for_sharpening}...")
sharpened_img = sharpen_image(initial_image_for_sharpening)
if sharpened_img is None:
    print("Pipeline stopped: Sharpening failed.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit() # Exit if an early step fails

# 2. Apply green tint and contrast adjustments
print("Step 2: Applying green tint and contrast adjustments...")
green_adjusted_img = make_green_and_adjust(sharpened_img)
if green_adjusted_img is None:
    print("Pipeline stopped: Green adjustment failed.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit()

# 3. Isolate green from the adjusted image
print("Step 3: Isolating green color...")
isolated_green_img = isolate_green_from_array(green_adjusted_img)
if isolated_green_img is None:
    print("Pipeline stopped: Green isolation failed.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit()

# 4. Overlay the isolated green onto the darkened base image
print(f"Step 4: Overlaying isolated green onto darkened {base_image_for_final_overlay}...")
# The darken_factor is set to 0.4 to make the base image a bit darker.
final_overlaid_image = overlay_and_darken(base_image_for_final_overlay, isolated_green_img, darken_factor=0.4)

if final_overlaid_image is not None:
    # Display the final combined image
    cv2.namedWindow('Final Pipeline Result', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Final Pipeline Result', 600, 600)
    cv2.imshow('Final Pipeline Result', final_overlaid_image)

    # Save the final image to a file
    output_filename = 'full_pipeline_result.png'
    cv2.imwrite(output_filename, final_overlaid_image)
    print(f"Full pipeline result saved as '{output_filename}'")

    # Wait for a key press and then close all OpenCV windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Failed to create the final overlaid image. Check base image path or dimensions.")
