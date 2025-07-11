import cv2
import numpy as np

def isolate_green(image_path):
    """
    Isolates green color components from an image and makes
    all other parts black.

    Args:
        image_path (str): The path to the input image.

    Returns:
        numpy.ndarray: The processed image with only green parts,
                       or None if the image cannot be loaded.
    """
    # Load the image
    img = cv2.imread(image_path)

    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return None

    # Convert the image from BGR to HSV color space
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # --- Define color ranges for Green ---
    # Adjusted lower hue to be more restrictive to remove yellowish tones.
    # These values might need slight tuning based on your specific image.
    lower_green = np.array([42, 90, 95])
    upper_green = np.array([95, 255, 255])

    # Create a mask for green color
    mask_green = cv2.inRange(hsv_img, lower_green, upper_green)

    # Apply the mask to the original image to get only green parts on a black background
    result_img = cv2.bitwise_and(img, img, mask=mask_green)

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

# --- Main execution block ---
# Define the paths to your input images
image_to_process = 'converted_to_image_jpeg_style.png' # Updated path
base_image_for_overlay = 'test_before_train/4.png' # Updated path

# Step 1: Isolate the green color from the first image
print(f"Processing image: {image_to_process} to isolate green...")
green_isolated_img = isolate_green(image_to_process)

if green_isolated_img is not None:
    # Display the original base image
    original_base_img = cv2.imread(base_image_for_overlay)
    if original_base_img is not None:
        cv2.namedWindow('Original Base Image (4.png)', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Original Base Image (4.png)', 600, 600)
        cv2.imshow('Original Base Image (4.png)', original_base_img)
    else:
        print(f"Error: Could not load original base image from {base_image_for_overlay}")

    # Step 2: Overlay the green isolated image onto the darkened base image
    print(f"Overlaying green isolated image onto darkened {base_image_for_overlay}...")
    # You can adjust the darken_factor (e.g., 0.5 for more dark, 0.8 for less dark)
    final_overlaid_image = overlay_and_darken(base_image_for_overlay, green_isolated_img, darken_factor=0.4)

    if final_overlaid_image is not None:
        # Display the final combined image
        cv2.namedWindow('Final Overlaid Image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Final Overlaid Image', 600, 600)
        cv2.imshow('Final Overlaid Image', final_overlaid_image)

        # Save the final image to a file
        output_filename = 'final_overlaid_image.png'
        cv2.imwrite(output_filename, final_overlaid_image)
        print(f"Final overlaid image saved as '{output_filename}'")

        # Wait for a key press and then close all OpenCV windows
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Failed to create the final overlaid image. Check base image path or dimensions.")
else:
    print("Failed to isolate green from the initial image. Check image path.")
