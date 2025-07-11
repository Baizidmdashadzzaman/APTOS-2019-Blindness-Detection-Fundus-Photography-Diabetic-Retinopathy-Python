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
    # To remove yellowish-green, increase the lower hue bound.
    # Typical pure green hue is around 60. Values below that tend towards yellow.
    # Adjust these values based on your specific image's green shades.
    lower_green = np.array([42, 90, 95])  # Increased hue from 20 to 35-40
    upper_green = np.array([95, 255, 255])

    # Create a mask for green color
    mask_green = cv2.inRange(hsv_img, lower_green, upper_green)

    # --- Apply the mask to the original image ---
    # Create a black image of the same size as the original
    black_background = np.zeros_like(img)

    # Copy only the pixels where the mask_green is white (i.e., green)
    result_img = cv2.bitwise_and(img, img, mask=mask_green)

    # Display the original and the processed image
    cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Original Image', 600, 600)
    cv2.imshow('Original Image', img)

    cv2.namedWindow('Green Isolated', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Green Isolated', 600, 600)
    cv2.imshow('Green Isolated', result_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(0) # Wait indefinitely until a key is pressed
    cv2.destroyAllWindows() # Close all OpenCV windows

    return result_img

# --- How to use the function ---
image_file = 'converted_to_image_jpeg_style.png' # Using the image name from the user's prompt
processed_image = isolate_green(image_file)

if processed_image is not None:
    # You can save the result if needed
    cv2.imwrite('green_isolated_fundus.png', processed_image)
    print("Processed image saved as 'green_isolated_fundus.png'")