import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def isolate_red_objects(image_path):
    """
    Isolates red color components from an image and makes
    all other parts black. Specifically tuned for fundus images
    to highlight blood vessels and the optic disc.

    Args:
        image_path (str): The path to the input image.

    Returns:
        numpy.ndarray: The processed image with only red parts,
                       or None if the image cannot be loaded.
    """
    # Load the image
    img = cv2.imread(image_path)

    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return None

    # Convert the image from BGR to HSV color space
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # --- Define color ranges for Red ---
    # Red color in HSV wraps around 0 (or 179 in OpenCV's 0-179 H range).
    # So, we need two ranges: one for lower red hues and one for upper red hues.

    # Lower red hue range
    # Hue: 0-10 (adjust as needed for specific reds)
    # Saturation: Min 50 (to avoid desaturated colors like grays/blacks)
    # Value: Min 50 (to avoid very dark colors)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])

    # Upper red hue range
    # Hue: 160-179 (adjust as needed)
    # Saturation: Min 50
    # Value: Min 50
    lower_red2 = np.array([140, 30, 30])
    upper_red2 = np.array([179, 255, 255])

    # Create masks for both red ranges
    mask_red1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv_img, lower_red2, upper_red2)

    # Combine the two red masks
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    # --- Apply the combined mask to the original image ---
    # Copy only the pixels where the mask_red is white (i.e., red)
    result_img = cv2.bitwise_and(img, img, mask=mask_red)

    # Display the original and the processed image using Matplotlib for better clarity
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # Convert BGR to RGB for matplotlib
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)) # Convert BGR to RGB for matplotlib
    plt.title('Red Objects Isolated')
    plt.axis('off')

    plt.show()

    return result_img

# --- How to use the function ---
image_file = 'output/converted_to_image_jpeg_style.png' # The image you provided

# Ensure the image file exists
if not os.path.exists(image_file):
    print(f"Error: The file '{image_file}' was not found. Please ensure it's in the same directory as the script.")
else:
    processed_image = isolate_red_objects(image_file)

    if processed_image is not None:
        # Define output directory and filename
        output_directory = 'RedIsolated'
        output_filename = 'red_isolated_fundus.png'
        output_path = os.path.join(output_directory, output_filename)

        # Create the output directory if it doesn't exist
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
            print(f"Created directory: {output_directory}")

        # Save the result
        cv2.imwrite(output_path, processed_image)
        print(f"Processed image saved as '{output_path}'")
    else:
        print("Image processing failed.")