import cv2
import numpy as np
import matplotlib.pyplot as plt
import os # Import the os module

# Load the image
image_path = '../test_before_train/4.png'
image = cv2.imread(image_path)

if image is None:
    print(f"Error: Could not load image from {image_path}")
else:
    # Convert the image from BGR to RGB for consistent display with matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Define a sharpening kernel
    sharpening_kernel = np.array([[-1, -1, -1],
                                  [-1,  10, -1],
                                  [-1, -1, -1]])

    # Apply the sharpening kernel to the image
    sharpened_image = cv2.filter2D(image, -1, sharpening_kernel)

    # Convert the sharpened image to RGB for consistent display
    sharpened_image_rgb = cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2RGB)

    # Display the original and sharpened images
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(sharpened_image_rgb)
    plt.title('Sharpened Image')
    plt.axis('off')

    plt.show()

    # --- Start of modifications for saving ---
    output_directory = 'output'
    output_filename = 'sharpened_retina.jpg'
    output_path = os.path.join(output_directory, output_filename)

    # Create the directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Created directory: {output_directory}")

    # Save the sharpened image
    success = cv2.imwrite(output_path, sharpened_image)
    if success:
        print(f"Sharpened image saved successfully as '{output_path}'")
    else:
        print(f"Error: Could not save sharpened image to '{output_path}'")
    # --- End of modifications for saving ---