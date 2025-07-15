import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
# Assuming the image file is named '4.jpg' and is in the same directory as your script
image_path = 'test_before_train/7.png'
image = cv2.imread(image_path)

if image is None:
    print(f"Error: Could not load image from {image_path}")
else:
    # Convert the image from BGR to RGB for consistent display with matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Define a sharpening kernel
    # This is a common kernel that enhances edges
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

    # Optionally, save the sharpened image
    cv2.imwrite('sharpened_retina2.jpg', sharpened_image)
    print("Sharpened image saved as 'sharpened_retina.jpg'")