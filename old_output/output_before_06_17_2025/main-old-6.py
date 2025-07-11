import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv2.imread('old/4.png')  # Replace with your image path
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Preprocessing: Resize & CLAHE
img_resized = cv2.resize(img_rgb, (512, 512))
lab = cv2.cvtColor(img_resized, cv2.COLOR_RGB2LAB)
l, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
l_clahe = clahe.apply(l)
lab_clahe = cv2.merge((l_clahe, a, b))
img_prep = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

# Image decomposition (simulate bright and dark)
img_bri = cv2.convertScaleAbs(img_prep, alpha=1.2, beta=30)
img_dark = cv2.convertScaleAbs(img_prep, alpha=0.6, beta=-30)

# Simulated FCN output (edge detection)
def get_feature_map(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return cv2.applyColorMap(edges, cv2.COLORMAP_JET)

M_bri = get_feature_map(img_bri)
M_dark = get_feature_map(img_dark)
M_prep = get_feature_map(img_prep)

# Simulated attention (Gaussian mask)
def apply_attention(img, center=(256, 256), radius=100):
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.circle(mask, center, radius, (255), -1)
    mask = cv2.GaussianBlur(mask, (101, 101), 0)
    att = cv2.addWeighted(img, 0.6, cv2.merge([mask]*3), 0.4, 0)
    return att

M_bri_att = apply_attention(M_bri)
M_dark_att = apply_attention(M_dark)

# Final fusion (simulated)
fused = cv2.addWeighted(M_bri_att, 0.5, M_dark_att, 0.5, 0)

# Show results
titles = ['Bright', 'Dark', 'Preprocessed', 'Att-Bri', 'Att-Dark', 'Fused']
images = [img_bri, img_dark, img_prep, M_bri_att, M_dark_att, fused]

plt.figure(figsize=(12, 8))
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.savefig("output_pipeline.png")
