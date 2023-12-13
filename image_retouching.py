import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def detect_image_retouching(original_image_path, retouched_image_path, threshold=0.95):
    # Load images
    original_image = cv2.imread(original_image_path)
    retouched_image = cv2.imread(retouched_image_path)

    # Convert images to grayscale
    original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    retouched_gray = cv2.cvtColor(retouched_image, cv2.COLOR_BGR2GRAY)

    # Calculate Structural Similarity Index (SSI)
    ssim_index, _ = ssim(original_gray, retouched_gray, full=True)

    # Set a threshold for detecting retouching
    if ssim_index < threshold:
        print("The image has been retouched.")
    else:
        print("The image appears to be authentic.")

# Example usage
detect_image_retouching("monalisa.jpeg", "watermarked_image_fragile.jpg")
