import cv2
import numpy as np

def calculate_ssim(image1, image2):
    # Ensure both images are in the same data type
    image1 = image1.astype(np.float64)
    image2 = image2.astype(np.float64)

    mean1 = np.mean(image1)
    mean2 = np.mean(image2)

    covar = np.cov(image1.flatten(), image2.flatten())[0, 1]
    c1 = (0.01 * 255)**2  # To stabilize the division with weak denominator
    c2 = (0.03 * 255)**2  # To stabilize the division with weak denominator

    print("Mean of Image 1:", mean1)
    print("Mean of Image 2:", mean2)
    print("Covariance:", covar)


    ssim_index = (2 * mean1 * mean2 + c1) * (2 * covar + c2) / ((mean1**2 + mean2**2 + c1) * (np.var(image1) + np.var(image2) + c2))

    return ssim_index

def detect_image_retouching(original_image_path, retouched_image_path, threshold=0.90):
    # Load images in grayscale
    original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
    retouched_image = cv2.imread(retouched_image_path, cv2.IMREAD_GRAYSCALE)

    # Check if images are loaded successfully
    if original_image is None or retouched_image is None:
        print("Error loading images.")
        return

    # Ensure both images have the same size
    h, w = original_image.shape
    retouched_image = cv2.resize(retouched_image, (w, h))

    # Calculate Structural Similarity Index (SSI)
    ssim_index = calculate_ssim(original_image, retouched_image)

    # Set a threshold for detecting retouching
    if ssim_index < threshold:
        print("The image has been retouched.")
    else:
        print("The image appears to be authentic.")

# Example usage
detect_image_retouching("BeforeRetouching.jpg", "AfterRetouching.jpg")
