import cv2
import numpy as np

def add_fragile_watermark(image_path, output_path):
    img = cv2.imread(image_path)

    # Add a simple and easily detectable fragile watermark
    watermark = np.zeros_like(img)
    watermark[:, :, 0] = 0  # Blue channel
    watermark[:, :, 1] = 255  # Green channel
    watermark[:, :, 2] = 0  # Red channel

    # Blend the original image and the watermark
    result = cv2.addWeighted(img, 1, watermark, 0.5, 0)

    cv2.imwrite(output_path, result)

# Example usage
add_fragile_watermark("monalisa.jpeg", "watermarked_image_fragile.jpg")
