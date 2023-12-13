import cv2
import numpy as np

def embed_watermark(original_image, watermark_image, alpha):
    img = cv2.imread(original_image)
    watermark = cv2.imread(watermark_image, cv2.IMREAD_UNCHANGED)

    # Ensure both images have the same number of channels
    if img.shape[2] == 3 and watermark.shape[2] == 4:
        watermark = watermark[:, :, :3]

    # Resize the watermark to a fraction of the original image size
    resized_watermark = cv2.resize(watermark, (int(img.shape[1] * alpha), int(img.shape[0] * alpha)))

    # Get the starting point for placing the watermark
    x_offset = 0
    y_offset = 0

    # Ensure that the watermark fits within the bounds of the original image
    if x_offset + resized_watermark.shape[1] > img.shape[1]:
        x_offset = img.shape[1] - resized_watermark.shape[1]
    if y_offset + resized_watermark.shape[0] > img.shape[0]:
        y_offset = img.shape[0] - resized_watermark.shape[0]

    # Create a region of interest (ROI) for the watermark
    roi = img[y_offset:y_offset + resized_watermark.shape[0], x_offset:x_offset + resized_watermark.shape[1]]

    # Combine the original image and the resized watermark
    result = cv2.addWeighted(roi, 1, resized_watermark, 0.5, 0)

    # Update the original image with the watermarked ROI
    img[y_offset:y_offset + resized_watermark.shape[0], x_offset:x_offset + resized_watermark.shape[1]] = result

    return img

# Example usage
watermarked_image = embed_watermark("monalisa.jpeg", "watermark.jpeg", alpha=0.1)
cv2.imwrite("watermarked_image_visible.jpg", watermarked_image)
