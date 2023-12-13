import cv2
import numpy as np

def embed_invisible_watermark(image_path, watermark_path, alpha, output_path):
    img = cv2.imread(image_path)

    # Convert the image and watermark to YUV color space
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_y, img_u, img_v = cv2.split(img_yuv)

    watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)

    # Resize the watermark to match the dimensions of the image
    resized_watermark = cv2.resize(watermark, (img.shape[1], img.shape[0]))

    # Perform Discrete Cosine Transform (DCT) on luminance channel
    img_y_dct = cv2.dct(np.float32(img_y))
    watermark_dct = cv2.dct(np.float32(resized_watermark))

    # Embed the watermark in the DCT domain (simple example)
    alpha_factor = 0.1  # Adjust this factor based on the desired strength of the watermark
    watermarked_dct = img_y_dct + alpha_factor * watermark_dct

    # Inverse DCT to obtain watermarked luminance channel
    watermarked_y = cv2.idct(np.float32(watermarked_dct))

    # Clip the values to the valid range
    watermarked_y = np.clip(watermarked_y, 0, 255)

    # Convert back to uint8 type
    watermarked_y = np.uint8(watermarked_y)

    # Merge the watermarked luminance channel with the original color channels
    watermarked_img_yuv = cv2.merge([watermarked_y, img_u, img_v])

    # Convert back to BGR color space
    watermarked_img = cv2.cvtColor(watermarked_img_yuv, cv2.COLOR_YUV2BGR)

    # Save the watermarked image
    cv2.imwrite(output_path, watermarked_img)

# Example usage
embed_invisible_watermark("monalisa.jpeg", "watermark.jpeg", alpha=0.1, output_path="watermarked_image_invisible.jpg")
