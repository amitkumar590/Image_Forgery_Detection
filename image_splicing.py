import cv2
import numpy as np

def mark_splicing_blocks(image, block_size=1, threshold=0.008):
    # Apply DCT to image
    dct_img = cv2.dct(np.float32(image))

    # Set a block size for analysis
    h, w = image.shape
    block_h, block_w = block_size, block_size

    # Create a copy of the image for marking
    marked_image = image.copy()

    # Iterate over blocks
    for i in range(0, h, block_h):
        for j in range(0, w, block_w):
            # Extract block
            block = dct_img[i:i+block_h, j:j+block_w]

            # Calculate energy of the block
            energy = np.sum(np.abs(block))

            # Set a threshold for detecting splicing
            if energy < threshold:
                # Mark the splicing block with a rectangle
                cv2.rectangle(marked_image, (j, i), (j+block_w, i+block_h), (0, 255, 0), 2)
                print(f"Splicing detected at ({j}, {i}) - Energy: {energy}")

    return marked_image

# Load the image
image_path = "spliced_image.jpg"
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Mark splicing blocks with rectangles
marked_image = mark_splicing_blocks(original_image, block_size=1, threshold=0.008)

# Display or save the marked image
cv2.imshow("Marked Image", marked_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
