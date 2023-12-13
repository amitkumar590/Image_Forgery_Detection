import cv2
import numpy as np

def block_based_copy_move_detection(image_path, block_size=128, threshold=0.9):
    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Define block size and step
    block_size_x = block_size
    block_size_y = block_size
    step = block_size // 2

    # Iterate over blocks
    for y in range(0, img.shape[0] - block_size_y, step):
        for x in range(0, img.shape[1] - block_size_x, step):
            # Extract block
            block = img[y:y+block_size_y, x:x+block_size_x]

            # Search for similar blocks in the image
            matches = cv2.matchTemplate(img, block, cv2.TM_CCOEFF_NORMED)

            # Find the location of maximum correlation
            _, _, _, max_loc = cv2.minMaxLoc(matches)

            # Check if correlation is above the threshold
            if matches[max_loc[1], max_loc[0]] > threshold:
                print(f"Copy-move forgery detected at ({x}, {y}) - ({x+block_size_x}, {y+block_size_y})")

# Example usage
block_based_copy_move_detection("copy_move.png")
