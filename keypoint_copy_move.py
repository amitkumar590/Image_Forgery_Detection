import cv2
import numpy as np

def detect_copy_move(original_image_path, forged_image_path, threshold=0.9):
    # Load images
    original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
    forged_image = cv2.imread(forged_image_path, cv2.IMREAD_GRAYSCALE)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Find keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(original_image, None)
    kp2, des2 = sift.detectAndCompute(forged_image, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    # FLANN-based matching
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for _ in range(len(matches))]

    # Ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]

    # Apply the mask to filter out outliers
    good_matches = []
    for i, (m, n) in enumerate(matches):
        if matchesMask[i][0] == 1:
            good_matches.append(m)

    # Set a threshold for detecting copy-move
    similarity_ratio = len(good_matches) / len(kp1)
    
    print(f"Number of keypoints in original image: {len(kp1)}")
    print(f"Number of keypoints in forged image: {len(kp2)}")
    print(f"Number of good matches: {len(good_matches)}")
    print(f"Similarity ratio: {similarity_ratio}")

    if similarity_ratio < threshold:
        print("The image has undergone copy-move forgery.")
    else:
        print("The image appears to be authentic.")

# Example usage
detect_copy_move("original.png", "copy_move.png")
