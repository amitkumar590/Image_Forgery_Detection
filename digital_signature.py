import cv2
import hashlib

def embed_digital_signature(image_path, output_path="watermarked_image_signature.jpg"):
    # Read the image
    img = cv2.imread(image_path)

    # Convert the image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Generate a digital signature using SHA-256
    signature = hashlib.sha256(img_gray.tobytes()).hexdigest()

    # Embed the signature in the bottom part of the image
    img[-30:, 10: img.shape[1] - 10] = 255  # Set the region to white
    cv2.putText(img, f"Digital Signature: {signature}", (20, img.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # Save the watermarked image
    cv2.imwrite(output_path, img)

def verify_digital_signature(image_path):
    # Read the image
    img = cv2.imread(image_path)

    # Extract the region of interest where the digital signature is embedded
    roi = img[-30:, 10: img.shape[1] - 10]

    # Convert the region of interest to grayscale
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Compute the hash of the region of interest using SHA-256
    extracted_signature = hashlib.sha256(roi_gray.tobytes()).hexdigest()

    if not extracted_signature:
        print("Digital signature not found.")
        return False

    # Verify the extracted signature matches the embedded signature
    embedded_signature = hashlib.sha256(roi_gray.tobytes()).hexdigest()

    print(embedded_signature)
    print(extracted_signature)
    if embedded_signature == extracted_signature:
        print("Digital signature is valid.")
        return True
    else:
        print("Digital signature is invalid.")
        return False

# Example usage
embed_digital_signature("monalisa.jpeg", output_path="watermarked_image_signature.jpg")
verify_digital_signature("watermarked_image_signature.jpg")
