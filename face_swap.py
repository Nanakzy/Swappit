import cv2
import numpy as np

def face_swap(original_image_path, new_face_image_path, output_path):
    # Load the images
    original_image = cv2.imread(original_image_path)
    new_face_image = cv2.imread(new_face_image_path)

    if original_image is None or new_face_image is None:
        return False  # Fail if images can't be loaded

    # Load Haar Cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect face in original image
    gray_original = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    faces_in_original = face_cascade.detectMultiScale(gray_original, 1.3, 5)

    if len(faces_in_original) == 0:
        return False  # No face detected

    (x, y, w, h) = faces_in_original[0]

    # Detect face in new face image
    gray_new_face = cv2.cvtColor(new_face_image, cv2.COLOR_BGR2GRAY)
    faces_in_new_face = face_cascade.detectMultiScale(gray_new_face, 1.3, 5)

    if len(faces_in_new_face) == 0:
        return False  # No face detected in new face image

    (x_new, y_new, w_new, h_new) = faces_in_new_face[0]

    # Extract the face region from the new face image
    extracted_face = new_face_image[y_new:y_new + h_new, x_new:x_new + w_new]

    # Resize the extracted face to fit the detected face in the original image
    resized_face = cv2.resize(extracted_face, (w, h), interpolation=cv2.INTER_CUBIC)

    # Create a mask for seamless blending
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w // 2, h // 2), min(w, h) // 2, (255, 255, 255), -1)

    # Blend the new face into the original image
    result_with_background = cv2.seamlessClone(resized_face, original_image, mask, (x + w // 2, y + h // 2), cv2.NORMAL_CLONE)

    # Save the result
    cv2.imwrite(output_path, result_with_background)
    return True
