import cv2
import numpy as np

def detect_anomaly(reference_image_path, test_image_path, threshold=0.8):
    
    reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
    test_image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)

   
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(reference_image, None)
    kp2, des2 = sift.detectAndCompute(test_image, None)

    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    non_matching_kp1 = [kp1[m[0].queryIdx] for m in matches if m[0] not in good_matches]
    non_matching_kp2 = [kp2[m[0].trainIdx] for m in matches if m[0] not in good_matches]

    
    anomaly_score = 1 - len(good_matches) / len(kp1)

    
    if anomaly_score > threshold:
        print("Anomaly Detected!")
    else:
        print("No Anomaly Detected.")

    
    img_matches = cv2.drawMatches(reference_image, kp1, test_image, kp2, good_matches, None)

    
    resized_reference_image = cv2.resize(reference_image, (400, 300))
    resized_test_image = cv2.resize(test_image, (400, 300))
    resized_img_matches = cv2.resize(img_matches, (800, 600))

   
    cv2.imshow("Resized Reference Image", resized_reference_image)
    cv2.imshow("Resized Test Image", resized_test_image)
    cv2.imshow("Resized Matches", resized_img_matches)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
    img_non_matching = cv2.drawKeypoints(test_image, non_matching_kp2, None, color=(0, 0, 255))

    
    resized_img_non_matching = cv2.resize(img_non_matching, (400, 300))

    cv2.imshow("Resized Non-Matching Keypoints", resized_img_non_matching)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


reference_image_path = "C:\\Users\\NILANJAN\\Pictures\\Screenshots\\Screenshot 2023-12-20 111801.png"
test_image_path = "C:\\Users\\NILANJAN\\Pictures\\Screenshots\\Screenshot 2023-12-20 111801.png"
detect_anomaly(reference_image_path, test_image_path)
