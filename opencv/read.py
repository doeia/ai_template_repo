import cv2

img = cv2.imread("download.jpg")

if img is None:
    print("Error: Unable to load image. Please check the file path.")
else:
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
