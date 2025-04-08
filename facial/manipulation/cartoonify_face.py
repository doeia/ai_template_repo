# Import Libraries
import os
import argparse
import uuid
import mediapipe
import cv2
import filetype
import numpy as np
import config

# To reduce Mediapipe false results increase the confidence level
MIN_CONFIDENCE_LEVEL = 0.5


def initialize_mediapipe():
    """
    Initializing mediapipe face detection sub-module
    """
    # Enable face detection
    mpFaceDetection = mediapipe.solutions.face_detection.FaceDetection(
        MIN_CONFIDENCE_LEVEL)

    return mpFaceDetection


def change_color_quantization(img, k):
    """
    Reduce the number of colors in the image using the K-means clustering algorithm
    k -> Determine the number of colors in the output picture
    """
# Defining the input data for clustering
    data = np.float32(img).reshape((-1, 3))
# Defining the criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
# Applying the cv2.kmeans function
    ret, label, center = cv2.kmeans(
        data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(img.shape)
    return result


def cartoonify_img(input_img):
    """
    Cartoonify an input image
    """
    # Convert it to gray scale
    gray_frame = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

    # Transformation Phase
    # Applying median blur to smoothen the image and reduce noise
    # Replace each pixel value with the median value of all pixels in small pixel neighborhood.
    # ksize -> the filter size
    smooth_gray_frame = cv2.medianBlur(src=gray_frame, ksize=5)
    # Applying adaptive thresholding technique to retrieve the edges and highlight them
    # It calculates the threshold for smaller regions of the image
    # If a pixel value is above the threshold value 255 it will be set to 255 otherwise it will be set to 0
    smooth_gray_frame_edges = cv2.adaptiveThreshold(src=smooth_gray_frame, maxValue=255                                                    # ADAPTIVE_THRESH_MEAN_C -> Mean of the neighborhood area.
                                                    # ADAPTIVE_THRESH_GAUSSIAN_C -> Weighted sum of the neighborhood values.
                                                    # Type of threshold applied
                                                    # Size of the neighbourhood area
                                                    # A constant subtracted from the mean of the neighborhood pixels
                                                    , adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C, thresholdType=cv2.THRESH_BINARY, blockSize=9, C=9
                                                    )

    # Image Filtering
    # Applying bilateral filter to remove noise and to keep edge sharp as required.
    # bilateral filter replaces each pixel value with a weighted average of nearby pixel values.
    # d -> diameter of the pixel neighborhood used during filtering (number of pixels around a certain pixel).
    # sigmaColor / sigmaSpace -> Give a sigma effect, make the image look vicious like water paint.
    filtered_frame = cv2.bilateralFilter(src=input_img, d=9                                         # Standard deviation of the filter in color space.
                                         # Standard deviation of the filter in coordinate space.
                                         , sigmaColor=200, sigmaSpace=200
                                         )
    # Giving a cortoon effect by combining the filtered_frame with the smooth_gray_frame_edges
    # Masking edged image with the new image
    cartoon_frame = cv2.bitwise_and(
        filtered_frame, filtered_frame, mask=smooth_gray_frame_edges)
    # Applying color quantization to reduce the number of colors
    cartoon_frame = change_color_quantization(cartoon_frame, 10)
    return cartoon_frame


def cartoonify_face_image(input_path: str, display_output: bool = False):
    """
    Cartoonify the faces detected within an image
    """
    # Initialize mediapipe face detection sub-module
    mpFaceDetection = initialize_mediapipe()

    # Read Input Image
    img = cv2.imread(input_path)

    # Preserve a copy of the original
    frame = img.copy()

    # Convert the image from BGR to RGB format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces using the rgb frame
    faces = mpFaceDetection.process(rgb_frame)

    output = []
    output_info = []

    # Output message the number of faces detected...
    output_msg = {'msg': "{} face(s) detected.".format(
        len(faces.detections)), 'category': "info"}
    output_info.append(output_msg)

    # Loop over the faces detected
    for idx, face_detected in enumerate(faces.detections):
        # Output message
        label = f"Face ID = {(idx+1)} - Detection Score {int(face_detected.score[0]*100)}%"
        output_msg = {'msg': label, 'category': "info"}
        output_info.append(output_msg)
        print(output_msg.get('category'), output_msg.get('msg'))

        # Get the face relative bounding box
        relativeBoundingBox = face_detected.location_data.relative_bounding_box
        frameHeight, frameWidth, frameChannels = frame.shape
        faceBoundingBox = int(relativeBoundingBox.xmin * frameWidth), int(relativeBoundingBox.ymin * frameHeight), int(
            relativeBoundingBox.width * frameWidth), int(relativeBoundingBox.height * frameHeight)
        # Get the coordinates of the face bounding box
        x, y, w, h = faceBoundingBox

        # Extract the face -> region of interest
        roi_face_color = frame[y:y + h, x:x + w]

        # Cartoonify the face area
        cartoon_frame = cartoonify_img(roi_face_color)

        # Store the cartoon face in the image
        frame[y:y + h, x:x + w] = cartoon_frame

    if display_output:
        # Display Image on screen
        label = "Cartoonifying Faces"
        cv2.imshow(label, frame)
        # Mantain output until user presses a key
        cv2.waitKey(0)
        # Cleanup
        cv2.destroyAllWindows()
    # Save and Output the resulting image
    output_filepath = os.path.join(config.PROCESSED_PATH, str(
        uuid.uuid4().hex)+os.path.splitext(input_path)[1])
    cv2.imwrite(output_filepath, frame)
    output_item = {'id': 1, 'folder': config.PROCESSED_FOLDER, 'name': os.path.basename(
        output_filepath), 'msg': os.path.basename(output_filepath)}
    output.append(output_item)
    mpFaceDetection.close()
    return output_info, output


def is_valid_path(path):
    """
    Validates the path inputted and makes sure that is a file of type image
    """
    if not path:
        raise ValueError(f"Invalid Path")
    if os.path.isfile(path) and 'image' in filetype.guess(path).mime:
        return path
    else:
        raise ValueError(f"Invalid Path {path}")


def parse_args():
    """
    Get user command line parameters
    """
    parser = argparse.ArgumentParser(description="Available Options")

    parser.add_argument('-i', '--input_path', dest='input_path', type=is_valid_path,
                        required=True, help="Enter the path of the image file to process")

    parser.add_argument('-d', '--display_output', dest='display_output', default=False,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']), help="Display output on screen")

    args = vars(parser.parse_args())

    # To Display The Command Line Arguments
    print("## Command Arguments #################################################")
    print("\n".join("{}:{}".format(i, j) for i, j in args.items()))
    print("######################################################################")

    return args


if __name__ == '__main__':
    # Parsing command line arguments entered by user
    args = parse_args()
    cartoonify_face_image(
        input_path=args['input_path'], display_output=args['display_output'])
