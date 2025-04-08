# Import Libraries
import os
import argparse
import uuid
import mediapipe
import cv2
import filetype
import numpy as np
import config
import imutils

# To reduce Mediapipe false results increase the confidence level
MIN_CONFIDENCE_LEVEL = 0.5
EXPANDING_FACTOR = 0.40

TEMPLATE_IMG_NAME = 'aruco0.jpg'
TEMPLATE_IMG = os.path.join(config.basedir, 'static', 'img', TEMPLATE_IMG_NAME)
FRAME_WIDTH = 600
FRAME_HEIGHT = 600

# OPENCV SUPPORTED ARUCO TAGS
ARUCO_DICTIONARY = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50, "DICT_4X4_100": cv2.aruco.DICT_4X4_100, "DICT_4X4_250": cv2.aruco.DICT_4X4_250, "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000, "DICT_5X5_50": cv2.aruco.DICT_5X5_50, "DICT_5X5_100": cv2.aruco.DICT_5X5_100, "DICT_5X5_250": cv2.aruco.DICT_5X5_250, "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000, "DICT_6X6_50": cv2.aruco.DICT_6X6_50, "DICT_6X6_100": cv2.aruco.DICT_6X6_100, "DICT_6X6_250": cv2.aruco.DICT_6X6_250, "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000, "DICT_7X7_50": cv2.aruco.DICT_7X7_50, "DICT_7X7_100": cv2.aruco.DICT_7X7_100, "DICT_7X7_250": cv2.aruco.DICT_7X7_250, "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000, "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL, "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5, "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9, "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10, "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}


def initialize_mediapipe():
    """
    Initializing mediapipe face detection sub-module
    """
    # Enable face detection
    mpFaceDetection = mediapipe.solutions.face_detection.FaceDetection(
        MIN_CONFIDENCE_LEVEL)

    return mpFaceDetection


def load_aruco_template(template_path):
    # Load the template
    template_img = cv2.imread(template_path)
    # Resize the template
    template_img = imutils.resize(
        template_img, width=FRAME_WIDTH, height=FRAME_HEIGHT)
    return template_img


def grab_aruco_markers(template_img):
    # Convert it to grayscale
    gray_template = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
    # grab ARUCO markers
    for (aName, aDict) in ARUCO_DICTIONARY.items():
        aDict = cv2.aruco.Dictionary_get(aDict)
        aParams = cv2.aruco.DetectorParameters_create()
        (markersCorners, markersIDs, rejected) = cv2.aruco.detectMarkers(
            gray_template, dictionary=aDict, parameters=aParams)

        # if any ArUco marker was detected
        if len(markersCorners) > 0:
            print(f'{len(markersCorners)} ARUCO marker(s) detected of type {aName}')
            break
    return template_img, markersCorners, markersIDs, rejected


def order_marker_vertexes(markerVertexes):
    # Order the coordinates of the marker detected box
    # The output returned will be as per the following order:
    # Top left vertex, Bottom Right Vertex ,

    pts = np.array(markerVertexes)
    s = pts.sum(axis=1)
    # Top left vertex will have the smallest sum
    # Bottom right vertex will have the largest sum
    vertexTL, vertexBR = pts[np.argmin(s)], pts[np.argmax(s)]
    # Remove the considered vertexes
    pts = np.delete(pts, [np.argmin(s), np.argmax(s)], axis=0)
    # Top right will have the smallest difference.
    # Bottom left will have the largest difference.
    diff = np.diff(pts, axis=1)
    # print('diff', diff)
    vertexTR, vertexBL = pts[np.argmin(diff)], pts[np.argmax(diff)]
    return vertexTL, vertexBR, vertexTR, vertexBL


def process_aruco_template(template_img, src_img):
    # Constructing the augmented reality visualization
    processed_template_img, markersCorners, markersIDs, rejected = grab_aruco_markers(
        template_img)

    if len(markersCorners) <= 0:
        print('0 ARUCO marker(s) detected...')
        return template_img

    markersIDs = markersIDs.flatten()
    markersVertexes = []

    # Iterate through the Aruco codes and grab their corresponding vertexes
    for i in (markersIDs):
        # The index of the detected id.
        j = np.squeeze(np.where(markersIDs == i))
        # Extract the corners relevant to this index.
        corner = np.squeeze(markersCorners[j])
        # Add the corner to the reference points list
        markersVertexes.append(corner)

    # print('markersVertexes',markersVertexes)

    for markerVertexes in markersVertexes:
        # print('markerVertexes',markerVertexes)
        vertexTL, vertexBR, vertexTR, vertexBL = order_marker_vertexes(
            markerVertexes)
        # TL - Smallest Sum
        # cv2.circle(template_img, (int(vertexTL[0]), int(vertexTL[1])), 5, color=(255, 0, 0), thickness=-1)
        # BR - Largest Sum
        # cv2.circle(template_img, (int(vertexBR[0]), int(vertexBR[1])), 5, color=(255, 255, 255), thickness=-1)
        # TR - Smallest Diff
        # cv2.circle(template_img, (int(vertexTR[0]), int(vertexTR[1])), 5, color=(255, 255, 0), thickness=-1)
        # BL - Largest Diff
        # cv2.circle(template_img, (int(vertexBL[0]), int(vertexBL[1])), 5, color=(0, 0, 0), thickness=-1)

        # Shape the Destination Matrix
        destinationMatrix = [vertexTL, vertexTR, vertexBR, vertexBL]
        destinationMatrix = np.array(destinationMatrix)

        # Grab the spatial dimensions of the source image and define the transform matrix
        # for the source image in the top-left, top-right, bottom-right, bottom-left.
        (src_imgH, src_imgW) = src_img.shape[:2]
        # Shape the source matrix
        # sourceMatrix = np.array([ [0,0],[src_imgW,0],[src_imgW,src_imgH],[0,src_imgH] ])
        sourceMatrix = np.array(
            [[src_imgW, 0], [0, 0], [0, src_imgH], [src_imgW, src_imgH]])

        # Calculate the Homography matrix
        (H, status) = cv2.findHomography(sourceMatrix, destinationMatrix)
        # Grab the spatial dimensions of the template image
        (template_imgH, template_imgW) = template_img.shape[:2]
        # Warp the source image to the destination image based on homography
        warped_img = cv2.warpPerspective(
            src_img, H, (template_imgW, template_imgH))
###################################################################################################
        # Construct a mask representing the region of the warped source image to copy
        # into the template image.
        mask = np.zeros((template_imgH, template_imgW), dtype="uint8")
        cv2.fillConvexPoly(mask, destinationMatrix.astype(
            "int32"), (255, 255, 255), cv2.LINE_AA)

        # Give the wraped image a black border surrounding it.
        rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # Dilate the mask to add black border around the warped source image
        mask = cv2.dilate(mask, rect, iterations=3)
        # mask = cv2.erode(mask, rect, iterations=3)

        # Create a three channel version of the mask by stacking it depth-wise
        # Such that we can copy the wraped source image into the input image.
        maskScaled = mask.copy()/255.0
        maskScaled = np.dstack([maskScaled]*3)

        # Multiply the warped image by the mask give more weight where there are not masked pixels.
        warped_img_masked = cv2.multiply(
            warped_img.astype("float"), maskScaled)
        template_img_masked = cv2.multiply(
            template_img.astype("float"), 1.0-maskScaled)
        # Copy the masked warped image into template mask region.
        output = cv2.add(warped_img_masked, template_img_masked)
        output = output.astype("uint8")

        # To cater for the multiple markers
        template_img = output.copy()

    # cv2.imshow('template_img',template_img)
    # cv2.waitKey(0)
    return template_img
###################################################################################################


def enlarge_bounding_box(x, y, w, h):
    """
    Enlarge the bounding box based on the expanding factor
    """
    # create a larger bounding box with buffer around keypoints
    x1 = int(x - EXPANDING_FACTOR * w)
    w1 = int(w + 2 * EXPANDING_FACTOR * w)
    y1 = int(y - EXPANDING_FACTOR * h)
    h1 = int(h + 2 * EXPANDING_FACTOR * h)
    x1 = 0 if x1 < 0 else x1
    y1 = 0 if y1 < 0 else y1
    # print('x1,y1,w1,h1', x1, y1, w1, h1)
    return x1, y1, w1, h1


def ar_faces(input_path: str, display_output: bool = False):
    """
    Aging the faces detected within an image
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
        x, y, w, h = enlarge_bounding_box(x, y, w, h)
        # Extract the face -> region of interest
        roi_face_color = frame[y:y + h, x:x + w]

        template_img = load_aruco_template(template_path=TEMPLATE_IMG)
        processed_template_img = process_aruco_template(
            template_img, roi_face_color)

        # Save and Output the resulting image
        output_filepath = os.path.join(config.PROCESSED_PATH, str(
            uuid.uuid4().hex)+os.path.splitext(input_path)[1])
        cv2.imwrite(output_filepath, processed_template_img)
        output_item = {'id': 1, 'folder': config.PROCESSED_FOLDER, 'name': os.path.basename(
            output_filepath), 'msg': os.path.basename(output_filepath)}
        output.append(output_item)
        if display_output:
            # Display Image on screen
            # label = "Add Augmenting Reality Effects To Faces"
            cv2.imshow(label, processed_template_img)
            # Mantain output until user presses a key
            cv2.waitKey(0)

    if display_output:
        # Cleanup
        cv2.destroyAllWindows()

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
    ar_faces(input_path=args['input_path'],
             display_output=args['display_output'])
