# Import Libraries
import os
import argparse
import uuid
import mediapipe
import cv2
import filetype
import numpy as np
import config
from scipy import ndimage

# To reduce false results increase the confidence level
MIN_CONFIDENCE_LEVEL = 0.5
GLASSES_TRANSLATION_FACTOR = 0.4
GLASSES_IMG_NAME = ['glasses.png', 'sunglasses.png']
GLASSES_IMG = [os.path.join(config.basedir, 'static', 'img', i)
               for i in GLASSES_IMG_NAME]


def initialize_mediapipe():
    """
    Initializing mediapipe sub-modules
    """
    # Enable the face detection sub-module
    mpFaceDetection = mediapipe.solutions.face_detection.FaceDetection(
        MIN_CONFIDENCE_LEVEL)

    # Enable the face mesh sub-module
    mpFaceModule = mediapipe.solutions.face_mesh

    return mpFaceDetection, mpFaceModule
############################################################################################


def grab_face_coordinates(img, img_landmarks):
    """
    Grab the coordinates of the face landmarks
    """
    face_coordinates = np.array([[int(min([landmark.x * img.shape[1], img.shape[1] - 1])), int(min([landmark.y * img.shape[0], img.shape[0] - 1]))]
                                 for landmark in img_landmarks.landmark]).astype(int)
    face_coordinates[face_coordinates < 0] = 0
    return face_coordinates
##############################################################################################


def resize_img(img, width):
    """
    Resize the image to the width specified
    """
    img_height, img_width, _ = img.shape

    # Ratio of resizing
    ratio = float(width) / img_width
    # Resize image as per the resizing ratio
    resized_img_dimensions = (width, int(img_height*ratio))
    # cv2.INTER_AREA --> To shrink the image.
    resized_img = cv2.resize(
        img, (resized_img_dimensions), interpolation=cv2.INTER_AREA)
    return resized_img


def blend_images(main_img, overlay_img):
    # Get the first 3 channels - colored part
    overlay_bgr = overlay_img[:, :, :3]
    # Devise a mask for the overlay image (frame will be in white and the background in black)
    overlay_mask = overlay_img[:, :, 3:]
    # Invert mask by flipping the black and white colors of the overlay mask
    background_mask = 255 - overlay_mask
    # Normalize by dividing the pixel values by 255.0 in order to keep the values between 0 and 1.
    # Place the background mask on top of the original image
    main_img_part = (main_img * (1/255.0)) * (background_mask * (1/255.0))
    # Gather the overlay colored
    overlay_part = (overlay_bgr * (1/255.0)) * (overlay_mask * (1/255.0))
    # alpha blend the images
    result = cv2.addWeighted(main_img_part, 255.0, overlay_part, 255.0, 0.0)
    return np.uint8(result)


def load_glasses(face_img, glasses_img):

    # Grab face image dimensions
    (face_imgH, face_imgW) = face_img.shape[:2]

    # Load the glasses image with its Alpha values -
    # Read the 4 channels of the transparent image BGRA (BGR for color and A for transparency).
    gl_img = cv2.imread(glasses_img, cv2.IMREAD_UNCHANGED)

    # Resize the glasses image
    gl_img = resize_img(gl_img, width=(face_imgW))
    return gl_img


def find_eye_center(eye_box_coordinates):
    """
    Localize the eye center point
    """
    np_eye_landmarks = np.array(eye_box_coordinates)

    # Devise a baunding rectangle around the coordinates specified
    (x, y, w, h) = cv2.boundingRect(np_eye_landmarks)

    # Calculating the center point of the rectangle
    eye_center = [int(x+w/2), int(y+h/2)]

    return eye_center


def find_left_eye_center(face_coordinates):
    '''
    Get Left eye's center point
    '''
    eye_coordinates = [face_coordinates[158], face_coordinates[160],
                       face_coordinates[144], face_coordinates[153]]
    # Localize the center of the eye
    eye_center = find_eye_center(eye_coordinates)
    return eye_center


def find_right_eye_center(face_coordinates):
    '''
    Get Right eye's center point
    '''
    # Right eye region of interest
    eye_coordinates = [face_coordinates[385], face_coordinates[387],
                       face_coordinates[380], face_coordinates[373]]

    # Localize the center of the eye
    eye_center = find_eye_center(eye_coordinates)
    return eye_center


def try_glasses(input_path: str, display_output: bool = False):
    """
    Try virtual glasses on faces showing within an image
    """
    # Initialize the mediapipe sub-modules
    mpFaceDetection, mpFaceModule = initialize_mediapipe()

    # Read Input Image
    img = cv2.imread(input_path)

    # Preserve a copy of the original
    frame = img.copy()

    # Convert it to rgb format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the rgb frame
    faces = mpFaceDetection.process(rgb_frame)

    output = []
    output_info = []

    # Output a message showing the faces detected
    output_msg = {'msg': "{} face(s) detected.".format(
        len(faces.detections)), 'category': "info"}
    output_info.append(output_msg)

    mpFaceMesh = mpFaceModule.FaceMesh(
        static_image_mode=True, max_num_faces=len(faces.detections), refine_landmarks=True, min_detection_confidence=MIN_CONFIDENCE_LEVEL
    )
    landmarks = mpFaceMesh.process(rgb_frame).multi_face_landmarks

    # Loop over the selected frames
    for glass_idx, glass_img in enumerate(GLASSES_IMG):
        # Loop over the faces detected
        for idx, (face_detected, face_landmarks) in enumerate(zip(faces.detections, landmarks)):

            # Output message
            if glass_idx == 1:
                label = f"Face ID = {(idx + 1)} - Detection Score {int(face_detected.score[0] * 100)}%"
                output_msg = {'msg': label, 'category': "info"}
                output_info.append(output_msg)
                print(output_msg.get('category'), output_msg.get('msg'))

            # Determine the face bounding box
            relativeBoundingBox = face_detected.location_data.relative_bounding_box
            frameHeight, frameWidth, frameChannels = frame.shape
            faceBoundingBox = int(relativeBoundingBox.xmin*frameWidth), int(relativeBoundingBox.ymin*frameHeight), int(
                relativeBoundingBox.width*frameWidth), int(relativeBoundingBox.height*frameHeight)
            x, y, w, h = faceBoundingBox
            # Region of interest in color mode
            roi_face_color = frame[y:y+h, x:x+w]

            # Grab the coordinate of the face landmark points
            face_coordinates = grab_face_coordinates(frame, face_landmarks)
######################
            left_eye_center = find_left_eye_center(face_coordinates)
            right_eye_center = find_right_eye_center(face_coordinates)

            # Calculate a degree of rotation between eyes
            degree = np.rad2deg(np.arctan2(left_eye_center[0] - right_eye_center[0],
                                           left_eye_center[1] - right_eye_center[1]))

#           print('Rotation degree',degree)

            # Find a center point between eyes
            eye_center_x = int((left_eye_center[0] + right_eye_center[0])/2)
            eye_center_y = int((left_eye_center[1] + right_eye_center[1])/2)

            # Glass translation
            glass_trans = int(GLASSES_TRANSLATION_FACTOR * (eye_center_y - y))
            # Load the glasses frame
            gl_img = load_glasses(roi_face_color, glasses_img=glass_img)

            # Rotate the glasses image
            gl_img_rotated = ndimage.rotate(gl_img, (degree+90))

            gh, gw, gc = gl_img_rotated.shape

            g_center_x, g_center_y = x+gw//2, y + gh // 2
            x1 = x - (g_center_x - eye_center_x)
            x2 = x1 + gw
            y1 = y + glass_trans
            y2 = y + gh + glass_trans
#           print('x1',x1,'x2',x2,'eye_center_x', eye_center_x,'g_center_x',g_center_x)
#           cv2.circle(frame, (x1,y1)                  ,15, color=(255, 0, 0), thickness=-1)
#           cv2.circle(frame, (x2,y2)                  ,15, color=(255, 0, 0), thickness=-1)
#           cv2.circle(frame, (g_center_x , g_center_y),15, color=(255, 0, 0), thickness=-1)

            # Crop the region of ineterest
            frame_part = frame[y1:y2, x1:x2]
            # Overlay the rotated glass image over the cropped area of the face image
            blend_glasses = blend_images(frame_part, gl_img_rotated)
            # Restore back the cropped area
            frame[y1:y2, x1:x2] = blend_glasses
######################
        label = f"Trying on {os.path.basename(glass_img)}"

        # Output the processed image
        output_filepath = os.path.join(config.PROCESSED_PATH,
                                       str(uuid.uuid4().hex)+os.path.splitext(input_path)[1])
        cv2.imwrite(output_filepath, frame)
        output_item = {'id': 1, 'folder': config.PROCESSED_FOLDER,
                       'name': os.path.basename(output_filepath), 'msg': label}
        output.append(output_item)

        if display_output:
            # Display Image on screen
            cv2.imshow(label, frame)
            # Mantain output until user presses a key
            cv2.waitKey(0)

        # Re-initialize frame
        frame = img.copy()
    ######################
    if display_output:
        # Cleanup
        cv2.destroyAllWindows()

    mpFaceMesh.close()
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
    try_glasses(input_path=args['input_path'],
                display_output=args['display_output'])
