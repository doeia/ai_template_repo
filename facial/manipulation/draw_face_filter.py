# Import Libraries
import os
import argparse
import uuid
import cv2
import filetype
import numpy as np
import config
import mediapipe

# To reduce false results increase the confidence level
MIN_CONFIDENCE_LEVEL = 0.5

# Face Filter to apply
FILTER_IMG_NAME = 'TheJoker.jpg'  # 'Theexorcisgirl.png'
FILTER_IMG = os.path.join(config.basedir, 'static', 'filters', FILTER_IMG_NAME)


def initialize_mediapipe():
    """
    Initializing mediapipe sub-modules
    """
    # Enable the face detection sub-module
    mpFaceDetection = mediapipe.solutions.face_detection.FaceDetection(
        MIN_CONFIDENCE_LEVEL)

    # Enables face detection and landmarks identification
    mpFaceModule = mediapipe.solutions.face_mesh

    return mpFaceDetection, mpFaceModule


def read_filter_img(mpFaceMesh):
    """
    Load the face filter to apply
    """
    filter_img = cv2.imread(FILTER_IMG)
    filter_img_height, filter_img_width, filter_img_channels = filter_img.shape
    filter_img_rgb = cv2.cvtColor(filter_img, cv2.COLOR_BGR2RGB)

    # Grab the landmarks relative to the face filter image
    landmarks = mpFaceMesh.process(filter_img_rgb).multi_face_landmarks

    # Gather the coordinates respective to these landmarks
    filter_img_coordinates = np.array(
        [[int(landmark.x * filter_img_width), int(landmark.y * filter_img_height)]
         for landmark in landmarks[0].landmark])
    return filter_img, filter_img_coordinates


def draw_filter(img                     # Source image
                , img_landmarks           # Source image landmarks
                , filter_img              # Face filter image
                , filter_img_coordinates  # Face filter landmarks
                , vertices=config.MEDIAPIPE_VERTICES):
    """
    Draw face filter over the faces of the source image.
    """
    # Create a black mask having the same dimensions as the source image.
    converted_img = np.zeros(img.shape, dtype=np.uint8)

    # Loop over the landmarks of the faces showing in the source image.
    for face_landmarks in img_landmarks:

        # Grab the coordinates of these landmarks points.
        face_coordinates = np.array([[int(min([landmark.x*img.shape[1], img.shape[1]-1])), int(min([landmark.y*img.shape[0], img.shape[0]-1]))]
                                     for landmark in face_landmarks.landmark]).astype(int)
        face_coordinates[face_coordinates < 0] = 0

        # Devise triangles based on the vertices provided
        for idx, triangle_id in enumerate(range(0, len(vertices), 3)):
            # Get triangle vertex coordinates
            corner1_id = vertices[triangle_id][0]
            corner2_id = vertices[triangle_id+1][0]
            corner3_id = vertices[triangle_id+2][0]

            # Grab the coordinates respective to the triangle from the face filter image
            filter_pix_coords = filter_img_coordinates[[
                corner1_id, corner2_id, corner3_id], :]
            # Grab the coordinates respective to the triangle from the sourse image
            face_pix_coords = face_coordinates[[
                corner1_id, corner2_id, corner3_id], :]

            # Crop the triangles areas out from the face filter image and the sourse image.
            ex_x, ex_y, ex_w, ex_h = cv2.boundingRect(filter_pix_coords)
            face_x, face_y, face_w, face_h = cv2.boundingRect(face_pix_coords)
            cropped_filter = filter_img[ex_y:ex_y+ex_h, ex_x:ex_x+ex_w]
            cropped_face = img[face_y:face_y+face_h, face_x:face_x+face_w]

            # Update the triangle coordinates for the cropped image
            filter_pix_crop_coords = filter_pix_coords.copy()
            face_pix_crop_coords = face_pix_coords.copy()
            filter_pix_crop_coords[:, 0] -= ex_x
            filter_pix_crop_coords[:, 1] -= ex_y
            face_pix_crop_coords[:, 0] -= face_x
            face_pix_crop_coords[:, 1] -= face_y

            # Get the mask for the triangle in the cropped face image
            cropped_face_mask = np.zeros((face_h, face_w), np.uint8)
            triangle = (np.round(np.array([face_pix_crop_coords]))).astype(int)
            cv2.fillConvexPoly(cropped_face_mask, triangle, 255)

            # Warp cropped filter triangle into the cropped face triangle
            warp_mat = cv2.getAffineTransform(filter_pix_crop_coords.astype(
                np.float32), face_pix_crop_coords.astype(np.float32))
            warped_triangle = cv2.warpAffine(
                cropped_filter, warp_mat, (face_w, face_h))
            warped_triangle = cv2.bitwise_and(
                warped_triangle, warped_triangle, mask=cropped_face_mask)

            # Put the warped triangle into the destination image
            cropped_new_face = converted_img[face_y:face_y +
                                             face_h, face_x:face_x+face_w]
            cropped_new_face_gray = cv2.cvtColor(
                cropped_new_face, cv2.COLOR_BGR2GRAY)
            _, non_filled_mask = cv2.threshold(
                cropped_new_face_gray, 1, 255, cv2.THRESH_BINARY_INV)
            warped_triangle = cv2.bitwise_and(
                warped_triangle, warped_triangle, mask=non_filled_mask)
            cropped_new_face = cv2.add(cropped_new_face, warped_triangle)
            converted_img[face_y:face_y+face_h,
                          face_x:face_x+face_w] = cropped_new_face

        upd_img = img.copy()
        # Add the mask to the original image
        converted_image_gray = cv2.cvtColor(converted_img, cv2.COLOR_BGR2GRAY)
        _, non_drawn_mask = cv2.threshold(
            converted_image_gray, 1, 255, cv2.THRESH_BINARY_INV)
        upd_img = cv2.bitwise_and(upd_img, upd_img, mask=non_drawn_mask)
        filtered_img = cv2.add(converted_img, upd_img)
        ##############################################################################################
        smoothened_img = smooth_filtered_image(
            img, face_coordinates, filtered_img=converted_img)

        # To cater for multiple faces in the same image
        img = smoothened_img.copy()
        ##############################################################################################
    return img, converted_img, filtered_img


def smooth_filtered_image(img, face_coordinates, filtered_img):
    """
    Smooth the resulting image by applying the seamless cloning.
    """
    convexhull = cv2.convexHull(face_coordinates)
    (x, y, w, h) = cv2.boundingRect(convexhull)
    center_face = (int((x + x + w) / 2), int((y + y + h) / 2))

    converted_image_mask = cv2.fillConvexPoly(
        np.zeros_like(filtered_img), convexhull, (255, 255, 255))

    out = cv2.seamlessClone(
        filtered_img, img, converted_image_mask, center_face, cv2.MIXED_CLONE  # cv2.NORMAL_CLONE
    )

    # Blur the image to remove the noise and remove black pixels
    out = cv2.medianBlur(src=out, ksize=3)

    return out
    ##############################################################################################


def apply_face_filters(input_path: str, display_output: bool = False):
    """
    Apply Face Filter on faces showing in a digital image.
    """
    # Initialize mediapipe module
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
    # Call the face mesh sub-module
    mpFaceMesh = mpFaceModule.FaceMesh(
        static_image_mode=True, max_num_faces=len(faces.detections), refine_landmarks=True, min_detection_confidence=MIN_CONFIDENCE_LEVEL
    )
    # Loading the face filter image
    filter_img, filter_img_coordinates = read_filter_img(mpFaceMesh)
    # Grabbing the landmarks of the original image
    landmarks = mpFaceMesh.process(rgb_frame).multi_face_landmarks
    # Drawing the filter
    refined_img, masked_img, filtered_img = draw_filter(
        img=frame, img_landmarks=landmarks, filter_img=filter_img, filter_img_coordinates=filter_img_coordinates, vertices=config.MEDIAPIPE_VERTICES)
    # Outputs
    label = f'Filter Applied and Smoothened {FILTER_IMG_NAME}'
    output_filepath = os.path.join(config.PROCESSED_PATH, str(
        uuid.uuid4().hex)+os.path.splitext(input_path)[1])
    cv2.imwrite(output_filepath, refined_img)
    output_item = {'id': 1, 'folder': config.PROCESSED_FOLDER,
                   'name': os.path.basename(output_filepath), 'msg': label}
    output.append(output_item)

    label = f'Filter Applied {FILTER_IMG_NAME}'
    output_filepath = os.path.join(config.PROCESSED_PATH, str(
        uuid.uuid4().hex)+os.path.splitext(input_path)[1])
    cv2.imwrite(output_filepath, filtered_img)
    output_item = {'id': 2, 'folder': config.PROCESSED_FOLDER,
                   'name': os.path.basename(output_filepath), 'msg': label}
    output.append(output_item)

    label = 'Mask Image'
    output_filepath = os.path.join(config.PROCESSED_PATH, str(
        uuid.uuid4().hex)+os.path.splitext(input_path)[1])
    cv2.imwrite(output_filepath, masked_img)
    output_item = {'id': 3, 'folder': config.PROCESSED_FOLDER,
                   'name': os.path.basename(output_filepath), 'msg': label}
    output.append(output_item)

    output_item = {'id': 4, 'folder': 'filters',
                   'name': FILTER_IMG_NAME, 'msg': FILTER_IMG_NAME}
    output.append(output_item)

    mpFaceMesh.close()
    mpFaceDetection.close()
    if display_output:
        # Display Image on screen
        cv2.imshow('Masked Image', masked_img)
        # Mantain output until user presses a key
        cv2.waitKey(0)

        # Display Image on screen
        cv2.imshow('Image After Filter', filtered_img)
        # Mantain output until user presses a key
        cv2.waitKey(0)

        # Display Image on screen
        cv2.imshow('Image Smoothened', refined_img)
        # Mantain output until user presses a key
        cv2.waitKey(0)

        # Cleanup
        cv2.destroyAllWindows()

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
    apply_face_filters(
        input_path=args['input_path'], display_output=args['display_output'])
