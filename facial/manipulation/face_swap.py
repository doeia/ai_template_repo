# Import Libraries
import os
import argparse
import uuid
import dlib
import cv2
import filetype
import imutils
import numpy as np
import config

# Landmark's facial detector to estimate the location of 68 coordinates that map the facial points
# in a person's face
FACIAL_LANDMARK_PREDICTOR = os.path.join(
    config.MODELS_PATH, 'shape_predictor_68_face_landmarks.dat')
FRAME_WIDTH = 500
FRAME_HEIGHT = 500
SWAP_FACE_IMAGE = "img10.jpg"  # "W0Te2.jpg" "sIkPO.jpg" #"YEzOe.jpg" #"ZjTty.jpg" #


def initialize_dlib(facial_landmark_predictor: str):
    """
    Initialize dlib's face detetctor (HOG-based) and then create the facial landmark predictor
    """
    print('Loading facial landmark predictor...')
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(FACIAL_LANDMARK_PREDICTOR)

    return detector, predictor


def extract_index_nparray(nparray):
    """
    # Extracting index from a numpy array
    """
    index = None
    for num in nparray[0]:
        index = num
        break
    return index


def load_image(input_path: str):
    # Load image
    img = cv2.imread(input_path)
    # Resize image
    img = imutils.resize(img, width=FRAME_WIDTH, height=FRAME_HEIGHT)
    # Convert image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Converting image to numpy array
    img_np = np.array(img)
    # Create a mask of the image
    img_mask = np.zeros_like(img)
    return {
        "img": img, "img_gray": img_gray, "img_np": img_np, "img_mask": img_mask
    }


def reload_image(img):
    # Resize image
    img = imutils.resize(img, width=FRAME_WIDTH, height=FRAME_HEIGHT)
    # Convert image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Converting image to numpy array
    img_np = np.array(img)
    # Create a mask of the image
    img_mask = np.zeros_like(img)
    return {
        "img": img, "img_gray": img_gray, "img_np": img_np, "img_mask": img_mask
    }


def extract_faces_landmarks(img, detector, predictor):
    # Convert image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the gray scale frame
    faces = detector(img_gray, 0)

    for idx, face in enumerate(faces):
        landmarks = predictor(img_gray, face)
        landmarks_points = []
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points.append((x, y))
        yield {
            "face": face, "landmarks": landmarks_points
        }


def segment_into_triangles(landmarks_points):
    points = np.array(landmarks_points, np.int32)
    convexhull = cv2.convexHull(points)
    # Delaunay triangulation
    rect = cv2.boundingRect(convexhull)
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(landmarks_points)
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)
    indexes_triangles = []
    for t in triangles:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        index_pt1 = np.where((points == pt1).all(axis=1))
        index_pt1 = extract_index_nparray(index_pt1)
        index_pt2 = np.where((points == pt2).all(axis=1))
        index_pt2 = extract_index_nparray(index_pt2)
        index_pt3 = np.where((points == pt3).all(axis=1))
        index_pt3 = extract_index_nparray(index_pt3)
        if index_pt1 is not None and \
                index_pt2 is not None and \
                index_pt3 is not None:
            triangle = [index_pt1, index_pt2, index_pt3]
            indexes_triangles.append(triangle)
    return indexes_triangles


def draw_triangles(src_img, indexes_triangles, landmarks_points):
    img = src_img.copy()
    # Triangulation of both faces
    for triangle_index in indexes_triangles:
        # Triangulation of the first face
        tr1_pt1 = landmarks_points[triangle_index[0]]
        tr1_pt2 = landmarks_points[triangle_index[1]]
        tr1_pt3 = landmarks_points[triangle_index[2]]
        triangle = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)
        cv2.drawContours(img, [triangle], 0, (0, 255, 0), 1)
    return img
####################################################################################################################


def swap_triangles(src_img, src_landmarks_points, src_indexes_triangles, dst_img, dst_landmarks_points):
    """
    Swap triangles between source and destination image
    Seamless clone the destination image
    """
    # Create a mask for the source image
    src_img_space_mask = np.zeros_like(src_img['img_gray'])

    # Create a mask for the destination image
    dst_img_space_mask = np.zeros_like(dst_img['img_gray'])
    height, width, channels = (dst_img['img']).shape
    dst_new_img = np.zeros((height, width, channels), np.uint8)

    # Triangulation of both images
    for idx, triangle_index in enumerate(src_indexes_triangles):

        # Triangulation of the first image
        tr1_pt1 = src_landmarks_points[triangle_index[0]]
        tr1_pt2 = src_landmarks_points[triangle_index[1]]
        tr1_pt3 = src_landmarks_points[triangle_index[2]]
        triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)

        rect1 = cv2.boundingRect(triangle1)
        (x, y, w, h) = rect1
        cropped_triangle = src_img['img'][y: y + h, x: x + w]
        cropped_tr1_mask = np.zeros((h, w), np.uint8)

        points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                           [tr1_pt2[0] - x, tr1_pt2[1] - y],
                           [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)

        cv2.fillConvexPoly(cropped_tr1_mask, points, 255)

        # Lines space of the source image for a graphical representation
        cv2.line(src_img_space_mask, tr1_pt1, tr1_pt2, 255)
        cv2.line(src_img_space_mask, tr1_pt2, tr1_pt3, 255)
        cv2.line(src_img_space_mask, tr1_pt1, tr1_pt3, 255)
        lines_space_src = cv2.bitwise_and(
            src_img['img'], src_img['img'], mask=src_img_space_mask)

        # Triangulation of second image
        tr2_pt1 = dst_landmarks_points[triangle_index[0]]
        tr2_pt2 = dst_landmarks_points[triangle_index[1]]
        tr2_pt3 = dst_landmarks_points[triangle_index[2]]
        triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)

        rect2 = cv2.boundingRect(triangle2)
        (x, y, w, h) = rect2

        cropped_tr2_mask = np.zeros((h, w), np.uint8)

        points2 = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
                            [tr2_pt2[0] - x, tr2_pt2[1] - y],
                            [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)

        cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)

        # Lines space of the destination image for a graphical representation
        cv2.line(dst_img_space_mask, tr2_pt1, tr2_pt2, 255)
        cv2.line(dst_img_space_mask, tr2_pt2, tr2_pt3, 255)
        cv2.line(dst_img_space_mask, tr2_pt1, tr2_pt3, 255)
        lines_space_dst = cv2.bitwise_and(
            dst_img['img'], dst_img['img'], mask=dst_img_space_mask)

        # Warp triangles
        points = np.float32(points)
        points2 = np.float32(points2)
        M = cv2.getAffineTransform(points, points2)
        warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
        warped_triangle = cv2.bitwise_and(
            warped_triangle, warped_triangle, mask=cropped_tr2_mask)

        # Reconstruct destination image
        dst_new_img_rect_area = dst_new_img[y: y + h, x: x + w]
        dst_new_img_rect_area_gray = cv2.cvtColor(
            dst_new_img_rect_area, cv2.COLOR_BGR2GRAY)

        _, mask_triangles_designed = cv2.threshold(
            dst_new_img_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
        warped_triangle = cv2.bitwise_and(
            warped_triangle, warped_triangle, mask=mask_triangles_designed)
        dst_new_img_rect_area = cv2.add(dst_new_img_rect_area, warped_triangle)
        dst_new_img[y: y + h, x: x + w] = dst_new_img_rect_area

    # Swapping by convex hull partioning
    dst_img_mask = np.zeros_like(dst_img['img_gray'])
    dst_points = np.array(dst_landmarks_points, np.int32)
    dst_convexhull = cv2.convexHull(dst_points)
    dst_head_mask = cv2.fillConvexPoly(dst_img_mask, dst_convexhull, 255)
    dst_img_mask = cv2.bitwise_not(dst_head_mask)

    dst_head_noimg = cv2.bitwise_and(
        np.array(dst_img['img']), np.array(dst_img['img']), mask=dst_img_mask)

    result = cv2.add(dst_head_noimg, dst_new_img)

    (x, y, w, h) = cv2.boundingRect(dst_convexhull)
    dst_center_img = (int((x + x + w) / 2), int((y + y + h) / 2))
    seamlessclone = cv2.seamlessClone(
        result, np.array(
            dst_img['img']), dst_head_mask, dst_center_img, cv2.NORMAL_CLONE
    )
    return result, seamlessclone


def refine_img(src_img):
    # Blur the image to remove the noise and remove black pixels
    out = cv2.medianBlur(src=src_img, ksize=3)

    # Image denoising
    out = cv2.fastNlMeansDenoisingColored(
        # Source Image
        src=out  # src_img
        # Output Image
        , dst=None        # Filter strength regulator
        # Bigger value removes noise but also removes image details
        , h=1, hColor=10, templateWindowSize=7, searchWindowSize=21
    )
    # cv2.imshow('Denoised image',out)
    # cv2.waitKey(0)

    # Adjust brightness and contrast
    alpha = 1.0  # Contrast control (1.0 -> 3.0)
    beta = 0  # Brightness control (0 -> 100)
    out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)
    # cv2.imshow('Adjusted image',out)
    # cv2.waitKey(0)
    return out

####################################################################################################################


def swap_faces_dlib(input_path: str, display_output: bool = False):
    """
    Swap faces
    """
    # Initialize dlib face detector using the facial landmark recognition
    detector, predictor = initialize_dlib(
        facial_landmark_predictor=FACIAL_LANDMARK_PREDICTOR)

    # Read Swap Image
    src_img_path = os.path.join(config.UPLOAD_PATH, SWAP_FACE_IMAGE)
    src_img = load_image(src_img_path)

    # Read Input Image
    dst_img = load_image(input_path)

    output = []
    output_info = []

    # Detecting faces in source Swap image (Should be one face)
    for idx, src_face_landmarks in enumerate(extract_faces_landmarks(src_img['img'], detector, predictor)):
        src_face = src_face_landmarks['face']
        src_landmarks = src_face_landmarks['landmarks']
        output_msg = {'msg': "Face {} detected In Source Image on position (Left:{} Top:{} Right:{} Botton:{}).".
                      format((idx+1), src_face.left(), src_face.top(), src_face.right(), src_face.bottom()), 'category': "info"}
        output_info.append(output_msg)
        print(output_msg.get('category'), output_msg.get(
            'msg'), 'landmarks', len(src_landmarks))

        # Delaunay triangulation
        src_indexes_triangles = segment_into_triangles(src_landmarks)
        trg_src_img = draw_triangles(
            src_img['img'], src_indexes_triangles, src_landmarks)

        output_filepath = os.path.join(config.PROCESSED_PATH, str(
            uuid.uuid4().hex) + os.path.splitext(input_path)[1])
        cv2.imwrite(output_filepath, trg_src_img)
        output_item = {'id': 1, 'folder': config.PROCESSED_FOLDER, 'name': os.path.basename(
            output_filepath), 'msg': os.path.basename(output_filepath)}
        output.append(output_item)

        ##############################
        # Detecting faces in source image
        for idx, dst_face_landmarks in enumerate(extract_faces_landmarks(dst_img['img'], detector, predictor)):
            dst_face = dst_face_landmarks['face']
            dst_landmarks = dst_face_landmarks['landmarks']
            output_msg = {'msg': "Face {} detected on position (Left:{} Top:{} Right:{} Botton:{}).".
                          format((idx+1), dst_face.left(), dst_face.top(), dst_face.right(), dst_face.bottom()), 'category': "info"}
            print(output_msg.get('category'), output_msg.get(
                'msg'), 'landmarks', len(dst_landmarks))
##############################
            out_result, out_clone = swap_triangles(
                src_img, src_landmarks, src_indexes_triangles, dst_img, dst_landmarks)

            output_filepath = os.path.join(config.PROCESSED_PATH,
                                           str(uuid.uuid4().hex) + os.path.splitext(input_path)[1])
            cv2.imwrite(output_filepath, out_result)
            output_item = {'id': 2, 'folder': config.PROCESSED_FOLDER, 'name': os.path.basename(
                output_filepath), 'msg': os.path.basename(output_filepath)}
            output.append(output_item)

            out_refined = refine_img(out_clone)

            output_filepath = os.path.join(config.PROCESSED_PATH,
                                           str(uuid.uuid4().hex) + os.path.splitext(input_path)[1])
            cv2.imwrite(output_filepath, out_refined)
            output_item = {'id': 3, 'folder': config.PROCESSED_FOLDER, 'name': os.path.basename(
                output_filepath), 'msg': os.path.basename(output_filepath)}
            output.append(output_item)

            dst_img = reload_image(out_refined)

    if display_output:
        # Display Image on screen
        cv2.imshow('trg_src_img', trg_src_img)
        cv2.imshow('dst_img', dst_img['img'])
        # Mantain output until user presses a key
        cv2.waitKey(0)
        # Cleanup
        cv2.destroyAllWindows()

    return output_info, output
#####################################################################################################################


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
    # swap_faces()
    swap_faces_dlib(input_path=args['input_path'],
                    display_output=args['display_output'])
