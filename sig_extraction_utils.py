from flexible_rPPG.filters import *
import mediapipe as mp
# from mediapipe.tasks.python import vision, BaseOptions
from flexible_rPPG.utils import *


def extract_frames_yield(input_video):
    """
    Extract frames from either a list of frame paths or a video file.
    :param input_video: List of frame paths or path to a video file.
    :return: Yields frames.
    """
    if isinstance(input_video, list):  # Check if input_source is a list
        for frame_path in input_video:
            yield cv2.imread(frame_path)
    else:  # Assume input_source is a video file path
        cap = cv2.VideoCapture(input_video)
        success, image = cap.read()
        while success:
            yield image
            success, image = cap.read()
        cap.release()


def extract_raw_sig(input_video, ROI_name=None, ROI_type='None', width=1, height=1, pixel_filtering=True, mode=None, frame_length=None):
    """
    :param input_video:
        This takes in an input video file
    :param ROI_name:
        This is to specify the ROI name. Different frameworks have different ways of extracting raw RGB signal.
        The available types are "CHROM", "POS", "ICA", "GREEN", and "LiCVPR".
    :param ROI:
        Select the region of interest
            - BB: Bounding box of the whole face
            - FH: Forehead box using Viola Jones Face Detector
    :param width:
        Select the width of the detected face bounding box
    :param height:
        Select the height of the detected face bounding box
    :param pixel_filtering:
        Apply a simple skin selection process through pixel filtering
    :return:
        if framework == 'PCA':
            Returns the sum of RGB pixel values of video sequence from the ROI
        elif framework == 'CHROM':

        elif framework == 'ICA':
            Returns the averaged raw RGB signal from the ROI
    """

    if (r'33\2' in input_video) or (r'33\3' in input_video):
        scaleFactor = 1.1
        minSize_x = 50
        minSize_y = 50
    elif 'subject31' in input_video:
        scaleFactor = 1.3
        minSize_x = 30
        minSize_y = 30
    else:
        scaleFactor = 1.1
        minSize_x = 30
        minSize_y = 30

    raw_sig = []

    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                                   min_detection_confidence=0.5)

    file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             'flexible_rPPG', 'Necessary_Files', 'haarcascade_frontalface_default.xml')
    face_cascade = cv2.CascadeClassifier(file_path)

    face_coordinates_prev = None
    mp_coordinates_prev = None
    mask = None
    frame_count = 0
    dl_frame_count = 0

    for frame in extract_frames_yield(input_video):
        frame_count += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5, minSize=(minSize_x, minSize_y), flags=cv2.CASCADE_SCALE_IMAGE)

        # Look through the first 200 frames until face is detected
        if len(faces) == 0 and frame_count <= 200:
            continue

        if (len(faces) == 0 or len(faces) > 1) and face_coordinates_prev is not None:
            x, y, w, h = face_coordinates_prev
            x1 = max(0, int(x + (1 - width) / 2 * w))
            y1 = max(0, int(y + (1 - height) / 2 * h))
            x2 = min(frame.shape[1], int(x + (1 + width) / 2 * w))
            y2 = min(frame.shape[0], int(y + (1 + height) / 2 * h))
            roi = frame[y1:y2, x1:x2]
            dl_frame_count += 1

        else:
            for (x, y, w, h) in faces:
                x1 = max(0, int(x + (1 - width) / 2 * w))
                y1 = max(0, int(y + (1 - height) / 2 * h))
                x2 = min(frame.shape[1], int(x + (1 + width) / 2 * w))
                y2 = min(frame.shape[0], int(y + (1 + height) / 2 * h))
                roi = frame[y1:y2, x1:x2]

                face_coordinates_prev = (x, y, w, h)
                dl_frame_count += 1

        if ROI_name == 'LiCVPR':
            results = mp_face_mesh.process(roi)
            region = roi
        else:
            results = mp_face_mesh.process(frame)
            region = frame

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                if ROI_name == 'GREEN':
                    selected_landmarks = [67, 299, 296, 297, 10]
                elif ROI_name == 'LiCVPR':
                    selected_landmarks = [234, 132, 136, 152, 365, 361, 454, 380, 144]
                else:
                    selected_landmarks = [0]

                selected_coordinates = [
                    (int(landmarks[i].x * region.shape[1]), int(landmarks[i].y * region.shape[0])) for i in
                    selected_landmarks]
                mp_coordinates_prev = selected_coordinates

        else:
            if mp_coordinates_prev is not None:  # Check if mp_coordinates_prev is not None
                selected_coordinates = mp_coordinates_prev
            else:
                continue

        if ROI_name == 'PCA':
            red_values = np.sum(roi[:, :, 2], axis=(0, 1))
            green_values = np.sum(roi[:, :, 1], axis=(0, 1))
            blue_values = np.sum(roi[:, :, 0], axis=(0, 1))
            raw_sig.append([red_values, green_values, blue_values])

        elif ROI_name == 'CHROM' or ROI_name == 'POS' or ROI_name == 'ICA':
            region_of_interest = roi

        elif ROI_name == 'PhysNet':
            resized_roi = cv2.resize(roi, (128, 128))
            # resized_roi = cv2.resize(roi, (224, 224))
            raw_sig.append(resized_roi)
            # if (mode == 'train' and dl_frame_count == frame_length + 1) or (mode == 'test' and dl_frame_count == frame_length + 1):
            #     break
            # if dl_frame_count == frame_length + 1:
            #     break
            continue

        elif ROI_name == 'DeepPhys':
            if ROI_type == 'center-cropped':
                h, w, _ = frame.shape

                # Crop the frame to 492x492
                center_x, center_y = w // 2, h // 2
                start_x = center_x - 492 // 2
                start_y = center_y - 492 // 2
                end_x = start_x + 492
                end_y = start_y + 492
                cropped_frame = frame[start_y:end_y, start_x:end_x]

                downsampled_image = cv2.resize(cropped_frame, (36, 36), interpolation=cv2.INTER_CUBIC)
                raw_sig.append(downsampled_image)
            else:
                # print(dl_frame_count, roi.shape)
                downsampled_image = cv2.resize(roi, (36, 36), interpolation=cv2.INTER_CUBIC)
                raw_sig.append(downsampled_image)

            if (mode == 'train' and dl_frame_count == 1500) or (mode == 'test' and dl_frame_count == 1202):
                break

            continue

        elif ROI_name == 'HRCNN':
            h, w, _ = roi.shape
            input_aspect_ratio = w / h

            im_w_r = 192
            im_h_r = int(192 / input_aspect_ratio)
            img = cv2.resize(roi, (im_w_r, im_h_r))
            y1 = int((im_h_r - 128) / 2)
            y2 = y1 + 128
            resized_face = img[y1:y2, :, :]

            raw_sig.append(resized_face)
            continue

        elif ROI_name == 'LiCVPR':
            d1 = abs(selected_coordinates[0][0] - selected_coordinates[6][0])
            d2 = abs(selected_coordinates[1][0] - selected_coordinates[5][0])
            d3 = abs(selected_coordinates[2][0] - selected_coordinates[4][0])

            d4 = abs(selected_coordinates[8][1] - selected_coordinates[2][1])
            d5 = abs(selected_coordinates[7][1] - selected_coordinates[4][1])
            d6 = abs(selected_coordinates[3][1] - selected_coordinates[6][1])

            extension = [(int(0.05 * d1), 0), (int(0.075 * d2), 0), (int(0.1 * d3), 0), (0, -int(0.075 * d6)),
                         (-int(0.1 * d3), 0), (-int(0.075 * d2), 0), (-int(0.05 * d1), 0), (0, int(0.3 * d4)),
                         (0, int(0.3 * d5))]

            facial_landmark_coordinates = [(x[0] + y[0], x[1] + y[1]) for x, y in zip(selected_coordinates, extension)]

            contour = np.array(facial_landmark_coordinates, dtype=np.int32)
            contour = contour.reshape((-1, 1, 2))

            mask = np.zeros(region.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)

            region_of_interest = cv2.bitwise_and(region, region, mask=mask)

        elif ROI_name == 'GREEN':
            if ROI_type == 'ROI_I':
                x1 = selected_coordinates[0][0]
                y1 = selected_coordinates[4][1]
                x2 = selected_coordinates[3][0]
                distance = abs(selected_coordinates[1][1] - selected_coordinates[2][1])
                y2 = int(selected_coordinates[1][1] + distance * 0.1)

                roi = frame[y1:y2, x1:x2]

            elif ROI_type == 'ROI_II':
                x1 = selected_coordinates[0][0]
                y1 = selected_coordinates[4][1]
                x2 = selected_coordinates[3][0]
                distance = abs(selected_coordinates[1][1] - selected_coordinates[2][1])
                y2 = int(selected_coordinates[1][1] + distance * 0.1)

                x = int((x1 + x2) / 2)
                y = int(abs(y2 - y1) * 0.3 + y1)

                roi = frame[y, x]

            elif ROI_type == 'ROI_III':
                x1_new = int(x2 - (abs(x1 - x2) * 0.1))
                y1_new = y
                x2_new = int(x2 + (abs(x1 - x2) * 0.2))
                y2_new = y + int(h / 2)

                roi = frame[y1_new:y2_new, x1_new:x2_new]

            elif ROI_type == 'ROI_IV':
                h, w, _ = frame.shape
                x1 = int(w * 0.01)
                y1 = int(h * 0.06)
                x2 = int(w * 0.96)
                y2 = int(h * 0.98)
                roi = frame[y1:y2, x1:x2]

            else:
                assert False, "Invalid ROI type for the 'GREEN' framework. Please choose one of the valid ROI " \
                              "types: 'ROI_I', 'ROI_II', 'ROI_III', or 'ROI_IV' "

            region_of_interest = roi

        else:
            assert False, "Invalid framework. Please choose one of the valid available frameworks " \
                          "types: 'PCA', 'CHROM', 'ICA', 'LiCVPR', or 'GREEN' "

        if ROI_type != 'ROI_II':
            if pixel_filtering:
                filtered_roi = simple_skin_selection(region_of_interest, lower_rgb=75, higher_rgb=200)
            else:
                filtered_roi = region_of_interest

            if mask is not None and ROI_name == 'LiCVPR':
                b, g, r, a = cv2.mean(filtered_roi, mask=mask)
            else:
                b, g, r = calculate_mean_rgb(filtered_roi)
        else:
            b, g, r = region_of_interest

        if not np.isnan(np.array([r, g, b])).any():
            raw_sig.append([r, g, b])

    return raw_sig


def extract_raw_bg_signal(input_video, dataset, color='g'):
    """
    Get the frames-per-second (fps) of the input video or based on the dataset name.

    Parameters
    ----------
    input_video : str
        Path to the input video file.
    dataset : str, optional
        Name of the dataset. Defaults to None.
    color : str
        Color channel of the background. Defaults to the green color

    Returns
    -------
    list
        Returns a list of mean background color. Defaults to return the mean green background color
    """

    raw_bg_sig = []

    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'flexible_rPPG', 'Necessary_Files', 'selfie_segmenter_landscape.tflite')

    BaseOptions = mp.tasks.BaseOptions
    ImageSegmenter = mp.tasks.vision.ImageSegmenter
    ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Create an image segmenter instance with the video mode:
    options = ImageSegmenterOptions(base_options=BaseOptions(model_asset_path=model_path),
                                    running_mode=VisionRunningMode.VIDEO, output_category_mask=True)

    frame_counter = 0
    fps = get_fps(input_video, dataset)

    with ImageSegmenter.create_from_options(options) as segmenter:
        for frame in extract_frames_yield(input_video):
            frame_time = int(frame_counter * (1000 / fps))

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            segmented_masks = segmenter.segment_for_video(mp_image, frame_time)
            category_mask = segmented_masks.category_mask
            output = category_mask.numpy_view()

            output_mask_bool = np.where(output == 255, True, False)
            output_frame = np.zeros_like(frame)
            output_frame[output_mask_bool] = frame[output_mask_bool]

            output_mask_uint8 = output_mask_bool.astype(np.uint8)
            b, g, r, a = cv2.mean(frame, mask=output_mask_uint8)
            raw_bg_sig.append([r, g, b])

            frame_counter += 1

    if color == 'g':
        raw_bg_sig = [x[1] for x in raw_bg_sig]
    elif color == 'r':
        raw_bg_sig = [x[0] for x in raw_bg_sig]
    elif color == 'b':
        raw_bg_sig = [x[2] for x in raw_bg_sig]

    return raw_bg_sig

def get_fps(input_video, dataset=None):
    """
    Get the frames-per-second (fps) of the input video or based on the dataset name.

    Parameters
    ----------
    input_video : str
        Path to the input video file.
    dataset : str, optional
        Name of the dataset. Defaults to None.

    Returns
    -------
    int
        FPS value for the video or dataset.

    Raises
    ------
    AssertionError
        If the dataset name is invalid.
    """

    dataset_fps = {'UBFC1': 30, 'UBFC2': 30, 'PURE': 30, 'LGI_PPGI': 25, 'COHFACE': 20}
    if dataset is None:
        cap = cv2.VideoCapture(input_video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
    elif dataset in dataset_fps:
        fps = dataset_fps[dataset]
    else:
        raise ValueError("Invalid dataset name. Please choose one of the valid available datasets "
                         "types: 'UBFC1', 'UBFC2', 'LGI_PPGI', 'COHFACE' or 'PURE'. If you're using your "
                         "own dataset, enter 'None'")

    return fps

