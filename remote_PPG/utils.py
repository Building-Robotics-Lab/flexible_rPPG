import numpy as np
from itertools import combinations
import json
import h5py
import pandas as pd
import os

from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error


def normalize(signal, normalize_type=None):
    """
    :param signal:
        Input signal to normalize
    :param normalize_type:
        Normalize signal using one of three types:
            - Mean normalization: Dividing signal by its mean value
            - Zero mean: Subtracting signal by its mean value
            - Zero mean with unit variance: This is also known as standardization
    :return:
        Normalized signal in [[R] [G] [B]] format
    """
    signal = np.array(signal)
    mean = np.mean(signal, axis=0)
    std_dev = np.std(signal, axis=0)

    if normalize_type == 'mean_normalization':
        normalized_signal = np.where(mean != 0, signal / mean, 0)
    elif normalize_type == 'zero_mean':
        normalized_signal = signal - mean
    elif normalize_type == 'zero_mean_unit_variance':
        normalized_signal = np.where(std_dev != 0, (signal - mean) / std_dev, 1e-9)
    else:
        assert False, "Invalid normalization type. Please choose one of the valid available types " \
                      "Types: 'mean_normalization', 'zero_mean', or 'zero_mean_unit_variance' "

    # Turn normalized signal to [[R] [G] [B]]
    if normalized_signal.ndim == 2:
        normalized = np.array([normalized_signal[:, i] for i in range(0, 3)])
    else:
        normalized = np.array(normalized_signal)
    return normalized


def moving_window(sig, fps, window_size, increment):
    """
    :param sig:
        RGB signal
    :param fps:
        Frame rate of the video file (number of frames per second)
    :param window_size:
        Select the window size in seconds (s)
    :param increment:
        Select amount to be incremented in seconds (s)
    :return:
        returns the windowed signal
    """

    windowed_sig = []
    for i in range(0, len(sig), int(increment * fps)):
        end = i + int(window_size * fps)
        if end > len(sig):
            # windowed_sig.append(sig[len(sig) - int(window_size * fps):len(sig)])
            break
        windowed_sig.append(sig[i:end])

    return np.array(windowed_sig)


def get_filtering_combinations(filtering_methods):
    """
    :param filtering_methods:
        Enter a list of filtering methods you want to apply to get unique combinations
    :return:
        Returns a list of unique combinations of different types of filters
    """
    unique_combinations = []

    # Generate combinations of different lengths (0 to 4)
    for r in range(0, len(filtering_methods) + 1):
        # Generate combinations
        for combo in combinations(filtering_methods, r):
            # Check the condition of not having Butterworth and FIRWIN in the same line
            if not ('butterworth_bp_filter' in combo and 'fir_bp_filter' in combo):
                # Add the combination to the list
                unique_combinations.append(combo)

    return unique_combinations


def calculate_mean_rgb(frame):
    """
    Calculates the mean RGB values for each frame

    Parameters
    ----------
    frame : numpy.ndarray
        Input frame of the video

    Returns
    -------
    numpy.ndarray
        The mean RGB values of the non-black pixels in the input frame.
    """
    # Find the indices of the non-black pixels
    non_black_pixels = np.all(frame != [0, 0, 0], axis=-1)

    # Get the non-black pixels
    non_black_pixels_frame = frame[non_black_pixels]

    # Calculate and return the mean RGB values
    return np.mean(non_black_pixels_frame, axis=0)


def get_ground_truth_ppg_data(ground_truth_file, dataset):
    """
    Extract ground truth ppg signal and sampling frequency from dataset.

    Parameters
    ----------
    ground_truth_file : str
        Path to the ground truth file.
    dataset : str
        Name of the dataset.

    Returns
    -------
    tuple
        Returns the ground truth PPG signal of video and the sampling frequency

    Raises
    ------
    AssertionError
        If the dataset name is invalid.
    """

    if dataset == 'UBFC1':
        sampling_frequency = 60
        gtdata = pd.read_csv(ground_truth_file, header=None)
        gtTrace = gtdata.iloc[:, 3].tolist()
        gtTime = (gtdata.iloc[:, 0] / 1000).tolist()
        gtHR = gtdata.iloc[:, 1]

    elif dataset == 'UBFC2':
        sampling_frequency = 30
        gtdata = pd.read_csv(ground_truth_file, delimiter='\t', header=None)
        gtTrace = [float(item) for item in gtdata.iloc[0, 0].split(' ') if item != '']
        gtTime = [float(item) for item in gtdata.iloc[2, 0].split(' ') if item != '']
        gtHR = [float(item) for item in gtdata.iloc[1, 0].split(' ') if item != '']

    elif dataset == 'LGI_PPGI':
        sampling_frequency = 60
        gtdata = pd.read_xml(ground_truth_file)
        gtTrace = gtdata.iloc[:, 2].tolist()
        gtTime = (gtdata.iloc[:, 0]).tolist()
        gtHR = gtdata.iloc[:, 1].tolist()

    elif dataset == 'PURE':
        sampling_frequency = 60
        with open(ground_truth_file) as f:
            data = json.load(f)
        gtTrace = [gtdata["Value"]["waveform"] for gtdata in data['/FullPackage']]
        gtTime = [gtdata["Timestamp"] for gtdata in data['/FullPackage']]
        gtHR = [gtdata["Value"]["pulseRate"] for gtdata in data['/FullPackage']]

    elif dataset == 'COHFACE':
        sampling_frequency = 256
        with h5py.File(ground_truth_file, "r") as f:
            data = {key: list(f[key][:]) for key in f.keys()}
        gtTrace = data['pulse']
        gtTime = data['time']
    else:
        raise ValueError("Invalid dataset name. Please choose one of the valid available datasets "
                         "types: 'UBFC1', 'UBFC2', 'LGI_PPGI', 'COHFACE', or 'PURE'")

    return sampling_frequency, gtTrace


def get_video_and_gt_files(dataset, base_dir):
    """
    Extract the videos and ground truth files from dataset.

    Parameters
    ----------
    dataset : str
        Name of the dataset.
    base_dir : str
        Path to the dataset.

    Returns
    -------
    tuple
        Returns a list of videos and ground truth files

    Raises
    ------
    AssertionError
        If the dataset name is invalid or path is invalid.
    """

    if not os.path.exists(base_dir):
        raise ValueError(f"Invalid path provided: {base_dir}. Please check the path and try again.")

    video_files = []
    gt_files = []

    if dataset == "UBFC1":
        for folders in os.listdir(base_dir):
            subjects = os.path.join(base_dir, folders)
            for each_subject in os.listdir(subjects):
                if each_subject.endswith('.avi'):
                    video_files.append(os.path.join(subjects, each_subject))
                elif each_subject.endswith('.xmp'):
                    gt_files.append(os.path.join(subjects, each_subject))

    elif dataset == "UBFC2":
        for folders in os.listdir(base_dir):
            subjects = os.path.join(base_dir, folders)
            for each_subject in os.listdir(subjects):
                if each_subject.endswith('.avi'):
                    video_files.append(os.path.join(subjects, each_subject))
                elif each_subject.endswith('.txt'):
                    gt_files.append(os.path.join(subjects, each_subject))

    elif dataset == 'LGI_PPGI':
        for sub_folders in os.listdir(base_dir):
            for folders in os.listdir(os.path.join(base_dir, sub_folders)):
                subjects = os.path.join(base_dir, sub_folders, folders)
                for each_subject in os.listdir(subjects):
                    if each_subject.endswith('.avi'):
                        video_files.append(os.path.join(subjects, each_subject))
                    elif each_subject.endswith('cms50_stream_handler.xml'):
                        gt_files.append(os.path.join(subjects, each_subject))

    elif dataset == "PURE":
        subjects = ["{:02d}".format(i) for i in range(1, 11)]
        setups = ["{:02d}".format(i) for i in range(1, 7)]
        for each_setup in setups:
            for each_subject in subjects:
                if f"{each_subject}-{each_setup}" != "06-02":
                    dir = os.listdir(os.path.join(base_dir, f"{each_subject}-{each_setup}"))
                    vid = os.path.join(base_dir, f"{each_subject}-{each_setup}", dir[0])
                    video_files.append([os.path.join(vid, x) for x in os.listdir(vid)])
                    gt_files.append(os.path.join(base_dir, f"{each_subject}-{each_setup}", dir[1]))

    elif dataset == "COHFACE":
        subjects = [f"{i}" for i in range(1, 41)]
        conditions = [f"{i}" for i in range(0, 4)]
        for each_subject in subjects:
            for each_condition in conditions:
                dir = os.listdir(os.path.join(base_dir, each_subject, each_condition))
                for each_file in dir:
                    if each_file.endswith('.avi'):
                        video_files.append(os.path.join(base_dir, each_subject, each_condition, each_file))
                    elif each_file.endswith('.hdf5'):
                        gt_files.append(os.path.join(base_dir, each_subject, each_condition, each_file))

    else:
        raise ValueError("Invalid dataset name. Please choose one of the valid available datasets "
                         "types: 'UBFC1', 'UBFC2', 'LGI_PPGI', 'COHFACE', or 'PURE'")

    return video_files, gt_files


def evaluation_metrics(ground_truth_hr, estimated_hr):
    """
    Extract the videos and ground truth files from dataset.

    Parameters
    ----------
    ground_truth_hr : list
        A list of the ground truth heart rate
    estimated_hr : list
        A list of the estimated heart rate using rPPG

    Returns
    -------
    Tuple[float, float, float]
        A tuple containing the Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Pearson correlation
        coefficient (r) between the ground truth and estimated heart rates.
    """

    mae = mean_absolute_error(ground_truth_hr, estimated_hr)
    rmse = mean_squared_error(ground_truth_hr, estimated_hr, squared=False)
    r, _ = pearsonr(ground_truth_hr, estimated_hr)

    return round(mae, 2), round(rmse, 2), round(r, 2)
