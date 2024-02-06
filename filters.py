from importlib import import_module
import numpy as np
from scipy.signal import firwin, filtfilt, medfilt, butter
import cv2
from scipy.sparse import spdiags


def apply_filters(signal, combination, filtering_params=None):
    """
    Checks if the signal is a 1D, 2D, or 3D array. Depending on the type, the next function 'apply_filter_to_signal'
    will filter the signal.

    Parameters
    ----------
    signal : list, nparray
        Signal to filter
    combination : tuple
        Input a tuple of different combination of filtering methods. Available algorithms are 'detrending_filter',
        'moving_average_filter', 'butterworth_bp_filter', and 'fir_bp_filter'.
        For example: ('detrending_filter', 'fir_bp_filter')
    filtering_params : dict
        Specify the lower frequency and higher frequency limit to filter the signal on.
        Example: {'low': 0.75, 'high': 4.0}

    Returns
    -------
    nparray
        Returns a filtered signal

    """

    if signal.ndim == 3 or signal.ndim == 2:
        filtered = np.array([apply_filter_to_signal(each_window, combination, filtering_params) for each_window in signal])
    else:  # This is for GREEN and LiCVPR since they don't have signal windowing
        filtered = apply_filter_to_signal(signal, combination, filtering_params)

    return np.array(filtered)


def apply_filter_to_signal(window, combination, filtering_params):
    """
    Applies each type of filter to the whole signal by taking in each windowed signal.
    GREEN and LiCVPR does not have signal windowing so it filters the whole video's signal.

    Parameters
    ----------
    window : list, nparray
        A window of signal to filter
    combination : tuple
        Input a tuple of different combination of filtering methods. Available algorithms are 'detrending_filter',
        'moving_average_filter', 'butterworth_bp_filter', and 'fir_bp_filter'.
        For example: ('detrending_filter', 'fir_bp_filter')
    filtering_params : dict
        Specify the lower frequency and higher frequency limit to filter the signal on.
        Example: {'low': 0.75, 'high': 4.0}

    Returns
    -------
    nparray
        Returns a filtered window signal

    """

    # If the combination is empty, return the signal
    if combination == ():
        return np.array(window)

    # Use each filtering algorithm to filter the input windowed signal
    for each_filter in combination:
        # Import the filter module based on input filter name
        filter_module = import_module('flexible_rPPG.filters')
        filter = getattr(filter_module, each_filter)

        # For butterworth and FIR bandpass filters, specify the higher and lower frequency threshold
        if each_filter == 'butterworth_bp_filter' or each_filter == 'fir_bp_filter':
            window = filter(window, **filtering_params)
        # For other types of filter, just apply the filter
        else:
            window = filter(window)

    return np.array(window)


def fir_bp_filter(signal, fps, low=0.7, high=4.0):
    """
    :param signal:
        Takes in the signal to be bandpass filtered
    :param fps:
        This is the fps of the video file, which is also the sampling frequency
    :param low:
        This is the low frequency level
    :param high:
        This is the high frequency level
    :return:
        Returns the bandpass filtered signal
    """
    signal = np.array(signal)

    numtaps = int(3 * (fps // low))

    # Coefficients of FIR bandpass filter
    filter_coefficients = firwin(numtaps=numtaps, cutoff=[low, high], fs=fps, pass_zero=False, window='hamming')

    # Filtering using the FIR bandpass filter coefficients.
    # Since its FIR bandpass filter, the denominator coefficient is set as 1
    filtered_signal = filtfilt(filter_coefficients, 1, signal, padlen=len(signal)-1, axis=0)  # for filtfilt axis=0 is columns

    return filtered_signal


def butterworth_bp_filter(signal, fps, low=0.8, high=2.0, order=4):
    """
    :param signal:
        Takes in the signal to be bandpass filtered using butterworth
    :param fps:
        This is the fps of the video file, which is also the sampling frequency
    :param low:
        This is the low frequency level
    :param high:
        This is the high frequency level
    :param order:
        Filter order
    :return:
        Returns the bandpass filtered signal
    """

    signal = np.array(signal)

    # Coefficients of butterworth bandpass filter
    b, a = butter(order, Wn=[low, high], fs=fps, btype='bandpass')

    # Filtering using the butterworth bandpass filter coefficients
    filtered_signal = filtfilt(b, a, signal, axis=0)

    return filtered_signal


def detrending_filter(signal, Lambda=300):
    """
    This code is based on the following article "An advanced detrending method with application to HRV analysis".
    Tarvainen et al., IEEE Trans on Biomedical Engineering, 2002.

    :param signal: numpy.ndarray
        The signal where you want to remove the trend. This can be a 1D or 2D array.
    :param Lambda: int
        The smoothing parameter.
    :return filtered_signal: numpy.ndarray
        The detrended signal.
    """

    # Check if the signal is 1D or 2D.
    if len(signal.shape) == 1:
        # The signal is 1D. Process it directly.
        signal_length = signal.shape[0]

        # observation matrix
        H = np.identity(signal_length)

        # second-order difference matrix
        ones = np.ones(signal_length)
        minus_twos = -2 * np.ones(signal_length)
        diags_data = np.array([ones, minus_twos, ones])
        diags_index = np.array([0, 1, 2])
        D = spdiags(diags_data, diags_index, (signal_length - 2), signal_length).toarray()
        filtered_signal = np.dot((H - np.linalg.inv(H + (Lambda ** 2) * np.dot(D.T, D))), signal)
    else:
        # The signal is 2D. Process each column separately.
        n_signals = signal.shape[1]
        filtered_signal = np.empty_like(signal)
        for i in range(n_signals):
            filtered_signal[:, i] = detrending_filter(signal[:, i], Lambda)

    return filtered_signal


def moving_average_filter(signal, window_size=3):
    """
    :param signal:
         Takes in the signal to perform moving average filter on
    :param window_size:
        Window size to perform moving average (number of frames)
    :return:
        Returns moving average filtered signal
    """

    moving_averages = []
    for i in range(len(signal) - window_size + 1):
        moving_averages.append(sum(signal[i:i+window_size])/window_size)

    return np.array(moving_averages)


def simple_skin_selection(frame, lower_rgb=75, higher_rgb=200):
    """
    :param frame:
        Input frames of video
    :param lower_rgb:
        Lower RGB threshold level
    :param higher_rgb:
        Higher RGB threshold level
    :return:
        Returns filtered pixels that lies between given RGB threshold
    """
    lower_rgb_threshold = np.all(frame <= lower_rgb, axis=-1)  # Lower RGB threshold
    higher_rgb_threshold = np.all(frame >= higher_rgb, axis=-1)  # Higher RGB threshold
    indices = np.logical_or(lower_rgb_threshold, higher_rgb_threshold) # Combine these indices

    img_copy = frame.copy()  # Create a copy of the image to not overwrite the original one
    img_copy[indices] = 0  # Change these pixels to black

    return img_copy


def hsv_skin_selection(frame, alpha=0.2, filter_length=5):
    """
    This HSV skin selection algorithm is based on Lee, H., Ko, H., Chung, H., Nam, Y., Hong, S. and Lee, J., 2022.
    Real-time realizable mobile imaging photoplethysmography. Scientific Reports, 12(1), p.7141 which is available at
    https://www.nature.com/articles/s41598-022-11265-x

    :param frame:
        Input frames of video
    :param alpha:
        Constant alpha used to adjust the skin extraction regions
    :param filter_length:
        Median filter length
    :return:
        Returns filtered skin pixels
    """


    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Convert BGR to HSV
    h, s, v = cv2.split(hsv)  # Split HSV image into H, S and V channels
    histogram = cv2.calcHist([s], [0], None, [256], [0, 256])  # Calculate histogram of S channel
    saturation = [x[0] for x in histogram]

    filtered_data = medfilt(saturation, kernel_size=filter_length).tolist()  # Filter using median filter of length 5
    hist_max = filtered_data.index(max(filtered_data))  # Find the frequent saturation value

    # Calculate the threshold range based on alpha and maximum/frequent saturation value
    TH_range = alpha * hist_max
    TH_max = hist_max + TH_range / 2.0
    TH_min = hist_max - TH_range / 2.0

    mask = cv2.inRange(s, TH_min, TH_max)  # create a boolean mask where saturation value is between TH_min and TH_max

    selected_pixels = cv2.bitwise_and(frame, frame, mask=mask)  # apply the mask to the original image

    return selected_pixels
