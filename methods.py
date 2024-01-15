from scipy.stats import cumfreq

# from flexible_rPPG.LiCVPR import rectify_illumination, nlms, non_rigid_motion_elimination
from flexible_rPPG.filters import *
from flexible_rPPG.utils import *
from scipy.signal import windows
from flexible_rPPG.Necessary_Files.jadeR import jadeR


def CHROM(signal, fps, **params):

    w, l, c = signal.shape
    N = int(l + (w - 1) * (params['increment'] * fps))
    H = np.zeros(N)

    for enum, each_window in enumerate(signal):
        normalized = normalize(signal=each_window, normalize_type='mean_normalization')  # Normalize each windowed segment

        # Build two orthogonal chrominance signals
        Xs = 3 * normalized[0] - 2 * normalized[1]
        Ys = 1.5 * normalized[0] + normalized[1] - 1.5 * normalized[2]

        # Stack signals and apply the bandpass filter
        stacked_signals = np.stack([Xs, Ys], axis=-1)
        filtered_signals = fir_bp_filter(signal=stacked_signals, fps=fps, low=0.67, high=4.0)
        Xf, Yf = filtered_signals[:, 0], filtered_signals[:, 1]

        if np.std(Yf) != 0:
            alpha = np.std(Xf) / np.std(Yf)
        else:
            alpha = 0
        S = Xf - alpha * Yf

        SWin = np.multiply(S, windows.hann(len(S)))

        start = enum * (l // 2)
        end = enum * (l // 2) + l

        H[start:end] = H[start:end] + SWin

    return H


def POS(signal, fps, **params):

    w, l, c = signal.shape
    N = int(l + (w - 1) * (params['increment'] * fps))
    H = np.zeros(N)

    for enum, each_window in enumerate(signal):
        normalized = normalize(signal=each_window, normalize_type='mean_normalization')  # Normalize each windowed segment

        # Projection
        S1 = normalized[1] - normalized[2]
        S2 = normalized[1] + normalized[2] - 2 * normalized[0]

        if np.std(S2) != 0:
            alpha = np.std(S1) / np.std(S2)
        else:
            alpha = 0

        h = S1 + alpha * S2

        start = enum
        end = enum + l

        H[start:end] += (h - np.mean(h))

    return H


def ICA(signal, component=1):

    bvps = []
    for each_window in signal:
        normalized = normalize(each_window, normalize_type='zero_mean_unit_variance')

        # Apply JADE ICA algorithm and select the second component
        W = jadeR(normalized, m=3)
        bvp = np.array(np.dot(W, normalized))
        bvp = bvp[component].flatten()
        bvps.append(bvp)

    return np.array(bvps)


def GREEN(signal):

    return np.array(signal)


def LiCVPR(signal, bg_signal, fps):

    if len(signal) != len(bg_signal):
        bg_signal = bg_signal[abs(len(signal)-len(bg_signal)):]

    # Apply the Illumination Rectification filter
    g_ir = rectify_illumination(face_color=signal, bg_color=np.array(bg_signal))
    motion_eliminated = non_rigid_motion_elimination(signal=g_ir.tolist(), segment_length=1, fps=fps, threshold=0.05)

    return motion_eliminated

def rectify_illumination(face_color, bg_color, step=0.003, length=3):
    """performs illumination rectification.

    The correction is made on the face green values using the background green values,
    to remove global illumination variations in the face green color signal.

    Parameters
    ----------
    face_color: numpy.ndarray
      The mean green value of the face across the video sequence.
    bg_color: numpy.ndarray
      The mean green value of the background across the video sequence.
    step: float
      Step size in the filter's weight adaptation.
    length: int
      Length of the filter.

    Returns
    -------
    rectified color: numpy.ndarray
      The mean green values of the face, corrected for illumination variations.

    """
    # first pass to find the filter coefficients
    # - y: filtered signal
    # - e: error (aka difference between face and background)
    # - w: filter coefficient(s)
    yg, eg, wg = nlms(bg_color, face_color, length, step)

    # second pass to actually filter the signal, using previous weights as initial conditions
    # the second pass just filters the signal and does NOT update the weights !
    yg2, eg2, wg2 = nlms(bg_color, face_color, length, step, initCoeffs=wg, adapt=False)
    return eg2


def nlms(signal, desired_signal, n_filter_taps, step, initCoeffs=None, adapt=True):
    """Normalized least mean square filter.

    Based on adaptfilt 0.2:  https://pypi.python.org/pypi/adaptfilt/0.2

    Parameters
    ----------
    signal: numpy.ndarray
      The signal to be filtered.
    desired_signal: numpy.ndarray
      The target signal.
    n_filter_taps: int
      The number of filter taps (related to the filter order).
    step: float
      Adaptation step for the filter weights.
    initCoeffs: numpy.ndarray
      Initial values for the weights. Defaults to zero.
    adapt: bool
      If True, adapt the filter weights. If False, only filters.

    Returns
    -------
    y: numpy.ndarray
      The filtered signal.

    e: numpy.ndarray
      The error signal (difference between filtered and desired)

    w: numpy.ndarray
      The found weights of the filter.

    """
    eps = 0.001
    number_of_iterations = len(signal) - n_filter_taps + 1
    if initCoeffs is None:
        initCoeffs = np.zeros(n_filter_taps)

    # Initialization
    y = np.zeros(number_of_iterations)  # Filter output
    e = np.zeros(number_of_iterations)  # Error signal
    w = initCoeffs  # Initial filter coeffs

    # Perform filtering
    errors = []
    for n in range(number_of_iterations):
        x = np.flipud(signal[n:(n + n_filter_taps)])  # Slice to get view of M latest datapoints
        y[n] = np.dot(x, w)
        e[n] = desired_signal[n + n_filter_taps - 1] - y[n]
        errors.append(e[n])

        if adapt:
            normFactor = 1. / (np.dot(x, x) + eps)
            w = w + step * normFactor * x * e[n]
            y[n] = np.dot(x, w)

    return y, e, w


def non_rigid_motion_elimination(signal, segment_length, fps, threshold=0.05):
    """
    :param signal:
        Input signal to segment
    :param segment_length:
        The length of each segment in seconds (s)
    :param fps:
        The frame rate of the video
    :param threshold:
        The cutoff threshold of the segments based on their standard deviation
    :return:
        Returns motion eliminated signal
    """

    # Divide the signal into m segments of the same length
    segments = []
    for i in range(0, len(signal), int(segment_length * fps)):
        end = i + int(segment_length * fps)
        if end > len(signal):
            end_segment_index = i
            break
        segments.append(signal[i:end])
    else:
        end_segment_index = len(segments) * fps

    sd = np.array([np.std(segment) for segment in segments])  # Find the standard deviation of each segment

    # calculate the cumulative frequency of the data, which is effectively the CDF
    # 'numbins' should be set to the number of unique standard deviations
    a = cumfreq(sd, numbins=len(np.unique(sd)))

    # get the value that is the cut-off for the top 5% which is done by finding the smallest standard deviation that
    # has a cumulative frequency greater than 95% of the data
    cut_off_index = np.argmax(a.cumcount >= len(sd) * (1 - threshold))
    cut_off_value = a.lowerlimit + np.linspace(0, a.binsize * a.cumcount.size, a.cumcount.size)[cut_off_index]

    # create a mask where True indicates the value is less than the cut-off
    mask = sd < cut_off_value

    # get the new list of segments excluding the top 5% of highest SD
    segments_95_percent = np.concatenate((np.array(segments)[mask]), axis=None)

    # Add residual signal (leftover original signal) due to segmentation if there is any
    if len(signal) != end_segment_index:
        residual_signal = np.array(signal[end_segment_index:len(signal)])
        motion_eliminated = np.concatenate((segments_95_percent, residual_signal), axis=None)
    else:
        motion_eliminated = segments_95_percent

    return motion_eliminated
