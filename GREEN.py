"""

This module contains the framework implemented by https://opg.optica.org/oe/fulltext.cfm?uri=oe-16-26-21434&id=175396
also known as GREEN rPPG by other research papers. This is the closest implementation of the original
framework that has been proposed.

"""

from flexible_rPPG.hr_estimator import fft_estimator, welch_estimator
from flexible_rPPG.sig_extraction_utils import *
from flexible_rPPG.utils import *
from flexible_rPPG.filters import *
from tqdm import tqdm


class GREENImplementations:
    def __init__(self, dataset_name, dataset_dir, implementation='original'):

        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir
        self.implementation = implementation

        if dataset_name is not None and dataset_dir is not None:
            self.videos, self.gt_files = get_video_and_gt_files(dataset=self.dataset_name, base_dir=self.dataset_dir)
            if len(self.videos) != len(self.gt_files):
                raise ValueError("The number of videos does not match the number of ground truth files.")

    def simulate(self):
        hrES, hrGT = [], []

        if self.implementation == 'original':
            print(f"Processing {self.dataset_name} dataset using {self.implementation} implementation of GREEN")
            for i in tqdm(range(len(self.videos))):
                hrES.append(self.green_original(input_video=self.videos[i], dataset=self.dataset_name))
                hrGT.append(self.green_ground_truth(ground_truth_file=self.gt_files[i], dataset=self.dataset_name))
                # print(f"{i + 1}/{len(self.videos)} videos processed")

        elif self.implementation == 'improved':
            print(f"Processing {self.dataset_name} dataset using {self.implementation} implementation of GREEN")
            for i in tqdm(range(len(self.videos))):
                hrES.append(self.green_improved(input_video=self.videos[i], dataset=self.dataset_name))
                hrGT.append(self.green_ground_truth(ground_truth_file=self.gt_files[i], dataset=self.dataset_name))
                # print(f"{i + 1}/{len(self.videos)} videos processed")

        mae, rmse, r = evaluation_metrics(ground_truth_hr=hrGT, estimated_hr=hrES)
        print(f"MAE : {mae} , RMSE : {rmse} , PCC : {r}")

    def green_original(self, input_video, roi_type='ROI_I', signal='bp', dataset=None):
        """
        Estimate the heart rate of the input video using the original GREEN implementation

        Parameters
        ----------
        input_video : str
            Path to the input video file.
        roi_type : str
            Select the type of ROI to extract the green channel signal from:
                - 'ROI_I': forehead bounding box
                - 'ROI_II': single pixel in the forehead ROI
                - 'ROI_III': beside head (This doesn't include any skin pixels, only the background)
                - 'ROI_IV': whole frame scaled down by a fraction
        signal : str
            Select the type of signal to extract the heart rate from:
                - 'raw': PV_raw(t) no processing other than spatial averaging over ROI
                - 'ac': PV_AC(t) the mean over time of PV_raw(t) is subtracted (= PV_raw(t) minus DC)
                - 'bp': PV_BP(t) band-pass filtered PV_raw(t) signal. For the band-pass (BP) filter Butterworth coefficients
                        (4th order) were used in a phase with the specified lower and higher frequencies.
        dataset : str, optional
            Name of the dataset. If provided, it may be used for specialized processing.
            Defaults to None.

        Returns
        -------
        float
            Estimated heart rates of the subject in the input video.
        """

        raw_sig = extract_raw_sig(input_video, ROI_name='GREEN', ROI_type=roi_type, width=1, height=1, pixel_filtering=False)
        raw_sig = np.array(raw_sig)[:, 1]  # Select the green channel
        fps = get_fps(input_video, dataset)

        # Process the signals
        pv_raw = raw_sig
        pv_ac = normalize(pv_raw, normalize_type='zero_mean')
        pv_bp = butterworth_bp_filter(pv_raw, fps=fps, low=0.8, high=2.0)

        # Create a dictionary to map signal types to their processed signals
        signal_dict = {'raw': pv_raw, 'ac': pv_ac, 'bp': pv_bp}

        # Check if the signal type is valid
        if signal not in signal_dict:
            raise ValueError("Invalid signal type for the 'GREEN' framework. Please choose one of the valid signals "
                             "types: 'raw' (for raw G signal), 'ac' (removed DC component), or 'bp' (bandpass filtered "
                             "raw signal with the specified lower and higher frequency)")

        # Get the processed signal and its FFT using the dictionary
        pv = signal_dict[signal]

        # Compute FFT
        hrES = fft_estimator(signal=pv, fps=fps, remove_outlier=False, bpm_type='average', signal_length=None, increment=None, mask=[0.8, 2.0])

        return hrES

    def green_improved(self, input_video, dataset):
        """
        Estimate the heart rate of the input video using the imporved GREEN implementation depending on dataset used

        Parameters
        ----------
        input_video : str
            Path to the input video file.
        roi_type : str
            Select the type of ROI to extract the green channel signal from:
                - 'ROI_I': forehead bounding box
                - 'ROI_II': single pixel in the forehead ROI
                - 'ROI_III': beside head (This doesn't include any skin pixels, only the background)
                - 'ROI_IV': whole frame scaled down by a fraction
        signal : str
            Select the type of signal to extract the heart rate from:
                - 'raw': PV_raw(t) no processing other than spatial averaging over ROI
                - 'ac': PV_AC(t) the mean over time of PV_raw(t) is subtracted (= PV_raw(t) minus DC)
                - 'bp': PV_BP(t) band-pass filtered PV_raw(t) signal. For the band-pass (BP) filter Butterworth coefficients
                        (4th order) were used in a phase with the specified lower and higher frequencies.
        dataset : str, optional
            Name of the dataset. If provided, it may be used for specialized processing.
            Defaults to None.

        Returns
        -------
        float
            Estimated heart rates of the subject in the input video.
        """

        if dataset == 'UBFC2':
            raw_sig = extract_raw_sig(input_video, ROI_name='LiCVPR', ROI_type='None', width=1, height=1, pixel_filtering=False)
            raw_sig = np.array(raw_sig)[:, 1]  # Select the green channel
            fps = get_fps(input_video, dataset)

            # Pre-Filtering
            pre_filtered = apply_filters(signal=raw_sig, combination=['detrending_filter', 'fir_bp_filter'], filtering_params={'low': 0.8, 'high': 2.0, 'fps': fps})

            # Pre-Filtering
            post_filtered = apply_filters(signal=pre_filtered, combination=['detrending_filter', 'moving_average_filter'])

            # Compute FFT
            hrES = fft_estimator(signal=post_filtered, fps=fps, remove_outlier=True, bpm_type='average', signal_length=None, increment=None, mask=[0.8, 2.0])

        elif dataset == 'PURE':
            raw_sig = extract_raw_sig(input_video, ROI_name='LiCVPR', ROI_type='None', width=1, height=1, pixel_filtering=False)
            raw_sig = np.array(raw_sig)[:, 1]  # Select the green channel
            fps = get_fps(input_video, dataset)

            # Pre-Filtering
            pre_filtered = apply_filters(signal=raw_sig, combination=['butterworth_bp_filter'], filtering_params={'low': 0.8, 'high': 2.0, 'fps': fps})

            # Post-Filtering
            post_filtered = apply_filters(signal=pre_filtered, combination=['detrending_filter', 'butterworth_bp_filter'], filtering_params={'low': 0.8, 'high': 2.0, 'fps': fps})

            # Compute FFT
            hrES = fft_estimator(signal=post_filtered, fps=fps, remove_outlier=True, bpm_type='average', signal_length=None, increment=None, mask=[0.8, 2.0])

        elif dataset == 'COHFACE':
            raw_sig = extract_raw_sig(input_video, ROI_name='GREEN', ROI_type='ROI_I', width=1, height=1, pixel_filtering=False)
            raw_sig = np.array(raw_sig)[:, 1]  # Select the green channel
            fps = get_fps(input_video, dataset)

            # Pre-Filtering
            pre_filtered = apply_filters(signal=raw_sig, combination=['moving_average_filter', 'butterworth_bp_filter'], filtering_params={'low': 0.8, 'high': 2.0, 'fps': fps})

            # Post-Filtering
            post_filtered = apply_filters(signal=pre_filtered, combination=['moving_average_filter', 'butterworth_bp_filter'], filtering_params={'low': 0.8, 'high': 2.0, 'fps': fps})

            # Compute FFT
            hrES = welch_estimator(signal=post_filtered, fps=fps, remove_outlier=True, bpm_type='average', signal_length=None, increment=None, mask=[0.8, 2.0])

        return hrES

    def green_ground_truth(self, ground_truth_file=None, dataset=None, gtTrace=None, sampling_frequency=None, signal='bp'):
        """
        Obtain the ground truth heart rate of the input video using the GREEN framework.

        Parameters
        ----------
        ground_truth_file : str, optional
            Path to the ground truth file. Required if gtTrace and sampling_frequency are not provided.
        dataset : str, optional
            Name of the dataset. Required if you want to process the dataset based on the given ground_truth_file
            Defaults to None.
        gtTrace : list, optional
            PPG signal that you want processed. Required if ground_truth_file is not provided.
        sampling_frequency : int, optional
            sampling frequency of the PPG signal. Required if ground_truth_file is not provided.
        signal : str
            Select the type of signal to extract the heart rate from:
                - 'raw': PV_raw(t) no processing other than using the given PPG signal
                - 'ac': PV_AC(t) the mean over time of PV_raw(t) is subtracted (= PV_raw(t) minus DC)
                - 'bp': PV_BP(t) band-pass filtered PV_raw(t) signal. For the band-pass (BP) filter Butterworth coefficients
                        (4th order) were used in a phase with the specified lower and higher frequencies.

        Returns
        -------
        float
            Estimated heart rates of the subject in the input video.
        """

        if not gtTrace or not sampling_frequency:
            if not ground_truth_file:
                raise ValueError("Either provide 'ground_truth_file' and 'dataset' name or provide "
                                 "'gtTrace' and 'sampling_frequency'.")
            sampling_frequency, gtTrace = get_ground_truth_ppg_data(ground_truth_file, dataset)

        # Process the signals
        pv_raw = gtTrace
        pv_ac = normalize(pv_raw, normalize_type='zero_mean')
        pv_bp = butterworth_bp_filter(pv_raw, fps=sampling_frequency, low=0.8, high=2.0)

        # Create a dictionary to map signal types to their processed signals
        signal_dict = {'raw': pv_raw, 'ac': pv_ac, 'bp': pv_bp}

        # Check if the signal type is valid
        if signal not in signal_dict:
            raise ValueError("Invalid signal type for the 'GREEN' framework. Please choose one of the valid signals "
                             "types: 'raw' (for raw G signal), 'ac' (removed DC component), or 'bp' (bandpass filtered "
                             "raw signal with the specified lower and higher frequency)")

        # Get the processed signal and its FFT using the dictionary
        pv = signal_dict[signal]

        # Compute FFT
        hrGT = fft_estimator(signal=pv, fps=sampling_frequency, remove_outlier=False, bpm_type='average', signal_length=None, increment=None, mask=[0.8, 2.0])

        return hrGT
