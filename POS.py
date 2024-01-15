"""

This module contains the framework implemented by https://ieeexplore.ieee.org/document/7565547 which is
also known as POS rPPG by other research papers. This is the closest implementation of the original
framework that has been proposed. It also contains the improved version.

"""

from flexible_rPPG.hr_estimator import stft_estimator, welch_estimator, fft_estimator
from flexible_rPPG.sig_extraction_utils import *
from flexible_rPPG.utils import *
from flexible_rPPG.methods import POS
from tqdm import tqdm


class POSImplementations:
    def __init__(self, dataset_name, dataset_dir, implementation='original'):

        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir
        self.implementation = implementation

        self.videos, self.gt_files = get_video_and_gt_files(dataset=self.dataset_name, base_dir=self.dataset_dir)
        if len(self.videos) != len(self.gt_files):
            raise ValueError("The number of videos does not match the number of ground truth files.")

    def simulate(self):
        hrES, hrGT = [], []

        if self.implementation == 'original':
            print(f"Processing {self.dataset_name} dataset using {self.implementation} implementation of POS")
            for i in tqdm(range(len(self.videos))):
                hrES.append(self.pos_original(input_video=self.videos[i], dataset=self.dataset_name))
                hrGT.append(self.pos_ground_truth(ground_truth_file=self.gt_files[i], dataset=self.dataset_name))
                # print(f"{i+1}/{len(self.videos)} videos processed")

        elif self.implementation == 'improved':
            print(f"Processing {self.dataset_name} dataset using {self.implementation} implementation of POS")
            for i in tqdm(range(len(self.videos))):
                hrES.append(self.pos_improved(input_video=self.videos[i], dataset=self.dataset_name))
                hrGT.append(self.pos_ground_truth(ground_truth_file=self.gt_files[i], dataset=self.dataset_name))
                # print(f"{i+1}/{len(self.videos)} videos processed")

        mae, rmse, r = evaluation_metrics(ground_truth_hr=hrGT, estimated_hr=hrES)
        print(f"MAE : {mae} , RMSE : {rmse} , PCC : {r}")

    def pos_original(self, input_video, dataset=None):
        """
        Estimate the heart rate of the input video using the original POS implementation.

        Parameters
        ----------
        input_video : str
            Path to the input video file.
        dataset : str, optional
            Name of the dataset. If provided, it may be used for specialized processing.
            Defaults to None.

        Returns
        -------
        list
            Estimated heart rates of the subject in the input video.
        """

        raw_sig = extract_raw_sig(input_video, ROI_name='POS', ROI_type='None', width=1, height=1, pixel_filtering=True)
        fps = get_fps(input_video, dataset)

        window_size, increment = 1.6, 1 / fps
        window = moving_window(raw_sig, fps=fps, window_size=window_size, increment=increment)

        # Compute PPG Signal
        H = POS(signal=window, fps=fps, increment=increment)

        # Compute STFT
        hrES = stft_estimator(signal=H, fps=fps, remove_outlier=False, signal_length=12, increment=1, mask=[0.67, 4.0])
        hrES = np.mean(hrES)

        return hrES

    def pos_improved(self, input_video, dataset=None):
        """
        Estimate the heart rate of the input video using the improved POS implementation depending on the dataset used.

        Parameters
        ----------
        input_video : str
            Path to the input video file.
        dataset : str, optional
            Name of the dataset. If provided, it may be used for specialized processing.
            Defaults to None.

        Returns
        -------
        list
            Estimated heart rates of the subject in the input video.
        """

        if dataset == 'UBFC2':
            raw_sig = extract_raw_sig(input_video, ROI_name='GREEN', ROI_type='ROI_I', width=1, height=1, pixel_filtering=True)
            fps = get_fps(input_video, dataset)

            window_size, increment = 1.6, 1 / fps
            window = moving_window(raw_sig, fps=fps, window_size=window_size, increment=increment)

            # Compute PPG Signal
            H = POS(signal=window, fps=fps, increment=increment)

            # Post-Filtering
            filtered_H = apply_filters(signal=H, combination=['moving_average_filter'])

            # Compute STFT
            hrES = stft_estimator(signal=filtered_H, fps=fps, remove_outlier=False, signal_length=12, increment=1, mask=[0.67, 4.0])
            hrES = np.mean(hrES)

        elif dataset == 'PURE':
            raw_sig = extract_raw_sig(input_video, ROI_name='ICA', ROI_type='None', width=0.6, height=1, pixel_filtering=False)
            fps = get_fps(input_video, dataset)

            window_size, increment = 1.6, 1 / fps
            window = moving_window(raw_sig, fps=fps, window_size=window_size, increment=increment)

            # Compute PPG Signal
            H = POS(signal=window, fps=fps, increment=increment)

            # Post-Filtering
            filtered_H = apply_filters(signal=H, combination=['detrending_filter', 'moving_average_filter'])

            # Compute PSD
            hrES = fft_estimator(signal=filtered_H, fps=fps, remove_outlier=False, bpm_type='continuous', signal_length=12, increment=1, mask=[0.67, 4.0])
            hrES = np.mean(hrES)

        elif dataset == 'COHFACE':
            raw_sig = extract_raw_sig(input_video, ROI_name='ICA', width=0.6, height=1, pixel_filtering=True)
            fps = get_fps(input_video, dataset)

            window_size, increment = 1.6, 1 / fps
            window = moving_window(raw_sig, fps=fps, window_size=window_size, increment=increment)

            # Pre-Filtering
            filtered_H = apply_filters(signal=window, combination=['detrending_filter'])

            # Compute PPG Signal
            H = POS(signal=filtered_H, fps=fps, increment=increment)

            # Post-Filtering
            filtered_H = apply_filters(signal=H, combination=['moving_average_filter', 'fir_bp_filter'], filtering_params={'low': 0.67, 'high': 4.0, 'fps': fps})

            # Compute FFT
            hrES = fft_estimator(signal=filtered_H, fps=fps, remove_outlier=False, bpm_type='continuous', signal_length=12, increment=1, mask=[0.67, 4.0])
            hrES = np.mean(hrES)

        return hrES

    def pos_ground_truth(self, ground_truth_file=None, dataset=None, gtTrace=None, sampling_frequency=None):
        """
        Obtain the ground truth heart rate of the input video using the POS framework.

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

        Returns
        -------
        list
            Estimated heart rates of the subject in the input video.
        """

        if not gtTrace or not sampling_frequency:
            if not ground_truth_file:
                raise ValueError("Either provide 'ground_truth_file' and 'dataset' name or provide "
                                 "'gtTrace' and 'sampling_frequency'.")
            sampling_frequency, gtTrace = get_ground_truth_ppg_data(ground_truth_file, dataset)

        normalized = normalize(gtTrace, normalize_type='mean_normalization')
        filtered_signals = fir_bp_filter(signal=normalized, fps=sampling_frequency, low=0.67, high=4.0)

        # Compute STFT
        hrGT = stft_estimator(signal=filtered_signals, fps=sampling_frequency, remove_outlier=False, signal_length=12, increment=1, mask=[0.67, 4.0])
        hrGT = np.mean(hrGT)

        return hrGT
