"""

This module contains the framework implemented by
https://openaccess.thecvf.com/content_cvpr_2014/papers/Li_Remote_Heart_Rate_2014_CVPR_paper.pdf
also known as LiCVPR rPPG by other research papers. This is the closest implementation of the original
framework that has been proposed. This also includes the improved implementations

"""

from flexible_rPPG.hr_estimator import welch_estimator, fft_estimator
from flexible_rPPG.sig_extraction_utils import *
from flexible_rPPG.utils import *
from flexible_rPPG.filters import *
import numpy as np
from flexible_rPPG.methods import LiCVPR
from tqdm import tqdm


class LiCVPRImplementations:
    def __init__(self, dataset_name, dataset_dir, implementation='original'):

        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir
        self.implementation = implementation
        self.raw_bg_signal = []

        self.videos, self.gt_files = get_video_and_gt_files(dataset=self.dataset_name, base_dir=self.dataset_dir)
        if len(self.videos) != len(self.gt_files):
            raise ValueError("The number of videos does not match the number of ground truth files.")

    def extract_raw_bg_signals(self):
        print(f"Extracting background green signal from {self.dataset_name} dataset")
        for i in tqdm(range(len(self.videos))):
            raw_bg_signal = extract_raw_bg_signal(input_video=self.videos[i], dataset=self.dataset_name, color='g')
            self.raw_bg_signal.append(raw_bg_signal)

    def simulate(self):
        hrES, hrGT = [], []

        if self.implementation == 'original':
            print(f"Processing {self.dataset_name} dataset using {self.implementation} implementation of LiCVPR")
            for i in tqdm(range(len(self.videos))):
                # raw_bg_signal = extract_raw_bg_signal(input_video=self.videos[i], dataset=self.dataset_name, color='g')
                hrES.append(self.licvpr_original(input_video=self.videos[i], dataset=self.dataset_name, raw_bg_green_signal=self.raw_bg_signal[i]))
                hrGT.append(self.licvpr_ground_truth(ground_truth_file=self.gt_files[i], dataset=self.dataset_name))
                # print(f"{i + 1}/{len(self.videos)} videos processed")

        elif self.implementation == 'improved':
            print(f"Processing {self.dataset_name} dataset using {self.implementation} implementation of LiCVPR")
            for i in tqdm(range(len(self.videos))):
                # raw_bg_signal = extract_raw_bg_signal(input_video=self.videos[i], dataset=self.dataset_name, color='g')
                hrES.append(self.licvpr_improved(input_video=self.videos[i], dataset=self.dataset_name, raw_bg_green_signal=self.raw_bg_signal[i]))
                hrGT.append(self.licvpr_ground_truth(ground_truth_file=self.gt_files[i], dataset=self.dataset_name))
                # print(f"{i + 1}/{len(self.videos)} videos processed")

        mae, rmse, r = evaluation_metrics(ground_truth_hr=hrGT, estimated_hr=hrES)
        print(f"MAE : {mae} , RMSE : {rmse} , PCC : {r}")

    def licvpr_original(self, input_video, raw_bg_green_signal, heart_rate_calculation_mode='continuous',
                        hr_interval=None, dataset=None):
        """
        Estimate the heart rate of the input video using the LiCVPR framework

        Parameters
        ----------
        input_video : str
            Path to the input video file.
        raw_bg_green_signal : list
            Raw green background signal. This is extracted using mediapipe selfie segmentation
        heart_rate_calculation_mode : str
            The mode of heart rate calculation to be used. It can be set to one of the following:
            - 'average': The function computes the average heart rate over the entire duration of the video.
            - 'continuous': The function computes the heart rate at regular specified intervals throughout the video.
            Defaults to 'continuous'
        hr_interval : int, optional
            This parameter is used when 'heart_rate_calculation_mode' is set to 'continuous'. It specifies the time interval
            (in seconds) at which the heart rate is calculated throughout the video. If not set, a default interval of
            10 seconds is used.
            Defaults to 10 if None
        dataset : str, optional
            Name of the dataset. If provided, it may be used for specialized processing.
            Defaults to None.

        Returns
        -------
        hr : list
            Returns the estimated heart rate of the input video based on LiCVPR framework

        """

        if hr_interval is None:
            hr_interval = 10

        raw_green_sig = extract_raw_sig(input_video, ROI_name='LiCVPR', ROI_type='None', width=1, height=1,
                                        pixel_filtering=False)  # Get the raw green signal
        raw_green_sig = np.array(raw_green_sig)[:, 1]
        fps = get_fps(input_video, dataset)

        # Compute PPG Signal
        bvp = LiCVPR(signal=raw_green_sig, bg_signal=raw_bg_green_signal, fps=fps)

        # Filter the signal using detrending, moving average and bandpass filter
        filtered_bvp = apply_filters(signal=bvp, combination=['detrending_filter', 'moving_average_filter',
                                                              'butterworth_bp_filter'],
                                     filtering_params={'low': 0.7, 'high': 4.0, 'fps': fps})

        # Compute PSD
        hrES = welch_estimator(signal=filtered_bvp, fps=fps, remove_outlier=False, bpm_type=heart_rate_calculation_mode,
                               signal_length=hr_interval, increment=hr_interval, mask=[0.7, 4.0])
        hrES = np.mean(hrES)

        return hrES

    def licvpr_improved(self, input_video, raw_bg_green_signal, heart_rate_calculation_mode='continuous',
                        hr_interval=None, dataset=None):
        """
        Estimate the heart rate of the input video using the LiCVPR framework

        Parameters
        ----------
        input_video : str
            Path to the input video file.
        raw_bg_green_signal : list
            Raw green background signal. This is extracted using mediapipe selfie segmentation
        heart_rate_calculation_mode : str
            The mode of heart rate calculation to be used. It can be set to one of the following:
            - 'average': The function computes the average heart rate over the entire duration of the video.
            - 'continuous': The function computes the heart rate at regular specified intervals throughout the video.
            Defaults to 'continuous'
        hr_interval : int, optional
            This parameter is used when 'heart_rate_calculation_mode' is set to 'continuous'. It specifies the time interval
            (in seconds) at which the heart rate is calculated throughout the video. If not set, a default interval of
            10 seconds is used.
            Defaults to 10 if None
        dataset : str, optional
            Name of the dataset. If provided, it may be used for specialized processing.
            Defaults to None.

        Returns
        -------
        hr : list
            Returns the estimated heart rate of the input video based on LiCVPR framework

        """

        if dataset == 'UBFC2':
            if hr_interval is None:
                hr_interval = 10

            raw_green_sig = extract_raw_sig(input_video, ROI_name='LiCVPR', ROI_type='None', width=1, height=1,
                                            pixel_filtering=False)  # Get the raw green signal
            raw_green_sig = np.array(raw_green_sig)[:, 1]
            fps = get_fps(input_video, dataset)

            # Pre-Filtering
            pre_filtering = apply_filters(signal=raw_green_sig, combination=['detrending_filter', 'fir_bp_filter'],
                                          filtering_params={'low': 0.7, 'high': 4.0, 'fps': fps})

            # Compute PPG Signal
            bvp = LiCVPR(signal=pre_filtering, bg_signal=raw_bg_green_signal, fps=fps)

            # Compute PSD
            hrES = fft_estimator(signal=bvp, fps=fps, remove_outlier=False, bpm_type=heart_rate_calculation_mode,
                                 signal_length=hr_interval, increment=hr_interval, mask=[0.7, 4.0])
            hrES = np.mean(hrES)

        elif dataset == 'PURE':
            if hr_interval is None:
                hr_interval = 10

            raw_green_sig = extract_raw_sig(input_video, ROI_name='ICA', ROI_type='None', width=0.6, height=1,
                                            pixel_filtering=True)  # Get the raw green signal
            raw_green_sig = np.array(raw_green_sig)[:, 1]
            fps = get_fps(input_video, dataset)

            # Pre-Filtering
            pre_filtering = apply_filters(signal=raw_green_sig, combination=['detrending_filter',
                                                                             'butterworth_bp_filter'],
                                          filtering_params={'low': 0.7, 'high': 4.0, 'fps': fps})

            # Compute PPG Signal
            bvp = LiCVPR(signal=pre_filtering, bg_signal=raw_bg_green_signal, fps=fps)

            # Post-Filtering
            filtered_bvp = apply_filters(signal=bvp, combination=['detrending_filter', 'butterworth_bp_filter'],
                                         filtering_params={'low': 0.7, 'high': 4.0, 'fps': fps})

            # Compute PSD
            hrES = welch_estimator(signal=filtered_bvp, fps=fps, remove_outlier=False,
                                   bpm_type=heart_rate_calculation_mode, signal_length=hr_interval,
                                   increment=hr_interval, mask=[0.7, 4.0])
            hrES = np.mean(hrES)

        elif dataset == 'COHFACE':
            if hr_interval is None:
                hr_interval = 10

            raw_green_sig = extract_raw_sig(input_video, ROI_name='ICA', ROI_type='None', width=0.6, height=1,
                                            pixel_filtering=True)  # Get the raw green signal
            raw_green_sig = np.array(raw_green_sig)[:, 1]
            fps = get_fps(input_video, dataset)

            # Pre-Filtering
            filtered_bvp = apply_filters(signal=raw_green_sig, combination=['moving_average_filter',
                                                                            'butterworth_bp_filter'],
                                         filtering_params={'low': 0.7, 'high': 4.0, 'fps': fps})

            # Compute PPG Signal
            bvp = LiCVPR(signal=filtered_bvp, bg_signal=raw_bg_green_signal, fps=fps)

            # Post-Filtering
            filtered_bvp = apply_filters(signal=bvp, combination=['moving_average_filter'])

            # Compute PSD
            hrES = fft_estimator(signal=filtered_bvp, fps=fps, remove_outlier=False,
                                 bpm_type=heart_rate_calculation_mode, signal_length=hr_interval,
                                 increment=hr_interval, mask=[0.7, 4.0])
            hrES = np.mean(hrES)

        return hrES

    def licvpr_ground_truth(self, ground_truth_file=None, dataset=None, gtTrace=None, sampling_frequency=None,
                            heart_rate_calculation_mode='continuous', hr_interval=None):
        """
        Obtain the ground truth heart rate of the input video using the LiCVPR framework.

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
        heart_rate_calculation_mode : str
            The mode of heart rate calculation to be used. It can be set to one of the following:
            - 'average': The function computes the average heart rate over the entire duration of the video.
            - 'continuous': The function computes the heart rate at regular specified intervals throughout the video.
            Defaults to 'average'
        hr_interval : int, optional
            This parameter is used when 'heart_rate_calculation_mode' is set to 'continuous'. It specifies the time interval
            (in seconds) at which the heart rate is calculated throughout the video. If not set, a default interval of
            10 seconds is used.
            Defaults to 10 if None

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

        if hr_interval is None:
            hr_interval = 10

        # Compute PSD
        hrGT = welch_estimator(signal=np.array(gtTrace), fps=sampling_frequency, remove_outlier=False,
                               bpm_type=heart_rate_calculation_mode, signal_length=hr_interval, increment=hr_interval,
                               mask=[0.7, 4.0])
        hrGT = np.mean(hrGT)

        return hrGT
