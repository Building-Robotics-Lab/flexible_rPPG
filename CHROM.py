"""

This module contains the framework implemented by https://ieeexplore.ieee.org/document/6523142 which is
also known as CHROM rPPG by other research papers. This is the closest implementation of the original
framework that has been proposed. It also contains the improved version.

"""

from flexible_rPPG.sig_extraction_utils import *
from flexible_rPPG.utils import *
import numpy as np
from flexible_rPPG.methods import CHROM
from flexible_rPPG.hr_estimator import stft_estimator, welch_estimator, fft_estimator
from tqdm import tqdm


class CHROMImplementations:
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
            print(f"Processing {self.dataset_name} dataset using {self.implementation} implementation of CHROM")
            for i in tqdm(range(len(self.videos))):
                hrES.append(self.chrom_original(input_video=self.videos[i], dataset=self.dataset_name))
                hrGT.append(self.chrom_ground_truth(ground_truth_file=self.gt_files[i], dataset=self.dataset_name))
                # print(f"{i+1}/{len(self.videos)} videos processed")

        elif self.implementation == 'improved':
            print(f"Processing {self.dataset_name} dataset using {self.implementation} implementation of CHROM")
            for i in tqdm(range(len(self.videos))):
                hrES.append(self.chrom_improved(input_video=self.videos[i], dataset=self.dataset_name))
                hrGT.append(self.chrom_ground_truth(ground_truth_file=self.gt_files[i], dataset=self.dataset_name))
                # print(f"{i+1}/{len(self.videos)} videos processed")

        mae, rmse, r = evaluation_metrics(ground_truth_hr=hrGT, estimated_hr=hrES)
        print(f"MAE : {mae} , RMSE : {rmse} , PCC : {r}")

    def chrom_original(self, input_video, subject_type='motion', dataset=None):
        """
        Estimate the heart rate of the input video using the original CHROM implementation.

        Parameters
        ----------
        input_video : str
            Path to the input video file.
        subject_type : str, optional
            Describes the type of the subject in the video.
                - 'static': Subject remains still.
                - 'motion': Subject has some movement.
            Defaults to 'motion'.
        dataset : str, optional
            Name of the dataset.
            Defaults to None.

        Returns
        -------
        list
            Estimated heart rates of the subject in the input video.
        """

        if subject_type == 'static':
            segment_length = 500

            frames = []
            for frame in extract_frames_yield(input_video):
                frames.append(frame)

            motion = self.calculate_motion(frames)  # Calculate motion between consecutive images
            i_s = self.find_least_motion_segment(motion,
                                                 segment_length)  # Starting segment with the least inter frame motion

            raw_sig = extract_raw_sig(input_video, ROI_name='CHROM', width=1, height=1, pixel_filtering=True)
            fps = get_fps(input_video, dataset)

            selected_segment = raw_sig[i_s:i_s + segment_length]  # Select the segment with least inter frame motion
            normalized = normalize(selected_segment,
                                   normalize_type='mean_normalization')  # Normalize the selected segment

            # Build two orthogonal chrominance signals
            Xs = 3 * normalized[0] - 2 * normalized[1]
            Ys = 1.5 * normalized[0] + normalized[1] - 1.5 * normalized[2]

            # bandpass filter Xs and Ys here
            Xf = fir_bp_filter(signal=Xs, fps=fps, low=0.67, high=4.0)
            Yf = fir_bp_filter(signal=Ys, fps=fps, low=0.67, high=4.0)

            alpha = np.std(Xf) / np.std(Yf)
            S = Xf - alpha * Yf

            hrES = welch_estimator(signal=S, fps=fps, remove_outlier=False, bpm_type='average', signal_length=None, increment=None, mask=[0.67, 4.0])

        elif subject_type == 'motion':

            raw_sig = extract_raw_sig(input_video, ROI_name='CHROM', width=1, height=1, pixel_filtering=True)
            fps = get_fps(input_video, dataset)

            window_size, increment = 1.6, 0.8
            window = moving_window(raw_sig, fps=fps, window_size=1.6, increment=0.8)

            # Compute PPG Signal
            H = CHROM(signal=window, fps=fps, increment=increment)

            # Compute STFT
            hrES = stft_estimator(signal=H, fps=fps, remove_outlier=False, signal_length=12, increment=1, mask=[0.67, 4.0])
            hrES = np.mean(hrES)

        else:
            assert False, "Invalid subject type. Please choose one of the valid available types " \
                          "types: 'static', or 'motion' "

        return hrES

    def chrom_improved(self, input_video, dataset=None):
        """
        Estimate the heart rate of the input video using the improved CHROM implementation depending on the dataset used

        Parameters
        ----------
        input_video : str
            Path to the input video file.
        dataset : str, optional
            Name of the dataset.
            Defaults to None.

        Returns
        -------
        list
            Estimated heart rates of the subject in the input video.
        """

        if dataset == 'UBFC2':
            raw_sig = extract_raw_sig(input_video, ROI_name='GREEN', ROI_type='ROI_I', width=1, height=1, pixel_filtering=True)
            fps = get_fps(input_video, dataset)

            window_size, increment = 1.6, 0.8
            window = moving_window(raw_sig, fps=fps, window_size=1.6, increment=0.8)

            # Compute PPG Signal
            H = CHROM(signal=window, fps=fps, increment=increment)

            # Post-Filtering
            filtered_H = apply_filters(signal=H, combination=['butterworth_bp_filter'], filtering_params={'low': 0.67, 'high': 4.0, 'fps': fps})

            # Compute STFT
            hrES = stft_estimator(signal=filtered_H, fps=fps, remove_outlier=False, signal_length=12, increment=1, mask=[0.67, 4.0])
            hrES = np.mean(hrES)

        elif dataset == 'PURE':
            raw_sig = extract_raw_sig(input_video, ROI_name='ICA', width=0.6, height=1, pixel_filtering=True)
            fps = get_fps(input_video, dataset)

            window_size, increment = 1.6, 0.8
            window = moving_window(raw_sig, fps=fps, window_size=1.6, increment=0.8)

            # Pre-Filtering
            filtered_H = apply_filters(signal=window, combination=['detrending_filter'])

            # Compute PPG Signal
            H = CHROM(signal=filtered_H, fps=fps, increment=increment)

            # Post-Filtering
            filtered_H = apply_filters(signal=H, combination=['moving_average_filter'])

            # Compute Welch
            hrES = fft_estimator(signal=filtered_H, fps=fps, remove_outlier=False, bpm_type='continuous', signal_length=12, increment=1, mask=[0.67, 4.0])
            hrES = np.mean(hrES)

        elif dataset == 'COHFACE':
            raw_sig = extract_raw_sig(input_video, ROI_name='CHROM', width=1, height=1, pixel_filtering=True)
            fps = get_fps(input_video, dataset)

            window_size, increment = 1.6, 0.8
            window = moving_window(raw_sig, fps=fps, window_size=1.6, increment=0.8)

            # Pre-Filtering
            filtered_H = apply_filters(signal=window, combination=['detrending_filter'])

            # Compute PPG Signal
            H = CHROM(signal=filtered_H, fps=fps, increment=increment)

            # Post-Filtering
            filtered_H = apply_filters(signal=H, combination=['moving_average_filter'])

            # Compute FFT
            hrES = welch_estimator(signal=filtered_H, fps=fps, remove_outlier=False, bpm_type='continuous', signal_length=12, increment=1, mask=[0.67, 4.0])
            hrES = np.mean(hrES)

        return hrES

    def calculate_intensity(self, image):
        # calculate the intensity of the image
        return np.sum(image, axis=2) / 3.0

    def calculate_motion(self, images):
        # calculate the motion between consecutive images
        motion = []
        for i in range(len(images) - 1):
            motion.append(np.sum(np.abs(self.calculate_intensity(images[i]) - self.calculate_intensity(images[i + 1]))))

        return motion

    def find_least_motion_segment(self, motion, segment_length):
        # find the segment with the least motion
        min_motion = np.inf
        min_index = -1
        for i in range(len(motion) - segment_length):
            motion_in_segment = np.sum(motion[i:i + segment_length])
            if motion_in_segment < min_motion:
                min_motion = motion_in_segment
                min_index = i

        return min_index

    def chrom_ground_truth(self, ground_truth_file=None, dataset=None, gtTrace=None, sampling_frequency=None):
        """
        Obtain the ground truth heart rate of the input video using the CHROM framework.

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
