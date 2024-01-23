"""

This module contains the framework implemented by https://opg.optica.org/oe/fulltext.cfm?uri=oe-18-10-10762&id=199381
which is also known as ICA rPPG by other research papers. This is the closest implementation of the original framework
that has been proposed. It also contains the improved version.

"""

from flexible_rPPG.hr_estimator import fft_estimator, welch_estimator, stft_estimator
from flexible_rPPG.methods import ICA
from flexible_rPPG.sig_extraction_utils import *
from flexible_rPPG.utils import *
from flexible_rPPG.filters import *
from tqdm import tqdm


class ICAImplementations:
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
            print(f"Processing {self.dataset_name} dataset using {self.implementation} implementation of ICA")
            for i in tqdm(range(len(self.videos))):
                hrES.append(self.ica_original(input_video=self.videos[i], dataset=self.dataset_name))
                hrGT.append(self.ica_ground_truth(ground_truth_file=self.gt_files[i], dataset=self.dataset_name))
                # print(f"{i + 1}/{len(self.videos)} videos processed")

        elif self.implementation == 'improved':
            print(f"Processing {self.dataset_name} dataset using {self.implementation} implementation of ICA")
            for i in tqdm(range(len(self.videos))):
                hrES.append(self.ica_improved(input_video=self.videos[i], dataset=self.dataset_name))
                hrGT.append(self.ica_ground_truth(ground_truth_file=self.gt_files[i], dataset=self.dataset_name))
                # print(f"{i + 1}/{len(self.videos)} videos processed")

        mae, rmse, r = evaluation_metrics(ground_truth_hr=hrGT, estimated_hr=hrES)
        print(hrGT)
        print(hrES)
        print(f"MAE : {mae} , RMSE : {rmse} , PCC : {r}")

    def ica_original(self, input_video, comp=1, dataset=None):
        """
        Estimate the heart rate of the input video using the original ICA implementation

        Parameters
        ----------
        input_video : str
            Path to the input video file.
        comp : int
            Select which signal component to choose after applying ICA algorithm.
            Defaults to 1 since from literature, the second component is selected as
            it typically contains a strong plethysmographic signal
        dataset : str, optional
            Name of the dataset. If provided, it may be used for specialized processing.
            Defaults to None.

        Returns
        -------
        list
            Estimated heart rates of the subject in the input video.
        """

        # get the raw RGB signals
        raw_sig = extract_raw_sig(input_video, ROI_name='ICA', ROI_type='None', width=0.6, height=1, pixel_filtering=False)
        fps = get_fps(input_video, dataset)

        windowed_sig = moving_window(sig=raw_sig, fps=fps, window_size=30, increment=1)

        # Compute PPG Signal
        bvp = ICA(signal=windowed_sig, component=comp)
        bvp = apply_filters(signal=bvp, combination=['fir_bp_filter'], filtering_params={'low': 0.75, 'high': 4.0, 'fps': fps})

        # Compute FFT (Note that the hr threshold to remove outlier is set at 12 bpm.)
        # You can change it inside the function
        hrES = fft_estimator(signal=bvp, fps=fps, remove_outlier=True, bpm_type='continuous', signal_length=30, increment=1, mask=[0.75, 4.0])
        hrES = np.mean(hrES)

        return hrES

    def ica_improved(self, input_video, comp=1, dataset=None):
        """
        Estimate the heart rate of the input video using the ICA implementation depending on the dataset used.

        Parameters
        ----------
        input_video : str
            Path to the input video file.
        comp : int
            Select which signal component to choose after applying ICA algorithm.
            Defaults to 1 since from literature, the second component is selected as
            it typically contains a strong plethysmographic signal
        dataset : str, optional
            Name of the dataset. If provided, it may be used for specialized processing.
            Defaults to None.

        Returns
        -------
        list
            Estimated heart rates of the subject in the input video.
        """

        if dataset == 'UBFC2':
            # get the raw RGB signals
            raw_sig = extract_raw_sig(input_video, ROI_name='LiCVPR', ROI_type='None', width=1, height=1, pixel_filtering=False)
            fps = get_fps(input_video, dataset)

            windowed_sig = moving_window(sig=raw_sig, fps=fps, window_size=30, increment=1)

            # Pre-Filtering
            pre_filtered = apply_filters(signal=windowed_sig, combination=['detrending_filter'])

            # Compute PPG Signal
            bvp = ICA(signal=pre_filtered, component=comp)
            bvp = apply_filters(signal=bvp, combination=['detrending_filter', 'fir_bp_filter'], filtering_params={'low': 0.75, 'high': 4.0, 'fps': fps})

            # Compute FFT
            hrES = welch_estimator(signal=bvp, fps=fps, remove_outlier=True, bpm_type='continuous', signal_length=30, increment=1, mask=[0.75, 4.0])
            hrES = np.mean(hrES)

        if dataset == 'PURE':
            # get the raw RGB signals
            raw_sig = extract_raw_sig(input_video, ROI_name='ICA', ROI_type='None', width=0.6, height=1, pixel_filtering=False)
            fps = get_fps(input_video, dataset)

            windowed_sig = moving_window(sig=raw_sig, fps=fps, window_size=30, increment=1)

            # Pre-Filtering
            pre_filtered = apply_filters(signal=windowed_sig, combination=['detrending_filter'])

            # Compute PPG Signal
            bvp = ICA(signal=pre_filtered, component=comp)
            bvp = apply_filters(signal=bvp, combination=['butterworth_bp_filter'], filtering_params={'low': 0.75, 'high': 4.0, 'fps': fps})

            # Compute FFT
            hrES = stft_estimator(signal=bvp, fps=fps, remove_outlier=True, signal_length=30, increment=1, mask=[0.75, 4.0])
            hrES = np.mean(hrES)

        if dataset == 'COHFACE':
            # get the raw RGB signals
            raw_sig = extract_raw_sig(input_video, ROI_name='ICA', ROI_type='None', width=0.6, height=1, pixel_filtering=True)
            fps = get_fps(input_video, dataset)

            windowed_sig = moving_window(sig=raw_sig, fps=fps, window_size=30, increment=1)

            # Pre-Filtering
            pre_filtered = apply_filters(signal=windowed_sig, combination=['detrending_filter', 'moving_average_filter'])

            # Compute PPG Signal
            bvp = ICA(signal=pre_filtered, component=comp)
            bvp = apply_filters(signal=bvp, combination=['detrending_filter', 'butterworth_bp_filter'], filtering_params={'low': 0.75, 'high': 4.0, 'fps': fps})

            # Compute FFT
            hrES = stft_estimator(signal=bvp, fps=fps, remove_outlier=False, signal_length=30, increment=1, mask=[0.75, 4.0])
            hrES = np.mean(hrES)

        return hrES

    def ica_ground_truth(self, ground_truth_file=None, dataset=None, gtTrace=None, sampling_frequency=None):
        """
        Obtain the ground truth heart rate of the input video using the ICA framework.

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

        # signal windowing with 96.7% overlap
        windowed_sig = moving_window(sig=gtTrace, fps=sampling_frequency, window_size=30, increment=1)
        normalized_windowed_sig = np.array([normalize(sig, normalize_type='zero_mean_unit_variance') for sig in windowed_sig])

        # Compute FFT
        hrGT = fft_estimator(signal=normalized_windowed_sig, fps=sampling_frequency, remove_outlier=True, bpm_type='continuous', signal_length=30, increment=1, mask=[0.75, 4.0])
        hrGT = np.mean(hrGT)

        return hrGT
