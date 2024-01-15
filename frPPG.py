from flexible_rPPG.sig_extraction_utils import *
from flexible_rPPG.methods import *
from flexible_rPPG.hr_estimator import *
from importlib import import_module
from itertools import product
import csv


class frPPG:
    def __init__(self, dataset_name, dataset_dir):
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir

        self.videos, self.gt_files = get_video_and_gt_files(dataset=self.dataset_name, base_dir=self.dataset_dir)
        self.videos, self.gt_files = self.videos[:2], self.gt_files[:2]
        print(self.videos, self.gt_files)
        if len(self.videos) != len(self.gt_files):
            raise ValueError("The number of videos does not match the number of ground truth files.")

    def simulate(self):
        """
        Simulate to find the best combination of algorithms for each core rPPG extraction algorithms
        """
        # First simulate the exhaustive search algorithm
        self.exhaustive_search_algorithm()

        # Find the best combination of algorithms for each core rPPG extraction algorithms
        self.find_best_combination()

    def exhaustive_search_algorithm(self):
        """
        Performs the exhaustive search algorithm to find the best combination of algorithms for different core rPPG
        extraction algorithm. For every possible rPPG method combination on the specified dataset, the parameters and
        the results (MAE, RMSE, and r) are written to an csv file.
        """
        # All the available options and methods for each of the inputs. "GREEN_ROI_II", "GREEN_ROI_III", and
        # "GREEN_ROI_IV" has been removed from the exhaustive search algorithm since they very bad but you are welcome
        # to add them if you want during simulations. Keep in mind you might face some errors when working with these
        # 3 GREEN ROI's since the signal is not so clean.
        face_detection_and_tracking = ["CHROM_ROI", "POS_ROI", "ICA_ROI", "GREEN_ROI_I", "GREEN_ROI_II",
                                       "GREEN_ROI_III", "GREEN_ROI_IV", "LiCVPR_ROI"]
        rgb_thresholding = [True, False]
        filtering_methods = ['detrending_filter', 'moving_average_filter', 'butterworth_bp_filter', 'fir_bp_filter']
        filtering_combinations = get_filtering_combinations(filtering_methods)
        methods = ["CHROM", "POS", "ICA", "GREEN", "LiCVPR"]
        hr_estimation = ['stft_estimator', 'fft_estimator', 'welch_estimator']
        outlier_removal = [True, False]

        # Get all the possible combinations from the given inputs
        parameter_combinations = list(product(face_detection_and_tracking, rgb_thresholding, filtering_combinations,
                                              methods, filtering_combinations, hr_estimation, outlier_removal))

        for i, params in enumerate(parameter_combinations):
            each_face_detection_and_tracking, rgb_threshold, pre_filtering_combinations, each_method, post_filtering_combinations, each_hr_estimation, remove_outlier = params
            hrES, hrGT = [], []
            for enum, each_video in enumerate(self.videos):
                print(f"Combination {i}: {enum}/{len(self.videos)}")
                estimated_hr, ground_truth_hr = self.flexible_rPPG(input_video=self.videos[enum],
                                                                   gt_file=self.gt_files[enum],
                                                                   face_det_and_tracking=each_face_detection_and_tracking,
                                                                   rgb_threshold=rgb_threshold,
                                                                   pre_filtering=pre_filtering_combinations,
                                                                   method=each_method,
                                                                   post_filtering=post_filtering_combinations,
                                                                   hr_estimation=each_hr_estimation,
                                                                   remove_outlier=remove_outlier,
                                                                   dataset=self.dataset_name)
                hrES.append(estimated_hr)
                hrGT.append(ground_truth_hr)

            mae, rmse, r = evaluation_metrics(ground_truth_hr=hrGT, estimated_hr=hrES)

            with open(f'{self.dataset_name}_permutations.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                if i == 0:  # Write headers only for the first iteration
                    writer.writerow(
                        ["Iteration", "Face_Detection_and_Tracking_algorithm", "RGB_Threshold", "Pre_filtering",
                         "Core_rPPG_Extraction_Algorithm", "Post_filtering", "HR_estimation_algorithm",
                         "Outlier_removal", "MAE", "RMSE", "r"])
                writer.writerow([i + 1, each_face_detection_and_tracking, rgb_threshold, pre_filtering_combinations,
                                 each_method, post_filtering_combinations, each_hr_estimation, remove_outlier, mae,
                                 rmse, r])
            print(f"{(i + 1 / len(list(parameter_combinations))) * 100}% done or {i + 1}/{len(list(parameter_combinations))} or {len(list(parameter_combinations)) - i + 1} left")

    def find_best_combination(self):
        """
        Finds the best combination of algorithms for each core rPPG extraction algorithm after applying the exhaustive
        search algorithm, and prints them out
        """
        core_rPPG_algorithm = ["CHROM", "POS", "ICA", "GREEN", "LiCVPR"]
        simulated_file = pd.read_csv(f'{self.dataset_name}_permutations.csv', header=0)

        for algorithm in core_rPPG_algorithm:
            # Filter rows where the 5th column matches the current algorithm
            algorithm_rows = simulated_file[simulated_file['Core_rPPG_Extraction_Algorithm'] == algorithm]

            # Find the minimum value in the 10th column for the current algorithm
            min_value = algorithm_rows['MAE'].min()

            # Find the row where the minimum value is located
            min_row = algorithm_rows[algorithm_rows['MAE'] == min_value].iloc[0]
            min_row = min_row.tolist()

            print(f"For {algorithm}: best result is value is {min_value} MAE and its best combination of algorithms are:")
            print(f"Combination Number                     :  {min_row[0]}")
            print(f"Face Detection and Tracking Algorithm  :  {min_row[1]}")
            print(f"RGB Thresholding                       :  {min_row[2]}")
            print(f"Pre Filtering Algorithm Combinations   :  {min_row[3]}")
            print(f"Core rPPG Extraction Algorithm         :  {min_row[4]}")
            print(f"Post Filtering Algorithm Combinations  :  {min_row[5]}")
            print(f"Heart Rate Estimation Algorithm        :  {min_row[6]}")
            print(f"Outlier Removal                        :  {min_row[7]}")
            print(f"MAE                                    :  {min_row[8]}")
            print(f"RMSE                                   :  {min_row[9]}")
            print(f"R                                      :  {min_row[10]}")
            print('\n')

            with open(f"{self.dataset_name}_best combination results.txt", 'a') as f:
                f.write(f"For {algorithm}: best result is value is {min_value} MAE and its best combination of algorithms are:")
                f.write(f"Combination Number                     :  {min_row[0]}")
                f.write(f"Face Detection and Tracking Algorithm  :  {min_row[1]}")
                f.write(f"RGB Thresholding                       :  {min_row[2]}")
                f.write(f"Pre Filtering Algorithm Combinations   :  {min_row[3]}")
                f.write(f"Core rPPG Extraction Algorithm         :  {min_row[4]}")
                f.write(f"Post Filtering Algorithm Combinations  :  {min_row[5]}")
                f.write(f"Heart Rate Estimation Algorithm        :  {min_row[6]}")
                f.write(f"Outlier Removal                        :  {min_row[7]}")
                f.write(f"MAE                                    :  {min_row[8]}")
                f.write(f"RMSE                                   :  {min_row[9]}")
                f.write(f"R                                      :  {min_row[10]}")
                f.write('\n\n')

    def flexible_rPPG(self, input_video, gt_file, face_det_and_tracking, rgb_threshold, pre_filtering, method,
                      post_filtering,
                      hr_estimation, remove_outlier, dataset):

        """
        Get the estimated heart rate using rPPG based on the selected methods and input parameters

        Parameters
        ----------
        input_video : str
            Path of the input video
        face_det_and_tracking : str
            Select type of face detection and tracking algorithm from the widely used selected papers. Available types
            are "CHROM_ROI", "POS_ROI", "ICA_ROI", "GREEN_ROI_I", "GREEN_ROI_II", "GREEN_ROI_III", "GREEN_ROI_IV",
            and "LiCVPR_ROI".
        rgb_threshold : bool
            Apply a simple skin selection process through pixel filtering
        pre_filtering : tuple
            Input a tuple of different combination of filtering methods. Available algorithms are 'detrending_filter',
            'moving_average_filter', 'butterworth_bp_filter', and 'fir_bp_filter'.
            For example: ('detrending_filter', 'fir_bp_filter')
        method : str
            Select the type of core rPPG extraction methods. Available types are "CHROM", "POS", "ICA", "GREEN",
            and "LiCVPR"
        post_filtering : tuple
            Input a tuple of different combination of filtering methods. Available algorithms are 'detrending_filter',
            'moving_average_filter', 'butterworth_bp_filter', and 'fir_bp_filter'.
            For example: ('detrending_filter', 'fir_bp_filter')
        hr_estimation : str
            Input a type of heart rate estimation method to use. The available hr estimation methods are
            'stft_estimator', 'fft_estimator', and 'welch_estimator'.
        remove_outlier : bool
            Remove outlier estimated values during hr estimation stage. This is based on the algorithm mentioned in the
            ICA paper
        dataset : str
            Name of the dataset. Available dataset names are 'UBFC1', 'UBFC2', 'LGI_PPGI', 'COHFACE', and 'PURE'
        gt_file : str
            Path to the ground truth dataset file corresponding to the input video

        Returns
        -------
        float
            Returns an estimated heart rate and ground truth heart rate
        """

        # Get the important parameters
        sig_extraction_params, windowing_params, filtering_params, hr_estimation_params, ground_truth_method = self.get_params(
            face_det_and_tracking=face_det_and_tracking, method=method)

        # Extract raw RGB signal
        raw_sig = extract_raw_sig(input_video, **sig_extraction_params, pixel_filtering=rgb_threshold)

        # Get FPS and add fps (sampling rate) to the filtering parameters
        fps = get_fps(input_video, dataset)
        filtering_params['fps'] = fps

        # POS windowing parameter depends on the fps, so it has to be added later after determining the fps of the video
        if method == 'POS':
            windowing_params['increment'] = 1 / fps

        # The CHROM, POS, and ICA methods use sliding window, whereas LiCVPR and GREEN methods do not
        if method == 'CHROM' or method == 'POS' or method == 'ICA':
            sig = moving_window(raw_sig, fps=fps, **windowing_params)
        elif method == 'LiCVPR' or method == 'GREEN':
            sig = np.array(raw_sig)[:, 1]
            bg_sig = extract_raw_bg_signal(input_video, dataset, color='g')
        else:
            assert False, "Please choose the correct method. Available methods: 'CHROM', 'POS', 'ICA', 'LiCVPR', or 'GREEN'"

        # Applying filters to the raw signals
        pre_filtered_sig = apply_filters(sig, pre_filtering, filtering_params)

        # Extracting the BVP signal using the core rPPG extraction algorithms
        bvp_module = import_module('flexible_rPPG.methods')
        bvp_method = getattr(bvp_module, method)
        if method == 'ICA' or method == 'GREEN':
            bvp = bvp_method(pre_filtered_sig)
        elif method == 'CHROM' or method == 'POS':
            bvp = bvp_method(pre_filtered_sig, fps, **windowing_params)
        elif method == 'LiCVPR':
            bvp = bvp_method(pre_filtered_sig, bg_sig, fps)

        # Applying filters to the bvp signals
        post_filtered_sig = apply_filters(bvp, post_filtering, filtering_params)

        # Extract the heart rate from the bvp signals
        hrES = get_bpm(post_filtered_sig, fps, hr_estimation, remove_outlier=remove_outlier,
                       params=hr_estimation_params)
        hrGT = ground_truth_method(ground_truth_file=gt_file, dataset=dataset)

        return hrES, hrGT

    def get_params(self, face_det_and_tracking, method):
        """
        Get the parameters for the face detection and tracking algorithm and the specific parameters for each core
        rPPG extraction algorithm method

        Parameters
        ----------
        face_det_and_tracking : str
            Select type of face detection and tracking algorithm from the widely used selected papers. Available types
            are "CHROM_ROI", "POS_ROI", "ICA_ROI", "GREEN_ROI_I", "GREEN_ROI_II", "GREEN_ROI_III", "GREEN_ROI_IV",
            and "LiCVPR_ROI".
        method : str
            Select the type of method. Available types are "CHROM", "POS", "ICA", "GREEN", and "LiCVPR"

        Returns
        -------
        tuple[dict, dict, dict, dict, function]
            Returns a dictionary of face detection and tracking parameters, dictionary of sliding window algorithm
            parameters, dictionary of filtering parameters, heart rate estimation parameters, and function for ground
            truth method.

        Raises
        ------
        AssertionError
            If the face detection and tracking, and method name is invalid, it throws an error.
        """

        sig_parameters = {'CHROM_ROI': {'ROI_name': 'CHROM', 'ROI_type': 'None', 'width': 1, 'height': 1},
                          'POS_ROI': {'ROI_name': 'POS', 'ROI_type': 'None', 'width': 1, 'height': 1},
                          'ICA_ROI': {'ROI_name': 'ICA', 'ROI_type': 'None', 'width': 0.6, 'height': 1},
                          'GREEN_ROI_I': {'ROI_name': 'GREEN', 'ROI_type': 'ROI_I', 'width': 1, 'height': 1},
                          'GREEN_ROI_II': {'ROI_name': 'GREEN', 'ROI_type': 'ROI_II', 'width': 1, 'height': 1},
                          'GREEN_ROI_III': {'ROI_name': 'GREEN', 'ROI_type': 'ROI_III', 'width': 1, 'height': 1},
                          'GREEN_ROI_IV': {'ROI_name': 'GREEN', 'ROI_type': 'ROI_IV', 'width': 1, 'height': 1},
                          'LiCVPR_ROI': {'ROI_name': 'LiCVPR', 'ROI_type': 'None', 'width': 1, 'height': 1}}
        sig_params = sig_parameters[face_det_and_tracking]

        ground_truth_method = getattr(
            getattr(__import__(f'{method}'), f'{method}Implementations')(dataset_name=self.dataset_name,
                                                                         dataset_dir=self.dataset_dir),
            f'{method.lower()}_ground_truth')
        if method == 'CHROM':
            window_params = {'window_size': 1.6, 'increment': 0.8}
            filtering_params = {'low': 0.67, 'high': 4.0}
            hr_estimation_params = {'signal_length': 12, 'increment': 1, 'bpm_type': 'continuous', 'mask': [0.67, 4.0]}
        elif method == 'POS':
            window_params = {'window_size': 1.6}
            filtering_params = {'low': 0.67, 'high': 4.0}
            hr_estimation_params = {'signal_length': 12, 'increment': 1, 'bpm_type': 'continuous', 'mask': [0.67, 4.0]}
        elif method == 'ICA':
            window_params = {'window_size': 30, 'increment': 1}
            filtering_params = {'low': 0.75, 'high': 4.0}
            hr_estimation_params = {'signal_length': 30, 'increment': 1, 'bpm_type': 'continuous', 'mask': [0.75, 4.0]}
        elif method == 'LiCVPR':
            window_params = None
            filtering_params = {'low': 0.7, 'high': 4.0}
            hr_estimation_params = {'signal_length': 10, 'increment': 10, 'bpm_type': 'continuous', 'mask': [0.7, 4.0]}
        elif method == 'GREEN':
            window_params = None
            filtering_params = {'low': 0.8, 'high': 2.0}
            hr_estimation_params = {'signal_length': 6, 'increment': 1, 'bpm_type': 'average', 'mask': [0.8, 2.0]}
        else:
            assert False, "Please choose the correct method. Available methods: 'CHROM', 'POS', 'ICA', 'LiCVPR', or 'GREEN'"

        return sig_params, window_params, filtering_params, hr_estimation_params, ground_truth_method
