# frPPG

## Abstract
Over the past decade, novel Remote Photoplethysmography (rPPG) methods in the literature have been growing, aiming to offer an unobtrusive alternative to wearables and devices attached to the skin. Despite advancements in rPPG methods, their application in scalable, real-world deployments has encountered significant hurdles, primarily due to the lack of benchmark methods for refining and validating these novel methods. To address this, we conducted a comprehensive literature review on novel rPPG methods and extracted information related to the algorithms used in pre- and post-processing stages and validation process including their benchmark algorithms, datasets, evaluation metrics, training and testing data segmentation, reported results, and. We specifically illustrate the high variability in reported Mean Absolute Error (MAE) of benchmark rPPG methods applied to the same public datasets. Through dissecting the original implementation of the established benchmark rPPG methods, we propose a flexible framework to optimally select the algorithms in pre- and post-processing stages via an exhaustive search. We applied this framework to the original implementation of benchmark algorithms on three public datasets. Our results indicate that 80% of the refined methods fall within the top 25th percentile of the reported MAE, RMES and PCC results. Moreover, 60% of the refined methods surpassed the lowest reported accuracies. These refined methods could also be used as benchmark methods for evaluating the novel methods, offering a more stringent validation process. The codebase for this framework (frPPG) is accessible at xxx. It could be leveraged to either support the design of novel rPPG methods or their comparison with best performing benchmark algorithms on a given dataset. 

## Basic Setup Guidelines
- Clone the [repo](https://github.com/Building-Robotics-Lab/flexible_rPPG)
- Go into the repo `cd flexible_rPPG`
- Install the required libraries `pip install -r requirements.txt`

## Basic Usage Guidelines
- To apply the frPPG to find the best combination of pre- and post-processing algorithms for each of the available core rPPG extraction algorithms, run the `frPPG.py` file by doing:
  - `frPPG(dataset_name='name_of_dataset', dataset_dir='path/to/dataset').simulate()`
- If you want to independently use the frPPG method on your own videos, do the following:
  - `flexible_rPPG(input_video, gt_file=None, face_det_and_tracking, rgb_threshold, pre_filtering, method, post_filtering, hr_estimation, remove_outlier, dataset=None):`
  - Remember to set the `gt_file=None` and `dataset to None`

- To simulate the original implementations of the rPPG methods, you can do:
  - 
