import os
import copy
from typing import Dict, Tuple, Any
from ludwig.utils.data_utils import load_json, save_json
from ludwig.utils.misc_utils import merge_dict
from ludwig.constants import CACHE, EXPERIMENT_RUN, TRAIN_TAG, EVAL_TAG


def create_metrics_report(experiment_name: str) -> Tuple[Dict[str, Any], str]:
    """Compiles performance and non-performance metrics.

    `experiment_name`: name referring to the experiment.
    Returns a full report and the path where it's saved.
    """
    full_report = dict()
    os.makedirs(os.path.join(os.getcwd(), experiment_name, "metrics_report"), exist_ok=True)
    for tag in [TRAIN_TAG, EVAL_TAG]:
        if tag == TRAIN_TAG:
            non_performance_path = os.path.join(os.getcwd(), experiment_name, CACHE, "train_metrics.json")
            performance_path = os.path.join(os.getcwd(), experiment_name, EXPERIMENT_RUN, "training_statistics.json")
        elif tag == EVAL_TAG:
            non_performance_path = os.path.join(os.getcwd(), experiment_name, CACHE, "evaluate_metrics.json")
            performance_path = os.path.join(os.getcwd(), experiment_name, EXPERIMENT_RUN, "test_statistics.json")
        else:
            raise ValueError("Tag unrecognized. Please choose 'train' or 'evaluate'.")

        non_performance_metrics = load_json(non_performance_path)
        performance_metrics = load_json(performance_path)
        performance_metrics_copy = copy.deepcopy(performance_metrics)
        full_report[tag] = merge_dict({"performance_metrics": performance_metrics_copy},
                                      {"non_performance_metrics": non_performance_metrics})

    merged_file_path = os.path.join(os.getcwd(), experiment_name, "metrics_report", "{}.json".format("full_report"))
    save_json(merged_file_path, full_report)
    return full_report, merged_file_path
