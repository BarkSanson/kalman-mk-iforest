import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("data_path",
                        type=str,
                        help="Path to data folder")
    parser.add_argument("-r",
                        "--results_path",
                        help="Path to store results, both plots and tables. ./results by default",
                        default=f"{os.getcwd()}/results")
    parser.add_argument("-w",
                        "--window_sizes",
                        type=int,
                        nargs="+",
                        help="Window sizes to use for block window. Default is 64",
                        default=[64])
    parser.add_argument("-s",
                        "--step",
                        type=int,
                        help="Step size for sliding window. Default is 5",
                        default=5)
    parser.add_argument("-t",
                        "--score_threshold",
                        type=float,
                        help="Score threshold for anomaly detection. Default is 0.75",
                        default=0.75)

    return parser.parse_args()
