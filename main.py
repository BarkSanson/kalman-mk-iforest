import os
import sys
import time

from sklearn.metrics import \
    accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix

import matplotlib.pyplot as plt
import pandas as pd

from online_outlier_detection import \
    MKWKIForestBatchPipeline, MKWKIForestSlidingPipeline, MKWIForestBatchPipeline, MKWIForestSlidingPipeline

RESULTS_DIR = "./results"


def merge_data(date_dir):
    df = pd.DataFrame()
    for file in os.listdir(date_dir):
        new_df = pd.read_csv(f"{date_dir}/{file}")
        new_df['block'] = file.split(sep="_")[1]
        df = pd.concat([df, new_df])

    df = df.set_index(pd.to_datetime(df['dateTime']))
    df = df.drop(columns=['dateTime'])

    return df


def main():
    window_size = 128
    slope_threshold = 0.01
    alpha = 0.05
    step = 5

    if len(sys.argv) != 3:
        print("Usage:")
        print("python main.py <data-directory-path> <score_threshold>")
        sys.exit(1)

    data_dir = sys.argv[1]
    score_threshold = float(sys.argv[2])

    if not os.path.isdir(data_dir):
        print("Argument is not a directory")

    data_list = [x for x in os.listdir(data_dir) if os.path.isdir(f"{data_dir}/{x}")]

    report = pd.DataFrame(columns=['station', '3_weeks_start_date', 'model', 'time',
                                   'accuracy', 'precision', 'recall', 'f1', 'roc_auc'])

    for station in data_list:
        path = f"{data_dir}/{station}"

        for date in os.listdir(path):
            models = [
                MKWKIForestBatchPipeline(
                    score_threshold=score_threshold,
                    alpha=alpha,
                    window_size=window_size,
                    slope_threshold=slope_threshold),
                MKWKIForestSlidingPipeline(
                    score_threshold=score_threshold,
                    alpha=alpha,
                    window_size=window_size,
                    slope_threshold=slope_threshold,
                    step=step),
                MKWIForestBatchPipeline(
                    score_threshold=score_threshold,
                    alpha=alpha,
                    window_size=window_size,
                    slope_threshold=slope_threshold),
                MKWIForestSlidingPipeline(
                    score_threshold=score_threshold,
                    alpha=alpha,
                    window_size=window_size,
                    slope_threshold=slope_threshold,
                    step=step)]

            df = merge_data(f"{path}/{date}")

            for model in models:
                res = pd.DataFrame()

                initial_time = time.time()
                print(f"Applying {type(model).__name__} to {station} on {date}...")
                for i, x in df.iterrows():
                    result = model.update(x['value'])

                    if result is not None:
                        scores, labels = result
                        res = pd.concat([res, pd.DataFrame({'score': scores, 'label': labels})], ignore_index=True)

                total_time = time.time() - initial_time

                df = df.iloc[:len(res)]

                true_labels = df['label']
                predicted_labels = res['label']

                accuracy, precision, recall, f1, = \
                    accuracy_score(true_labels, predicted_labels), \
                    precision_score(true_labels, predicted_labels), \
                    recall_score(true_labels, predicted_labels), \
                    f1_score(true_labels, predicted_labels)

                try:
                    roc_auc = roc_auc_score(true_labels, predicted_labels)
                except ValueError as e:  # If only one class is present in the data
                    roc_auc = float('nan')  # Terrible solution, but it should work for my use case

                report = pd.concat([report, pd.DataFrame({
                    'station': [station],
                    '3_weeks_start_date': [date],
                    'model': [type(model).__name__],
                    'time': [total_time],
                    'accuracy': [accuracy],
                    'precision': [precision],
                    'recall': [recall],
                    'f1': [f1],
                    'roc_auc': [roc_auc]
                })], ignore_index=True)

                df['score'] = res['score'].values
                df['label'] = res['label'].values
                df = df.dropna(subset=['label', 'score'])

                save_path = f"{RESULTS_DIR}/{type(model).__name__}/{station}/{date}"
                os.makedirs(save_path, exist_ok=True)

                for two_days_date in df['block'].unique():
                    two_days = df[df['block'] == two_days_date]

                    plt.figure(figsize=(20, 10))
                    plt.plot(two_days.index, two_days['value'])
                    scatter = plt.scatter(two_days.index, two_days['value'], c=two_days['label'], cmap='seismic', s=20,
                                          label='Outliers')
                    plt.legend(handles=scatter.legend_elements()[0], labels=['Normal', 'Outlier'])
                    plt.title(
                        f"{type(model).__name__} with {window_size} window size, {slope_threshold} slope threshold"
                        f"and {score_threshold} score threshold. Two days from {two_days_date}")

                    plt.savefig(
                        f"{save_path}/"
                        f"score-thresh={score_threshold}_"
                        f"window_size={window_size}_"
                        f"slope-thresh={slope_threshold}_"
                        f"two_days={two_days_date}.png")
                    plt.close()

                df.to_csv(
                    f"{save_path}/"
                    f"score-thresh={score_threshold}_"
                    f"window-size={window_size}_"
                    f"slope-thresh={slope_threshold}.csv")

    report.to_csv(f"{RESULTS_DIR}/"
                  f"report_"
                  f"score-thresh={score_threshold}_"
                  f"window-size={window_size}_"
                  f"slope-thresh={slope_threshold}.csv")


if __name__ == '__main__':
    main()
