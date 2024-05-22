import os
import sys
import time

import pandas as pd
from sklearn.metrics import \
    accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix

from online_outlier_detection import \
    MKWKIForestBatchPipeline, MKWKIForestSlidingPipeline, MKWIForestBatchPipeline, MKWIForestSlidingPipeline
from plots import plot_confusion_matrix, plot_time_series_with_labels

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

    if len(sys.argv) != 4:
        print("Usage:")
        print("python main.py <data-directory-path> <score_threshold> <step>")
        sys.exit(1)

    data_dir = sys.argv[1]
    score_threshold = float(sys.argv[2])
    step = int(sys.argv[3])

    if not os.path.isdir(data_dir):
        print("Argument is not a directory")

    data_list = [x for x in os.listdir(data_dir) if os.path.isdir(f"{data_dir}/{x}")]

    report = pd.DataFrame(columns=['station', '3_weeks_start_date', 'model', 'time'])

    results = {
        'MKWKIForestBatchPipeline': pd.DataFrame(columns=['true_label', 'predicted_label', 'score']),
        'MKWKIForestSlidingPipeline': pd.DataFrame(columns=['true_label', 'predicted_label', 'score']),
        'MKWIForestBatchPipeline': pd.DataFrame(columns=['true_label', 'predicted_label', 'score']),
        'MKWIForestSlidingPipeline': pd.DataFrame(columns=['true_label', 'predicted_label', 'score'])
    }
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

                matching_df = df.iloc[:len(res)]

                true_labels = matching_df['label'].values
                predicted_labels = res['label'].values
                scores = res['score'].values

                results[type(model).__name__] = pd.concat([
                    results[type(model).__name__],
                    pd.DataFrame({'true_label': true_labels, 'predicted_label': predicted_labels, 'score': scores})],
                    ignore_index=True)

                report = pd.concat([report, pd.DataFrame({
                    'station': [station],
                    '3_weeks_start_date': [date],
                    'model': [type(model).__name__],
                    'time': [total_time],
                })], ignore_index=True)

                save_path = f"{RESULTS_DIR}/{type(model).__name__}/{station}/{date}"
                os.makedirs(save_path, exist_ok=True)

                for two_days_date in matching_df['block'].unique():
                    print(
                        f"Plotting {type(model).__name__} with {window_size} window size, {slope_threshold} slope for "
                        f"{two_days_date}...")
                    two_days = matching_df[matching_df['block'] == two_days_date]
                    plot_time_series_with_labels(two_days, window_size, slope_threshold, score_threshold,
                                                 two_days_date, save_path, type(model).__name__)

                df.to_csv(
                    f"{save_path}/"
                    f"score-thresh={score_threshold}_"
                    f"window-size={window_size}_"
                    f"slope-thresh={slope_threshold}.csv")

    for model in results:
        true_labels = pd.to_numeric(results[model]['true_label'])
        predicted_labels = pd.to_numeric(results[model]['predicted_label'])
        scores = pd.to_numeric(results[model]['score'])
        accuracy, precision, recall, f1, cm, roc_auc = \
            accuracy_score(true_labels, predicted_labels), \
            precision_score(true_labels, predicted_labels), \
            recall_score(true_labels, predicted_labels), \
            f1_score(true_labels, predicted_labels), \
            confusion_matrix(true_labels, predicted_labels), \
            roc_auc_score(true_labels, scores)

        plot_confusion_matrix(cm, RESULTS_DIR, model)

        metrics = pd.DataFrame({
            'accuracy': [accuracy],
            'precision': [precision],
            'recall': [recall],
            'f1': [f1],
            'roc_auc': [roc_auc],
            'mean_time': [report[report['model'] == model]['time'].mean()]
        })
        metrics.to_csv(f"{RESULTS_DIR}/{model}/"
                       f"metrics_score-thresh={score_threshold}_"
                       f"window-size={window_size}_"
                       f"slope-thresh={slope_threshold}.csv", index=False)


if __name__ == '__main__':
    main()
