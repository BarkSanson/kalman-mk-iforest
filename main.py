import os
import sys
import time

import matplotlib.pyplot as plt
import pandas as pd

from online_outlier_detection import \
    MKWKIForestBatch, MKWKIForestSliding, MKWIForestBatch, MKWIForestSliding

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
    slope_threshold = 0.001
    alpha = 0.05

    if len(sys.argv) != 3:
        print("Usage:")
        print("python main.py <data-directory-path> <score_threshold>")
        sys.exit(1)

    data_dir = sys.argv[1]
    score_threshold = float(sys.argv[2])

    if not os.path.isdir(data_dir):
        print("Argument is not a directory")

    data_list = [x for x in os.listdir(data_dir) if os.path.isdir(f"{data_dir}/{x}")]

    time_report = pd.DataFrame(columns=['station', 'date', 'model', 'time'])

    for station in data_list:
        path = f"{data_dir}/{station}"

        for date in os.listdir(path):
            models = [
                MKWKIForestBatch(
                    score_threshold=score_threshold,
                    alpha=alpha,
                    window_size=window_size,
                    slope_threshold=slope_threshold),
                MKWKIForestSliding(
                    score_threshold=score_threshold,
                    alpha=alpha,
                    window_size=window_size,
                    slope_threshold=slope_threshold),
                MKWIForestBatch(
                    score_threshold=score_threshold,
                    alpha=alpha,
                    window_size=window_size,
                    slope_threshold=slope_threshold),
                MKWIForestSliding(
                    score_threshold=score_threshold,
                    alpha=alpha,
                    window_size=window_size,
                    slope_threshold=slope_threshold)]

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

                time_report = pd.concat([time_report, pd.DataFrame({
                    'station': [station],
                    'date': [date],
                    'model': [type(model).__name__],
                    'time': [time.time() - initial_time]
                })], ignore_index=True)

                df = df.iloc[:len(res)]

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
                    f"score-thresh={score_threshold}_window_size={window_size}_slope-thresh={slope_threshold}.csv")

    time_report.to_csv(f"{RESULTS_DIR}/time_report.csv")


if __name__ == '__main__':
    main()
