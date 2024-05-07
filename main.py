import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

from online_outlier_detection import \
    MKWKIForestBatch, MKWKIForestSliding, MKWIForestBatch, MKWIForestSliding

SLIDING = False


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
    score_threshold = 0.65
    window_size = 128
    slope_threshold = 0.001

    if len(sys.argv) != 2:
        print("Usage:")
        print("python main.py <data-directory-path>")
        sys.exit(1)

    data_dir = sys.argv[1]

    if not os.path.isdir(data_dir):
        print("Argument is not a directory")

    data_list = [x for x in os.listdir(data_dir) if os.path.isdir(f"{data_dir}/{x}")]

    for station in data_list:
        path = f"{data_dir}/{station}"

        for date in os.listdir(path):
            models = [
                MKWKIForestBatch(
                    score_threshold=score_threshold,
                    window_size=window_size,
                    slope_threshold=slope_threshold),
                MKWKIForestSliding(
                    score_threshold=score_threshold,
                    window_size=window_size,
                    slope_threshold=slope_threshold),
                MKWIForestBatch(
                    score_threshold=score_threshold,
                    window_size=window_size,
                    slope_threshold=slope_threshold),
                MKWIForestSliding(
                    score_threshold=score_threshold,
                    window_size=window_size,
                    slope_threshold=slope_threshold)]
            df = merge_data(f"{path}/{date}")

            for model in models:
                res = pd.DataFrame()

                for i, x in df.iterrows():
                    result = model.update(x['value'])

                    if result is not None:
                        scores, labels = result
                        res = pd.concat([res, pd.DataFrame({'score': scores, 'label': labels})], ignore_index=True)

                df = df.iloc[:len(res)]

                df['score'] = res['score'].values
                df['label'] = res['label'].values

                df = df.dropna(subset=['label', 'score'])

                plt.figure()
                plt.plot(df.index, df['value'], label='Water level')
                plt.scatter(df.index, df['value'], c=df['label'], cmap='seismic', s=1)
                plt.colorbar()
                plt.title(f"MKWForestSliding with {window_size} window size, {slope_threshold} slope threshold"
                          f"and {score_threshold} score threshold")
                plt.show()


if __name__ == '__main__':
    main()
