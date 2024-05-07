import matplotlib.pyplot as plt
import pandas as pd

from mkwiforestsliding import MKWIForestSliding
from mkwkiforestbatch import MKWKIForestBatch

DATA = [
    'SunburyLock_2019-05-01_2019-07-15.csv',
    'SunburyLock_2024-02-01_2024-02-29.csv',
    'SunburyLock_2019-01-01_2024-02-01.csv'
]

SLIDING = False


def main():
    score_threshold = 0.75
    window_size = 128
    slope_threshold = 0.001

    df = pd.read_csv(f'data/{DATA[1]}')

    df = df.set_index(pd.to_datetime(df['dateTime']))

    df = df.drop(columns=['dateTime'])

    mkif = MKWIForestSliding(
        score_threshold=score_threshold,
        window_size=window_size,
        slope_threshold=slope_threshold)

    res = pd.DataFrame()

    for i, x in df.iterrows():
        scores = mkif.update(x['value'])

        if scores is not None:
            res = pd.concat([res, pd.DataFrame(scores)])

    # Shift the results to match the original data
    df = df.iloc[:len(res)]
    df['label'] = res.values
    df['label'] = df['label'].shift(1)

    df = df.dropna(subset=['label'])

    plt.figure()
    plt.plot(df.index, df['value'], label='Water level')
    plt.scatter(df.index, df['value'], c=df['label'], cmap='seismic', s=1)
    plt.colorbar()
    plt.title(f"MKWForestSliding with {window_size} window size, {slope_threshold} slope threshold"
              f"and {score_threshold} score threshold")
    plt.show()


if __name__ == '__main__':
    main()
