import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mkwkiforestbatch import MKWKIForestBatch

DATA = [
    'SunburyLock_2019-05-01_2019-07-15.csv',
    'SunburyLock_2024-02-01_2024-02-29.csv',
    'SunburyLock_2019-01-01_2024-02-01.csv'
]

SLIDING = False


def main():
    window_size = 128
    slope_threshold = 0.001

    df = pd.read_csv(f'data/{DATA[1]}')

    df = df.set_index(pd.to_datetime(df['dateTime']))

    df = df.drop(columns=['dateTime'])

    mkif = MKWKIForestBatch(window_size=window_size, slope_threshold=slope_threshold)

    res = pd.DataFrame()

    for i, x in df.iterrows():
        scores = mkif.update(x['value'])

        if scores is not None:
            res = pd.concat([res, pd.DataFrame(scores)], ignore_index=True)

    df = df[:len(res)]

    plt.plot(df.index, df['value'], label="Nivell de l'aigua")
    plt.scatter(df.index, df['value'], c=res, cmap="seismic", s=0.5)
    plt.colorbar()
    plt.title(f'MKiForest with {window_size} window size, {slope_threshold} slope threshold,'
              f'{"sliding window" if SLIDING else "batch window"}.png')

    plt.savefig(f'MKiForest with {window_size} window size, {slope_threshold} slope threshold,'
                f'{"sliding window" if SLIDING else "batch window"}.png')
    plt.show()


if __name__ == '__main__':
    main()
