import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mkiforest import MKKalmanIForestPipeline

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

    #df = df.drop(columns=['quality', 'dateTime'])
    df = df.drop(columns=['quality', 'dateTime', 'completeness', 'qcode', 'date', 'measure'])

    #z_score = (df['value'] - df['value'].mean()) / df['value'].std()
    #df[abs(z_score) > 3] = df['value'].mean()

    #flood = np.random.normal(0, 0.05, 40) + np.linspace(df['value'].iloc[100], 1, 40)
    #for i in range(100, 141):
    #    df.iloc[i] = flood[i - 101]

    #flood = 1 + np.random.normal(0, 0.05, 59)
    #for i in range(141, 199):
    #    df.iloc[i] = flood[i - 140]

    #flood = np.random.normal(0, 0.05, 40) + np.linspace(df['value'].iloc[199], 0, 40)
    #for i in range(199, 239):
    #    df.iloc[i] = flood[i - 199]

    #flood = np.random.normal(0, 0.05, 60) + np.linspace(df['value'].iloc[2000], 1, 60)
    #for i in range(2000, 2061):
    #    df.iloc[i] = flood[i - 2001]

    #flood = 1 + np.random.normal(0, 0.05, 100)
    #for i in range(2061, 2161):
    #    df.iloc[i] = flood[i - 2061]

    #flood = np.random.normal(0, 0.05, 70) + np.linspace(df['value'].iloc[2161], 0, 70)
    #for i in range(2161, 2231):
    #    df.iloc[i] = flood[i - 2161]

    mkif = MKKalmanIForestPipeline(window_size=window_size, slope_threshold=slope_threshold)

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
