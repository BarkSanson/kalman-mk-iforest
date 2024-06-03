import sys
import os

import pandas as pd


def main():
    if len(sys.argv) != 3:
        print("Usage:")
        print("python labeler.py <data-directory-path> <results-path>")
        sys.exit(1)

    data_dir = sys.argv[1]
    results_path = sys.argv[2]

    stations = os.listdir(data_dir)

    for station in stations:
        weeks = os.listdir(f"{data_dir}/{station}")
        for week in weeks:
            data_days = os.listdir(f"{data_dir}/{station}/{week}")
            for two_days in data_days:
                if not two_days.endswith(".csv"):
                    continue
                data = pd.read_csv(f"{data_dir}/{station}/{week}/{two_days}")
                station, date, kind = two_days.split("_")
                kind = kind[0]

                if kind in ['R', 'E']:
                    data['label'] = False
                else:
                    data['label'] = data['label'].fillna(False)

                data.to_csv(f"{data_dir}/{station}/{week}/{two_days}", index=False)



if __name__ == '__main__':
    main()
