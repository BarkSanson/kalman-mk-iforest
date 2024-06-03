import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_confusion_matrix(
        cm: np.ndarray,
        results_dir: str,
        model: str,
        window_size: int,
        slope_threshold: float,
        score_threshold: float) -> None:
    plt.figure(figsize=(10, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Normal', 'Outlier'],
        yticklabels=['Normal', 'Outlier'])
    plt.ylabel('Etiquetas reales')
    plt.xlabel('Etiquetas predichas')
    plt.title(f"Matriz de confusión de {model}")
    plt.savefig(f"{results_dir}/{model}/"
                f"confusion-matrix_"
                f"window-size={window_size}_"
                f"score-thresh={score_threshold}.png")


def plot_time_series_with_labels(df: pd.DataFrame, window_size: int, slope_threshold: float,
                                 score_threshold: float, two_days_date: str, save_path: str, model:str) -> None:
    plt.figure(figsize=(20, 10))
    plt.plot(df.index, df['value'])
    scatter = plt.scatter(df.index, df['value'], c=df['label'], cmap='seismic', s=20,
                          label='Outliers')
    plt.legend(handles=scatter.legend_elements()[0], labels=['Normal', 'Outlier'])
    plt.title(
        f"{model} with {window_size} window size, {slope_threshold} slope threshold"
        f"and {score_threshold} score threshold. Two days from {two_days_date}")

    plt.savefig(
        f"{save_path}/"
        f"score-thresh={score_threshold}_"
        f"window-size={window_size}_"
        f"slope-thresh={slope_threshold}_"
        f"two_days={two_days_date}.png")
