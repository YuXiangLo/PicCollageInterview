import pandas as pd
import matplotlib.pyplot as plt

def plot_predictions(test_csv, pred_csv, output_fn):
    # 1) Load DataFrames
    df_test = pd.read_csv(test_csv)        # e.g. columns: ['id', 'corr']
    df_pred = pd.read_csv(pred_csv)        # e.g. columns: ['id', 'pred_corr']

    # 2) Merge on 'id'
    df_merged = pd.merge(df_test, df_pred, on='id')

    # 3) Extract true vs. predicted
    true_corr = df_merged['corr']
    pred_corr = df_merged['pred_corr']

    # 4) Scatter plot
    plt.figure()
    plt.scatter(true_corr, pred_corr, alpha=0.5, label='Samples')
    plt.plot([-1, 1], [-1, 1], 'r--', label='y = x')  # reference line
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.xlabel('True Correlation')
    plt.ylabel('Predicted Correlation')
    plt.title('Predicted vs. True Correlation')
    plt.legend()
    # plt.show()
    plt.savefig(output_fn)

# Example usage:
plot_predictions("../csvs/test.csv", "../csvs/predictions.csv", '../outputs/visualization.png')

