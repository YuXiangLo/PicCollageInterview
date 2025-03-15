import pandas as pd

df_test = pd.read_csv('../csvs/test.csv')
df_pred = pd.read_csv('../csvs/predictions.csv')

corr_test = df_test['corr'].tolist()
corr_pred = df_pred['pred_corr'].tolist()

test_mean, pred_mean = 0, 0
L1, L2 = 0, 0

for t, p in zip(corr_test, corr_pred):
    test_mean += t
    pred_mean += p
    L1 += abs(t - p)
    L2 += (t - p) ** 2

print(f"Test Mean: {test_mean / len(corr_pred)}")
print(f"Pred Mean: {pred_mean / len(corr_pred)}")
print(f"L1 Loss: {L1 / len(corr_pred)}")
print(f"L2 Loss: {L2 / len(corr_pred)}")
