import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import Lasso
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from utils import setup_seed
setup_seed(2024)
x_train_dir=r"/pub/data/gaoss/New_Multi/code/MAGIC_prs_20/Lasso/train/train.dat"
y_train_dir=r"/pub/data/gaoss/New_Multi/code/MAGIC_prs_20/Lasso/train/labels.csv"

x_valid_dir="/pub/data/gaoss/New_Multi/code/MAGIC_prs_20/Lasso/valid/valid.npy"
y_valid_dir="/pub/data/gaoss/New_Multi/code/MAGIC_prs_20/Lasso/valid/labels.csv"

x_test_dir=r"/pub/data/gaoss/New_Multi/code/MAGIC_prs_20/Lasso/test/test.npy"
y_test_dir=r"/pub/data/gaoss/New_Multi/code/MAGIC_prs_20/Lasso/test/labels.csv"

total_samples=12600
features = 179712
#x_train = np.memmap(x_train_dir, dtype='float32', mode='r', shape=(total_samples, features))
#print("x_train.shape",x_train.shape)
df = pd.read_csv(y_train_dir)
y_train = df['label'].to_numpy()

x_valid = np.load(x_valid_dir)
# print(x_valid)
# print(x_valid.shape)
df = pd.read_csv(y_valid_dir)
y_valid = df['label'].to_numpy()
# print(y_valid.shape)

x_test = np.load(x_test_dir)
# print(x_test)
# print(x_test.shape)
df = pd.read_csv(y_test_dir)
y_test = df['label'].to_numpy()
model = Lasso(alpha=0.1)
# print(model)
model.fit(x_valid, y_valid)
# print(model)
'''
lasso_model = SGDRegressor(alpha=0.2, penalty='l1', max_iter=500, tol=1e-4)
lasso_model.fit(x_train, y_train)
'''
print(x_test[2:3])
y_pred = model.predict(x_test[2:3])
print(y_pred)
print(x_test[3:4])
y_pred = model.predict(x_test[3:4])
print(y_pred)
#auc = roc_auc_score(y_test, y_pred)
#print(f"AUC: {auc:.4f}")
print(len(y_pred))


