import shap
import os
import xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from utils import setup_seed
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score,precision_score,recall_score,roc_curve

def metrics_sklearn(y_test, y_pred_):

    accuracy = accuracy_score(y_test, y_pred_)
    print('Accuracy：%.2f%%' % (accuracy * 100))

    precision = precision_score(y_test, y_pred_)
    print('Precision：%.2f%%' % (precision * 100))

    recall = recall_score(y_test, y_pred_)
    print('Recall：%.2f%%' % (recall * 100))

    f1 = f1_score(y_test, y_pred_)
    print('F1：%.2f%%' % (f1 * 100))

    auc = roc_auc_score(y_test, y_pred_)
    print('AUC：%.2f%%' % (auc * 100))

    # ks值
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_)
    ks = max(abs(fpr - tpr))
    print('KS：%.2f%%' % (ks * 100))


def XGBOOST( x_train,y_train,x_valid,y_valid, x_test,y_test, output_path):
    X_train = np.load(x_train)
    print(X_train.shape)
    df = pd.read_csv(y_train)
    Y_train = df['label'].to_numpy()
    print(Y_train.shape)
    X_valid = np.load(x_valid)
    print(X_valid.shape)
    df = pd.read_csv(y_valid)
    Y_valid = df['label'].to_numpy()
    print(Y_valid.shape)
    X_test = np.load(x_test)
    print(X_test.shape)
    df = pd.read_csv(y_test)
    Y_test = df['label'].to_numpy()
    print(Y_test.shape)
    #X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=2024,stratify=Y)#,stratify=Y
    xgb_train = xgboost.DMatrix(X_train, label=Y_train)
    xgb_valid = xgboost.DMatrix(X_valid, label=Y_valid)
    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'max_depth':10,
        'alpha':0.4,
        'lambda':1,
        'subsample':0.4,
        'gamma':0,
        'colsample_bytree':0.4,
        'min_child_weight':1,
        'seed': 2024,
        'eta': 0.01,
        'nthread':-1,
    }
    '''
    num_rounds = 1000
    early_stopping_rounds = 00
    cv_results = xgb.cv(params, xgb_train, num_boost_round=num_rounds, nfold=5,
                    metrics=['auc'], seed=24, early_stopping_rounds=early_stopping_rounds,
                    verbose_eval=500)

    print(cv_results)
    num_boost_rounds = cv_results.shape[0]
    '''
    model = xgboost.train(params, xgb_train, num_boost_round=30000,evals = [(xgb_valid, "test")], verbose_eval=500,early_stopping_rounds=200)
    #model = xgboost.train(params, xgb_train, 30000, evals = [(xgb_test, "test")], verbose_eval=1000,early_stopping_rounds=100)
    xgb_test = xgb.DMatrix(X_test)
    Y_pred = model.predict(xgb_test)
    threshold = 0.5
    Y_pred = (Y_pred>= threshold).astype(int)
    metrics_sklearn(Y_test, Y_pred)
    
    shap_values = shap.TreeExplainer(model).shap_values(X_test)
    print(shap_values.shape)
    np.save(os.path.join(output_path, "test_shap_value005.npy"), shap_values)
    '''
    shap_interaction_values = shap.TreeExplainer(model).shap_interaction_values(X_test)
    print(shap_interaction_values.shape)
    np.save(os.path.join(output_path, "test_interaction.npy"), shap_interaction_values)
    '''
    print("save")

if __name__ == '__main__':
    x_train= r"/pub/data/gaoss/New_Multi/code/XGboost/all_commom/train/XGBoost.npy"
    y_train=r"/pub/data/gaoss/New_Multi/code/XGboost/all_commom/train/labels.csv"
    x_valid=r"/pub/data/gaoss/New_Multi/code/XGboost/all_commom/valid/XGBoost.npy"
    y_valid=r"/pub/data/gaoss/New_Multi/code/XGboost/all_commom/valid/labels.csv"
    x_test=r"/pub/data/gaoss/New_Multi/code/XGboost/all_commom/test/XGBoost.npy"
    y_test=r"/pub/data/gaoss/New_Multi/code/XGboost/all_commom/test/labels.csv"
    output_path=r"/pub/data/gaoss/New_Multi/code/XGboost/all_commom/test/"
    setup_seed(2024)
    XGBOOST( x_train,y_train,x_valid,y_valid, x_test,y_test, output_path)




