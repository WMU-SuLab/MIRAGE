import shap
import os
import xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import cv
from utils import setup_seed
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score,precision_score,recall_score,roc_curve

def metrics_sklearn(y_valid, y_pred_):
    accuracy = accuracy_score(y_valid, y_pred_)
    print('Accuracy：%.2f%%' % (accuracy * 100))

    precision = precision_score(y_valid, y_pred_)
    print('Precision：%.2f%%' % (precision * 100))

    recall = recall_score(y_valid, y_pred_)
    print('Recall：%.2f%%' % (recall * 100))

    f1 = f1_score(y_valid, y_pred_)
    print('F1：%.2f%%' % (f1 * 100))

    auc = roc_auc_score(y_valid, y_pred_)
    print('AUC：%.2f%%' % (auc * 100))

    fpr, tpr, thresholds = roc_curve(y_valid, y_pred_)
    ks = max(abs(fpr - tpr))
    print('KS：%.2f%%' % (ks * 100))


def XGBOOST(top_snp_path, label_path, output_path):
    X = np.load(top_snp_path)
    df = pd.read_csv(label_path)
    Y = df['label'].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=2024,stratify=Y)
    xgb_train = xgboost.DMatrix(X_train, label=y_train)
    xgb_test = xgboost.DMatrix(X_test, label=y_test)
    params = {
              
        'learning_rate': 0.05,
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'verbosity': 1,
        'seed': 2024,
        'nthread': -1,
        'colsample_bytree': 0.7,
        'subsample': 0.5,
        'eval_metric': 'logloss',
        'scale_pos_weight':1,
        'max_depth':4,
        'n_estimators':400,
        'reg_alpha':0.1,
              
         }

          
    model_xgb = xgb.XGBClassifier(**params)
          
    param_grid = {
              
        #'n_estimators': [50,100, 200, 300,400,500],
        #'max_depth': [1,2,3, 4, 5, 6, 7],
        #'learning_rate': [0.01, 0.02, 0.05, 0.1],
        #'min_child_weight': [1, 2, 3, 5],
        #'lambda':[0,0.1,0.5,1, 1.5, 2],         
        #'scale_pos_weight': [1, 2, 3],
        #'subsample':[0.3,0.4,0.5,0.6,0.7,0.8],
        #'colsample_bytree':[0.3,0.4,0.5,0.6,0.7,0.8],
        #'reg_alpha': [0,0.05,0.1,0.2,0.5],
              
    }

    grid_search = GridSearchCV(
        estimator=model_xgb,
        param_grid=param_grid,
        scoring='neg_log_loss',
        cv=5,
        n_jobs=-1,
        verbose=1
              
    )
              
    grid_search.fit(X_train, y_train)
              
    print("Best parameters found: ", grid_search.best_params_)
              
    print("Best Log Loss score: ", -grid_search.best_score_)
          
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    print(classification_report(y_test, y_pred))
    metrics_sklearn(y_test, y_pred)

    shap_interaction_values = shap.TreeExplainer(best_model).shap_interaction_values(X)
    print(shap_interaction_values.shape)
    #np.save(os.path.join(output_path, "valid_interaction.npy"), shap_interaction_values)
    #print("save")
    
  
if __name__ == '__main__':
    top_snp_path=r"/pub/data/gaoss/New_Multi/code/XGboost/valid-shap/shap0.0007/XGBoost.npy"
    label_path=r"/pub/data/gaoss/New_Multi/code/data1/valid/labels.csv"
    output_path=r"/pub/data/gaoss/New_Multi/code/XGboost/valid-shap/shap0.0007/"
    setup_seed(2024)
    XGBOOST(top_snp_path, label_path, output_path)
