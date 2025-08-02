import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score,precision_score,recall_score,roc_curve
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings('ignore')
x_train_dir= r"/pub/data/gaoss/New_Multi/code/XGboost/shap_top_snp/top1000/train/XGBoost.npy"
y_train_dir=r"/pub/data/gaoss/New_Multi/code/XGboost/shap_top_snp/top1000/train/labels.csv"
x_test_dir=r"/pub/data/gaoss/New_Multi/code/XGboost/shap_top_snp/top1000/test/XGBoost.npy"
y_test_dir=r"/pub/data/gaoss/New_Multi/code/XGboost/shap_top_snp/top1000/test/labels.csv"
output_path = r"/pub/data/gaoss/New_Multi/code/XGboost/shap_top_snp/top1000/valid/"
x_train = np.load(x_train_dir)
print(x_train.shape)
df = pd.read_csv(y_train_dir)
y_train = df['label'].to_numpy()
print(y_train.shape)

x_test = np.load(x_test_dir)
print(x_test.shape)
df = pd.read_csv(y_test_dir)
y_test = df['label'].to_numpy()
print(y_test.shape)
#x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=2024)#,stratify=Y

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

def model_fit():
    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'max_depth': 8,
        'reg_lambda':0.5,
        'reg_lambda':2,
        'subsample':0.6,
        'colsample_bytree': 0.7,
        'min_child_weight':1,
        'seed': 2024,
        'eval_metric': 'logloss',
        'nthread':-1,
        'n_estimators':10000,
        'scale_pos_weight':1,
        'learning_rate':0.05
         }
    model = XGBClassifier(**params)#n_estimators=10000,max_depth=8,min_child_weight=1,gamma=0,colsample_bytree=1,subsample=1,reg_alpha=0.01,learning_rate=0.05
          
    param_grid = {
       #'min_child_weight': [1,2,3,4,5],                   #1
       #'n_estimators':[300,400,500,600,5000,10000],       
       #'lambda':[0,0.01,0.02,0.05,0.1,0.2,0.5,1,2],       
       #'subsample':[0.3,0.4,0.5,0.6,0.7,0.8,0.9],         #0.6
       #'scale_pos_weight': [1, 2, 3],                        
       #'colsample_bytree':[0.3,0.4,0.5,0.6,0.7,0.8,0.9],  
       'max_depth':[3,4,5,6,7,8,9,10,20,25,30],                 
       #'reg_lambda': [0,0.5,0.1,1, 1.5, 2],               
       #'learning_rate':[0.01,0.05,0.02,0.001,0.0001]        
         }
    grid_search = GridSearchCV(
        estimator=model,     
        param_grid=param_grid, 
        scoring='neg_log_loss',
        cv=5,
        n_jobs=-1,
        verbose=1
        )
    grid_search.fit(x_train, y_train)
    print("Best parameters found: ", grid_search.best_params_)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(x_test)
    metrics_sklearn(y_test, y_pred)
    '''
    shap_interaction_values = shap.TreeExplainer(best_model).shap_interaction_values(x_test)
    print(shap_interaction_values.shape)
    np.save(os.path.join(output_path, "interaction.npy"), shap_interaction_values)
    print("save")
    '''
def model_save_type(clf_model):
    clf_model.save_model('xgboost_classifier_model.model')
    clf = clf_model.get_booster()
    clf.dump_model('dump.txt')
if __name__ == '__main__':

    model_fit()
    # model_save_type(model_xgbclf)