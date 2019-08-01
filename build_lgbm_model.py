# Basic Packages
from sklearn import model_selection, preprocessing, metrics
import pandas as pd
import numpy as np
import time
from elo_params import params
import warnings
warnings.filterwarnings('ignore')

# ML Packages
import lightgbm as lgb
from sklearn import linear_model
from sklearn import model_selection, preprocessing, metrics
pd.set_option('display.max_columns',None)
from sklearn.metrics import mean_squared_error

def load_train_test(params):
    train = pd.read_csv(f'{params.file_path}final_train.csv')
    test = pd.read_csv(f'{params.file_path}final_test.csv')
    return train,test

def get_features_and_target(train):
    drop_cols = [
        'target',
        'card_id',
        'new_authorized_flag_mean',
        'outliers']

    features = [col for col in train.columns if col not in drop_cols]
    target = train['target']
    return target,features

def encode_outliers(train):
    train['outliers'] = 0
    train.loc[train['target']<-30,'outliers'] = 1
    return train

def train_on_index(train,train_index,valid_index,lgbm_params,target,features):
    train_data = lgb.Dataset(train.iloc[train_index][features],label=target.iloc[train_index])
    val_data = lgb.Dataset(train.iloc[valid_index][features],label=target.iloc[valid_index])
    num_rounds = 10000
    lgb_model = lgb.train(lgbm_params,
                    train_data,
                    num_rounds,
                    valid_sets=[train_data,val_data],
                    verbose_eval=0,
                    early_stopping_rounds=200)
    return lgb_model

def get_feature_importance_df(lgbm_model,fold_,features):
        df = pd.DataFrame()
        df['feature'] = features
        df['importance'] = lgbm_model.feature_importance()
        df['fold'] = fold_ + 1
        return df

def train_lgbm(train,test,target,features,lgbm_params,splits,rs=15):
    
    folds = model_selection.StratifiedKFold(n_splits=splits,shuffle=True,random_state=rs)
    lgb_oof = np.zeros(len(train))
    lgb_pred = np.zeros(len(test))
    lgb_feature_importance = pd.DataFrame()

    for fold_, (train_index,valid_index) in enumerate(folds.split(train,train['outliers'].values)):
        print(f"fold number: {fold_ + 1}")
        
        # trains model using train and validation indices
        lgb_model = train_on_index(train,train_index,valid_index,lgbm_params,target,features)
        
        # get oof predictions
        lgb_oof[valid_index] = lgb_model.predict(train.iloc[valid_index][features],num_iteration=lgb_model.best_iteration)
        
        # get feature importances
        fold_importance_df = get_feature_importance_df(lgb_model,fold_,features)
        lgb_feature_importance = pd.concat([lgb_feature_importance,fold_importance_df],axis=0)
        
        # get predictions for each fold
        lgb_pred += lgb_model.predict(test[features],num_iteration=lgb_model.best_iteration)/folds.n_splits

        # save model
        lgb_model.save_model(f'lgbm_regressor_fold_{fold_+1}.txt',num_iteration=lgb_model.best_iteration)
    return lgb_oof,lgb_pred,lgb_feature_importance

def save_predictions(predictions,test):
    sub_df = pd.DataFrame({"card_id":test["card_id"].values})
    sub_df['target'] = predictions
    sub_df.to_csv("predictions.csv", index=False)
    print('predictions successfully saved')
    return

def build_model(params,save=True):
    train,test = load_train_test(params)
    target,features = get_features_and_target(train)
    train = encode_outliers(train)
    splits = 5
    lgb_oof,lgb_pred,lgb_feature_importance = train_lgbm(train,test,target,features,params.lgbm_params,splits,rs=15)
    print('RMSE:',np.sqrt(mean_squared_error(lgb_oof, target)))
    
    if save:
        save_predictions(lgb_pred,test)
        lgb_feature_importance.to_csv('model_feature_importance.csv')
    
    return lgb_oof,lgb_pred,lgb_feature_importance

if __name__ == "__main__":
    build_model(params())
