import lightgbm as lgb
import pandas as pd 
import os
import re
import numpy as np 
import warnings
from scipy import stats
from elo_params import params

def get_sample(n_rows):
    model_params = params()
    sample = pd.read_csv(f'{model_params.folder_path}sample_train.csv')
    sample = sample.sample(n=n_rows)
    return sample

def get_features(sample):
    drop_cols = [
        'target',
        'card_id',
        'new_authorized_flag_mean',
        'outliers']
    return [col for col in sample.columns if col not in drop_cols]

def get_models():
    model_params = params()
    return [file for file in os.listdir(path=model_params.folder_path) if re.match('lgbm.*.txt',file)]

def lgbm_ensemble_prediction(sample,features):
    pred = np.zeros(len(sample))
    model_params = params()
    for model in get_models():
        model = lgb.Booster(model_file=f"{model_params.folder_path}{model}")
        pred += model.predict(sample[features].apply(pd.to_numeric))/len(get_models())
    return pred

def format_predictions(pred,target):
    df = pd.DataFrame()
    df['predictions'] = pred
    df['target'] = target
    df['diff'] = np.abs(target-pred)
    return df

def get_metrics(sample):
    metric_columns = [
        'sum_purchase_amount_ratio',
        'hist_month_diff_mean',
        'mean_merchant_purchases',
        'hist_month_nunique',
        'hist_authorized_flag_mean',
    ]
    metric_df = sample[metric_columns]
    metric_df.columns = [
        'New Merchant Purchases / Historical Merchant Purchases',
        'Average Purchase Month (from today)',
        'Mean Purchases at Each Merchant',
        'Number of Unique Purchase Months',
        'Flagged Transactions Ratio',
    ]
    return dict(metric_df.iloc[0])

def get_sample_eval(sample):
    features=get_features(sample)
    model_params = params()
    pred = lgbm_ensemble_prediction(sample,features)
    target = pd.read_csv(f'{model_params.folder_path}target_values.csv').values
    return pred,target

def get_prediction(sample):
    features = get_features(sample)
    preds = lgbm_ensemble_prediction(sample,features)
    return preds

def get_percentile(pred):
    model_params = params()
    target = pd.read_csv(f'{model_params.folder_path}target_values.csv').values
    predicted_pct = stats.percentileofscore(target,pred)
    return predicted_pct

def demo_eval():
    sample = get_sample()
    features = get_features(sample)
    metrics = get_metrics(sample)
    pred,target = get_sample_eval(sample,features)
    pred_pct,actual_pct = get_percentiles(pred,target,sample)
    return sample[features],metrics,pred,target,pred_pct,actual_pct

# key metrics 
# Save target as csv
# output values as percentile

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    sample = get_sample(1)
    print(sample)