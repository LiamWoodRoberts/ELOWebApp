
import preprocess
from elo_params import params
import numpy as np
import pandas as pd 
import datetime

def get_aggregate_features(agg_func,df,prefix):
    
    # Aggregate columns based on dictionary passed to agg function
    gdf = df.groupby(['card_id']).agg(agg_func)

    # Rename columns before joining train/test set
    gdf.columns = [prefix+'_'+'_'.join(col).strip() for col in gdf.columns.values]
    return gdf

def get_repeat_purchase_features(df):
    gdf = pd.DataFrame(df.groupby(['card_id','purchase_amount']).size().max(level=0))
    gdf.columns = ['max_repeat_purchases']
    gdf['mean_repeat_purchases'] = df.groupby(['card_id','purchase_amount']).size().mean(level=0)
    gdf['mean_merchant_purchases'] = df.groupby(['card_id','merchant_id']).size().mean(level=0)
    gdf['mean_category_purchases'] = df.groupby(['card_id','merchant_category_id']).size().mean(level=0)
    gdf['mean_monthly_purchases'] = df.groupby(['card_id','month_diff']).size().mean(level=0)
    return gdf

def merge_features(df,feature_dfs):
    for gdf in feature_dfs:
        df = pd.merge(df,gdf,on="card_id",how="left")
    return df

def fill_zero_purchases(df):
    zero_cols = ['new_purchase_amount_mean',
                    'new_purchase_amount_sum',
                    'new_purchase_amount_max',
                    'new_duration_max',
                    'new_duration_min',
                    'new_amount_month_ratio_min',
                    'new_amount_month_ratio_max',
                    'new_card_id_count',
                   ]
    for col in zero_cols:
        df[col] = df[col].fillna(0)
    return df

def add_dt_features(df):
    df['hist_purchase_date_uptonow'] = (datetime.datetime.today() - 
                                      df['hist_purchase_date_max']).dt.days
    df['new_purchase_date_uptonow'] = (datetime.datetime.today() - 
                                      df['new_purchase_date_max']).dt.days
    
    df['quarter'] = df['first_active_month'].dt.quarter
    df['elapsed_time'] = (datetime.datetime.today() - df['first_active_month']).dt.days
    df['days_feature1'] = df['elapsed_time'] * df['feature_1']
    df['days_feature2'] = df['elapsed_time'] * df['feature_2']
    df['days_feature3'] = df['elapsed_time'] * df['feature_3']
    df['days_feature1_ratio'] = df['feature_1'] / df['elapsed_time']
    df['days_feature2_ratio'] = df['feature_2'] / df['elapsed_time']
    df['days_feature3_ratio'] = df['feature_3'] / df['elapsed_time']
    
    dt_features = ['hist_purchase_date_max','hist_purchase_date_min',
               'new_purchase_date_max','new_purchase_date_min','hist_purchase_date_ptp','new_purchase_date_ptp']
    
    # Models cannot use datetime features so they are encoded here as int64s
    for feature in dt_features:
        df[feature] = df[feature].astype(np.int64)*1e-9
    
    df['first_month'] = df['first_active_month'].dt.month
    df['first_year'] = df['first_active_month'].dt.year
    df.drop(columns = ['first_active_month'],inplace=True)
    return df

def add_interaction_features(df):
    # https://www.kaggle.com/mfjwr1/simple-lightgbm-without-blending
    df['category_1_mean'] = df['new_category_1_mean']+df['hist_category_1_mean']
    df['new_CLV'] = df['new_card_id_count'] * df['new_purchase_amount_sum'] / df['new_month_diff_mean']
    df['hist_CLV'] = df['hist_card_id_count'] * df['hist_purchase_amount_sum'] / df['hist_month_diff_mean']
        
    # Ratios
    df['transactions_ratio'] = df['new_card_id_count']/df['hist_card_id_count']
    df['mean_purchase_ratio']  = df['new_purchase_amount_mean']/df['hist_purchase_amount_mean']
    df['max_purchase_ratio'] = df['new_purchase_amount_max']/df['hist_purchase_amount_max']
    df['sum_purchase_amount_ratio'] = df['new_purchase_amount_sum'] / df['hist_purchase_amount_sum']
    df['mean_month_lag_ratio'] = df['new_month_lag_mean']/df['hist_month_lag_mean']
    df['mean_month_diff_ratio'] = df['new_month_diff_mean'] / df['hist_month_diff_mean']
    df['sum_installments_ratio'] = df['new_installments_sum'] / df['hist_installments_sum']
    df['mean_installments_ratio'] = df['new_installments_mean'] / df['hist_installments_mean']
    df['min_duration_ratio'] = df['new_duration_min']/df['hist_duration_min']

    # Products
    df['transactions_product'] = df['new_card_id_count']*df['hist_card_id_count']
    df['mean_purchase_product']  = df['new_purchase_amount_mean']*df['hist_purchase_amount_mean']
    df['max_purchase_product'] = df['new_purchase_amount_max']*df['hist_purchase_amount_max']
    df['sum_purchase_amount_product'] = df['new_purchase_amount_sum']*df['hist_purchase_amount_sum']
    df['mean_month_lag_product'] = df['new_month_lag_mean']*df['hist_month_lag_mean']
    df['mean_month_diff_product'] = df['new_month_diff_mean']*df['hist_month_diff_mean']
    df['sum_installments_product'] = df['new_installments_sum']*df['hist_installments_sum']
    df['mean_installments_product'] = df['new_installments_mean']*df['hist_installments_mean']
    df['min_duration_product'] = df['new_duration_min']*df['hist_duration_min']
    
    # Weighted Time Features
    df['hist_min_duration_weighted'] = df['hist_duration_min']*df['hist_card_id_count']
    df['hist_max_duration_weighted'] = df['hist_duration_max']*df['hist_card_id_count']
    df['hist_time_since_purchase_date_min_weighted'] = df['hist_time_since_purchase_date_min']*df['hist_card_id_count']
    
    df['new_min_duration_weighted'] = df['new_duration_min']*df['new_card_id_count']
    df['new_max_duration_weighted'] = df['new_duration_max']*df['new_card_id_count']
    df['new_time_since_purchase_date_min_weighted'] = df['new_time_since_purchase_date_min']*df['new_card_id_count']
    
    # Additional Features
    df['sum_price_ratio'] = df['new_price_sum'] / df['hist_price_sum']
    df['mean_price_ratio'] = df['new_price_mean'] / df['hist_price_mean']
    df['CLV_Ratio'] = df['new_CLV']/df['hist_CLV']
    return df

def get_trans_features(new_trans_df,hist_df,params):
    
    # Get historical Features
    hist_agg_features = get_aggregate_features(params.agg_func,hist_df,'hist')
    new_agg_features = get_aggregate_features(params.new_agg_func,new_trans_df,'new')
    repeat_features = get_repeat_purchase_features(hist_df)
    feature_dfs = [hist_agg_features,new_agg_features,repeat_features]
    return feature_dfs

def add_trans_features(df,feature_dfs):
    df = merge_features(df,feature_dfs)
    df = fill_zero_purchases(df)
    return df

def get_final_model_data(params,testing=True,save=True):
    # Load and preprocess initial csv files
    file_path = params.file_path
    train_df,test_df,new_trans_df,hist_df = preprocess.get_preprocessed_data(file_path,testing=testing)
    
    print('Getting Aggregate Features...')
    feature_dfs = get_trans_features(new_trans_df,hist_df,params)

    print('Adding Features for Train Set...')
    # Add features to train dataframe
    train_df = add_trans_features(train_df,feature_dfs)
    train_df = add_dt_features(train_df)
    train_df = add_interaction_features(train_df)
    train_df = train_df.drop(columns=params.drop_features)

    print('Adding Features for Test Set...')
    # Add features to test dataframe
    test_df = add_trans_features(test_df,feature_dfs)
    test_df = add_dt_features(test_df)
    test_df = add_interaction_features(test_df)
    test_df = test_df.drop(columns = params.drop_features)
    
    if save:
        train_df.to_csv(f'{params.file_path}final_train.csv')
        test_df.to_csv(f'{params.file_path}final_test.csv')
        print('Data Successfully Saved')
    return train_df,test_df

if __name__ == "__main__":
    train_df,test_df = get_final_model_data(params(),testing=False,save=True)
    print("Data Ready for Model Predictions")
    print("Train Shape:",train_df.shape)
    print("Test Shape:",test_df.shape)
