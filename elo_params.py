import numpy as np 
class params:
    '''
    Class Containing all variables needed to execute project
    '''
    def __init__(self):
        '''
        variables:

        label_path - absolute path to labels (with label file name)
        image_path - absolute path to images (with image folder name)
        n_images - the number of images to train on
        batch_size - what batch size to use when training CNN
        epochs - How many passes through the data set when training model

        '''
        # Path to Folder
        self.file_path = "/Users/LiamRoberts/Desktop/DSProjects/ELO/data/"
        
        # Aggregate function for historical_transactions.csv
        self.agg_func = {
                        'authorized_flag' : ['mean'],
                        'card_id' : ['count'],  
                        'city_id' : ['nunique'],
                        'category_1' : ['sum','mean','std'],
                        'category_2_1.0':['sum','mean'],
                        'category_2_2.0': ['sum', 'mean'],
                        'category_2_3.0': ['sum', 'mean'],
                        'category_2_4.0': ['sum', 'mean'],
                        'category_2_5.0': ['sum', 'mean'],
                        'category_3_A': ['sum', 'mean'],
                        'category_3_B': ['sum', 'mean'],
                        'category_3_C': ['sum', 'mean'],
                        'month':['nunique'],
                        'hour':['mean'],
                        'weekofyear':['mean','nunique'],
                        'day':['nunique',np.ptp,'mean'],
                        'dayofweek':['mean'],
                        'weekend':['sum','mean'],
                        'duration':['min','mean','max'],
                        'price':['sum','mean','max','min','var'],
                        'amount_month_ratio':['max','min',np.ptp],
                        'installments': ['sum','min','max','var','mean'],
                        'merchant_category_id':['nunique'],
                        'merchant_id':['nunique'],
                        'purchase_amount':['sum','mean','max','min','var','median'],
                        'purchase_date':['max','min',np.ptp],
                        'time_since_purchase_date':['min','max','mean'],
                        'month_lag':['min','max','mean','var',np.ptp],
                        'month_diff':['mean','min','max',np.ptp,'var'],
                        'store_size':['min','max','mean'],
                        'city_size':['min','max','mean'],
                        'state_size':['min','max','mean'],
                        'subsector_size':['min','max','mean'],
                        'category_size':['min','max','mean'],
           }
        
        # Aggregate function for new_merchant_transactions.csv
        self.new_agg_func = {
                        'card_id' : ['count'],
                        'city_id' : ['nunique'],
                        'category_1' : ['sum','mean'],
                        'category_2_1.0':['sum','mean'],
                        'category_2_2.0': ['sum', 'mean'],
                        'category_2_3.0': ['sum', 'mean'],
                        'category_2_4.0': ['sum', 'mean'],
                        'category_2_5.0': ['sum', 'mean'],
                        'category_3_A': ['sum', 'mean'],
                        'category_3_B': ['sum', 'mean'],
                        'category_3_C': ['sum', 'mean'],
                        'month':['nunique'],
                        'weekofyear':['nunique'],
                        'day':['nunique',np.ptp,'mean'],
                        'dayofweek':['mean'],
                        'duration':['min','max'],
                        'price':['sum','mean','max','min','var'],
                        'amount_month_ratio':['max','min',np.ptp],
                        'installments': ['sum','min','max','var','mean'],
                        'merchant_category_id':['nunique'],
                        'merchant_id':['nunique'],
                        'purchase_amount':['sum','mean','max','min','var'],
                        'purchase_date':['max','min',np.ptp],
                        'time_since_purchase_date':['min','max','mean'],
                        'weekend':['sum','mean'],
                        'month_lag':['min','max','mean','var',np.ptp],
                        'month_diff':['mean','min','max',np.ptp]
                    }
        
        # Features Found Not to Be Useful in PART IV
        self.drop_features = [
                 'feature_2',
                 'feature_3',
                 'hist_city_id_nunique',
                 'hist_category_2_1.0_mean',
                 'hist_category_2_2.0_sum',
                 'hist_category_2_2.0_mean',
                 'hist_category_2_3.0_sum',
                 'hist_category_2_3.0_mean',
                 'hist_category_2_5.0_sum',
                 'hist_category_2_5.0_mean',
                 'hist_day_ptp',
                 'hist_duration_mean',
                 'hist_amount_month_ratio_ptp',
                 'hist_installments_min',
                 'hist_purchase_amount_sum',
                 'hist_purchase_amount_var',
                 'hist_month_diff_ptp',
                 'hist_store_size_max',
                 'hist_city_size_max',
                 'hist_state_size_min',
                 'hist_state_size_max',
                 'new_city_id_nunique',
                 'new_category_2_1.0_sum',
                 'new_category_2_2.0_sum',
                 'new_category_2_2.0_mean',
                 'new_category_2_3.0_sum',
                 'new_category_2_3.0_mean',
                 'new_category_2_4.0_sum',
                 'new_category_2_4.0_mean',
                 'new_category_2_5.0_sum',
                 'new_category_2_5.0_mean',
                 'new_category_3_A_mean',
                 'new_category_3_B_sum',
                 'new_category_3_C_sum',
                 'new_category_3_C_mean',
                 'new_dayofweek_mean',
                 'new_installments_min',
                 'new_installments_max',
                 'new_installments_var',
                 'new_weekend_sum',
                 'new_weekend_mean',
                 'new_month_diff_min',
                 'new_month_diff_ptp',
                 'quarter',
                 'min_duration_ratio',
                 'max_purchase_product']
        self.lgbm_params = {
                'task': 'train',
                'boosting': 'gbdt',
                'objective': 'regression',
                'metric': 'rmse',
                'learning_rate': 0.01,
                'subsample': 0.95,
                'max_depth': 7,
                'num_leaves': 64,
                'min_child_weight': 42,
                'reg_alpha': 9.7,
                'colsample_bytree': 0.57,
                'min_split_gain': 9.8,
                'reg_lambda': 8.25,
                'min_data_in_leaf': 21,
                'verbose': -1,
                'seed':2333,
                }
        return
