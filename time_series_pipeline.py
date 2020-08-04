import numpy as np
import pandas as pd
import time
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
from sklearn.preprocessing import LabelEncoder
import gc
from time_series_pipeline import *
from tqdm import tqdm
from scipy import stats
from catboost import CatBoostRegressor
import lightgbm as lgb
from sklearn import preprocessing, metrics
from sklearn.model_selection import TimeSeriesSplit
from itertools import product

def load_data():
    '''load_datas
    '''
    start = time.time()
    items = pd.read_csv('../data/items.csv')
    print('items has {} rows and {} columns'.format(items.shape[0], items.shape[1]))
    shops = pd.read_csv('../data/shops.csv')
    print('shops has {} rows and {} columns'.format(shops.shape[0], shops.shape[1]))
    cats = pd.read_csv('../data/item_categories.csv')
    print('cats has {} rows and {} columns'.format(cats.shape[0], cats.shape[1]))
    train = pd.read_csv('../data/sales_train.csv')
    print('train has {} rows and {} columns'.format(train.shape[0], train.shape[1]))
    test  = pd.read_csv('../data/test.csv').set_index('ID')
    print('test has {} rows and {} columns'.format(test.shape[0], test.shape[1]))
    print('loading data costs %.2f seconds'%(time.time() - start))
    return items, shops, cats, train, test

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def data_transform(items, shops, cats, train, test):
    '''data transformation
    '''
    start = time.time()
    train = train[(train['item_price'] < 300000 ) & (train['item_cnt_day'] < 1000)]
    train = train[train['item_price'] > 0]
    median = train[(train['shop_id'] == 32)&(train['item_id'] == 2973)&(train['date_block_num'] == 4)&(train['item_price'] > 0)].item_price.median()
    train.loc[train['item_price'] < 0, 'item_price'] = median
    train.loc[train['item_cnt_day'] < 1, 'item_cnt_day'] = 0
    train.loc[train['shop_id'] == 0, 'shop_id'] = 57
    test.loc[test['shop_id'] == 0, 'shop_id'] = 57
    # Якутск ТЦ "Центральный"
    train.loc[train['shop_id'] == 1, 'shop_id'] = 58
    test.loc[test['shop_id'] == 1, 'shop_id'] = 58
    # Жуковский ул. Чкалова 39м²
    train.loc[train['shop_id'] == 10, 'shop_id'] = 11
    test.loc[test['shop_id'] == 10, 'shop_id'] = 11
    test['id'] = test['shop_id'].astype(str) + '_' + test['item_id'].astype(str)

    shops.loc[shops['shop_name'] == 'Сергиев Посад ТЦ "7Я"', 'shop_name'] = 'СергиевПосад ТЦ "7Я"'
    shops['city'] = shops['shop_name'].str.split(' ').transform(lambda x: x[0])
    shops.loc[shops['city'] == '!Якутск', 'city'] = 'Якутск'
    shops['city_code'] = LabelEncoder().fit_transform(shops['city'])
    shops = shops[['shop_id','city_code']]

    cats['split'] = cats['item_category_name'].str.split('-')
    cats['type'] = cats['split'].transform(lambda x: x[0].strip())
    cats['type_code'] = LabelEncoder().fit_transform(cats['type']) # 类型
    cats['subtype'] = cats['split'].map(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())
    cats['subtype_code'] = LabelEncoder().fit_transform(cats['subtype']) # 子类型
    cats = cats[['item_category_id','type_code', 'subtype_code']]
    items.drop(['item_name'], axis = 1, inplace = True)
    ##################### 数据增强
    matrix = [] 
    cols = ['date_block_num','shop_id','item_id']
    for i in range(34):
        sales = train[train.date_block_num==i]
        matrix.append(np.array(list(product([i], 
                                        sales.shop_id.unique(), 
                                        sales.item_id.unique())), 
                                        dtype = 'int16'))
    
    matrix = pd.DataFrame(np.vstack(matrix), columns=cols)
    matrix.sort_values(cols,inplace = True)
    matrix['id'] = matrix['shop_id'].astype(str) + '_' + matrix['item_id'].astype(str)
    ###########加入测试集
    test['date_block_num'] = 34
    test['date_block_num'] = test['date_block_num'].astype(np.int8)
    test['shop_id'] = test['shop_id'].astype(np.int8)
    test['item_id'] = test['item_id'].astype(np.int16)
    matrix = pd.concat([matrix, test], ignore_index = True, sort = False)
    #matrix.fillna(0, inplace = True)
    # 将日数据汇总为月数据
    df = pd.DataFrame() 
    grouped = train.groupby(['date_block_num','shop_id','item_id'])
    df['item_cnt_month'] = grouped['item_cnt_day'].sum()
    df.reset_index(inplace = True) 
    matrix = pd.merge(matrix, df, on = cols, how = 'left')
    matrix['item_cnt_month'] = (matrix['item_cnt_month']
                                .fillna(0)
                                .clip(0,20) 
                                .astype(np.float16))
    
    matrix = pd.merge(matrix, shops, on = ['shop_id'], how = 'left')
    matrix = pd.merge(matrix, items, on = ['item_id'], how = 'left')
    matrix = pd.merge(matrix, cats, on = ['item_category_id'], how = 'left')
    
    grouped = train.groupby(['date_block_num','shop_id','item_id'])['item_price'].mean()
    grouped = pd.DataFrame(grouped)
    grouped.reset_index(inplace = True)
    matrix = pd.merge(matrix, grouped, on = ['date_block_num','shop_id','item_id'], how = 'left')
    matrix['item_price'] = matrix.groupby(['id'])['item_price'].transform(lambda x: x.fillna(x.median()))
    matrix['item_price'] = matrix['item_price'].astype(np.float32)
    del cats, grouped, items, sales, shops, test, train
    gc.collect()
    print('data has {} rows and {} columns'.format(df.shape[0], df.shape[1]))
    print('The program costs %.2f seconds'%(time.time() - start))
    return matrix

#df.reset_index(inplace=True)
def groupby_shift(df, col, groupcol, shift_n, fill_na = np.nan):
    '''
    apply fast groupby shift
    df: data 
    col: column need to be shift 
    shift: n
    fill_na: na filled value
    '''
    rown = df.groupby(groupcol).size().cumsum()
    rowno = list(df.groupby(groupcol).size().cumsum()) # 获取每分组第一个元素的index
    lagged_col = df[col].shift(shift_n) # 不分组滚动
    na_rows = [i for i in range(shift_n)] # 初始化为缺失值的index
    #print(na_rows)
    for i in rowno:
        if i == rowno[len(rowno)-1]: # 最后一个index直接跳过不然会超出最大index
            continue 
        else:
            new = [i + j for j in range(shift_n)] # 将每组最开始的shift_n个值变成nan
            na_rows.extend(new) # 加入列表
    na_rows = list(set(na_rows)) # 去除重复值
    na_rows = [i for i in na_rows if i <= len(lagged_col) - 1] # 防止超出最大index
    #print(na_rows)
    lagged_col.iloc[na_rows] = fill_na # 变成nan
    return lagged_col

def lag_feature(df, lags, col):
    tmp = df[['date_block_num','shop_id','item_id',col]]
    for i in lags:
        shifted = tmp.copy()
        shifted.columns = ['date_block_num','shop_id','item_id', col+'_lag_'+str(i)]
        shifted['date_block_num'] += i
        df = pd.merge(df, shifted, on=['date_block_num','shop_id','item_id'], how='left')
    return df

def fill_na(df):
    for col in df.columns:
        if ('_lag_' in col):
            df[col].fillna(0, inplace=True) 
        if ('price' in col):
            df[col] = df[col].transform(lambda x: x.fillna(x.median()))         
    return df

features = ['date_block_num',
            'month',
            'shop_id',
            'item_id',
            'city_code', 
            'item_category_id', 
            'type_code', 
            'subtype_code',
            'lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12',
       'price_shift_1', 'price_shift_2', 'price_shift_3', 'price_shift_6',
       'price_shift_12', 'date_cnt_lag_1', 'date_item_lag_1',
       'date_item_lag_2', 'date_item_lag_3', 'date_item_lag_6',
       'date_item_lag_12', 'date_shop_lag_1', 'date_shop_lag_2',
       'date_shop_lag_3', 'date_shop_lag_6', 'date_shop_lag_12',
       'date_cat_lag_1']
cat_features = ['month', 'shop_id','item_id','city_code', 'item_category_id', 'type_code', 'subtype_code']

def train_catboost(df):
    '''train a catboost
    '''
    df.sort_values(['date_block_num','shop_id','item_id'], inplace = True)
    x_train = df[df['date_block_num'] < 34]
    y_train = x_train['item_cnt_month'].astype(np.float32)
    test = df[df['date_block_num'] == 34]
    
    folds = TimeSeriesSplit(n_splits = 3) # use TimeSeriesSplit cv
    splits = folds.split(x_train, y_train)
    val_pred = np.zeros(len(x_train))
    test_pred = np.zeros(len(test))
    for fold, (trn_idx, val_idx) in enumerate(splits):
        print(f'Training fold {fold + 1}')
          
        train_set = x_train.iloc[trn_idx][features]
        y_tra = y_train.iloc[trn_idx]
        val_set = x_train.iloc[val_idx][features]
        y_val = y_train.iloc[val_idx]

        model = CatBoostRegressor(iterations = 1500,
                              learning_rate = 0.03,
                              depth = 5,
                              loss_function = 'RMSE',
                              eval_metric = 'RMSE',
                              random_seed = 42,
                              bagging_temperature = 0.3,
                              od_type = 'Iter',
                              metric_period = 50,
                              od_wait = 28)
        model.fit(train_set, y_tra, 
              eval_set = (val_set, y_val),
              use_best_model = True, 
              cat_features = cat_features,
              verbose = 50)
        
        val_pred[val_idx] = model.predict(x_train.iloc[val_idx][features]) # prediction
        #test_pred += model.predict(test[features]) / 3 # calculate mean prediction value of 3 models
        print('-' * 50)
        print('\n')
    test_pred = model.predict(test[features])  
    val_rmse = np.sqrt(metrics.mean_squared_error(y_train, val_pred))
    print('Our out of folds rmse is {:.4f}'.format(val_rmse))
    return test_pred

def train_lightgbm(df):
    '''train a lightgbm
    '''
    df.sort_values(['date_block_num','shop_id','item_id'], inplace = True)
    x_train = df[df['date_block_num'] < 34]
    y_train = x_train['item_cnt_month'].astype(np.float32)
    test = df[df['date_block_num'] == 34]
    
    folds = TimeSeriesSplit(n_splits = 3) # use TimeSeriesSplit cv
    splits = folds.split(x_train, y_train)
    val_pred = np.zeros(len(x_train))
    test_pred = np.zeros(len(test))
    params = {
        'boosting_type': 'gbdt',
        'metric': 'rmse',
        'objective': 'rmse', # loss function
        'seed': 225,
        'learning_rate': 0.01,
        'lambda': 0.4, # l2 regularization
        'reg_alpha': 0.4, # l1 regularization
        'max_depth': 4, # max depth of decision trees
        'num_leaves': 68, #  number of leaves
        'bagging_fraction': 0.7, # bootstrap sampling
        'bagging_freq' : 1,
        'colsample_bytree': 0.7 # feature sampling
    }
    for fold, (trn_idx, val_idx) in enumerate(splits):
        print(f'Training fold {fold + 1}')
        
        train_set = lgb.Dataset(x_train.iloc[trn_idx][features], 
                                y_train.iloc[trn_idx], 
                                categorical_feature = cat_features)
        
        val_set = lgb.Dataset(x_train.iloc[val_idx][features], 
                              y_train.iloc[val_idx], 
                              categorical_feature = cat_features)

        model = lgb.train(params, train_set, 
                          num_boost_round = 1500, 
                          early_stopping_rounds = 100, 
                          valid_sets = [val_set], 
                          verbose_eval = 50)
        
        val_pred[val_idx] = model.predict(x_train.iloc[val_idx][features]) # prediction
        test_pred += model.predict(test[features]) / 3 # calculate mean prediction value of 3 models
        print('-' * 50)
        print('\n')
    #test_pred = model.predict(test[features])     
    val_rmse = np.sqrt(metrics.mean_squared_error(y_train, val_pred))
    print('Our out of folds rmse is {:.4f}'.format(val_rmse))
    return test_pred

def make_output(test_pred):
    '''make prediction
    '''
    test  = pd.read_csv('../data/test.csv')
    test.sort_values(['shop_id','item_id'], inplace = True)
    submission = pd.DataFrame({'ID': test['ID'],
                              'item_cnt_month': test_pred.clip(0,20)})
    #submission = pd.DataFrame({'ID': range(0,len(test_pred)),'item_cnt_month': test_pred.clip(0,20)})
    print(submission.head(15))
    submission.to_csv('../output/cat_submission.csv', index = False)
    return submission