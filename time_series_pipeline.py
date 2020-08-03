import numpy as np
import pandas as pd
import time
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)

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