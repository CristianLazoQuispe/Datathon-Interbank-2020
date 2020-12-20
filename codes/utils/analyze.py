import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def feature_range(datasets,column,bins=None,n_bins = 10,onehot=False):
    # datasets is a list of dataframes  datasets = [df1,df2]
    if bins is None:
        q, bins = pd.qcut(datasets[0][column], n_bins, retbins=True)
    for df in datasets:
        cnt = 1
        df[column+'_'+str(cnt)] = df[column]*0
        for n,i in enumerate(bins[:-1]):
            if onehot:
                df[column+'_'+str(cnt)] = pd.to_numeric((df[column]>=bins[n])&(df[column]<bins[n+1]))*(n+1)
                cnt+=1
            else:
                cnd = (df[column]>=bins[n])&(df[column]<bins[n+1])
                df[column+'_'+str(cnt)]+=pd.to_numeric(cnd)*(n+1)
    return datasets


def intersection_columns(data1,data2,name1="submission",name2="y_train"):
    #e_intersection = np.intersect1d(data1,data2).shape[0]
    e_intersection = list(set(data1) & set(data2))
    e_different    = list(set(data1) - set(data2))
    e_intersection.sort()
    e_different.sort()
    print("{:<30}: ".format("# key_value in "+name1),len(data1))
    print("{:<30}: ".format("# key_value in "+name2),len(data2))
    print("{:<30}: ".format("# intersection            "),len(e_intersection))
    print("{:<30}: ".format("# different               "),len(e_different))
    print("{:<30}: ".format("example intersection      "),e_intersection[:2]+e_intersection[-2:])
    print("{:<30}: ".format("example different         "),e_different[:2]+e_different[-2:])
    return e_intersection,e_different

def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
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
        #else:
        #    df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df
