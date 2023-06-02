import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
from numpy.random import seed
from sklearn.feature_selection import mutual_info_regression

def seperate_cat_num_cols(data):
    Categorical_col = []
    Numerical_col = []
    
    for i in range(data.shape[1]):
        if data[data.columns[i]].isnull().values.any()==True:
            if (data[data.columns[i]].dtypes == 'O'):
                Categorical_col.append(data.columns[i])
            else:
                Numerical_col.append(data.columns[i])
    return Numerical_col,Categorical_col


def prepare_data_for_corr(data):

    has_nan = data.isna().any().any()

    if has_nan:
    
        Numerical_col = seperate_cat_num_cols(data)[0]
        Categorical_col = seperate_cat_num_cols(data)[1]

        cat_col_index,num_col_index = [],[]            
        for i in Categorical_col:
            cat_col_index.append(data.columns.tolist().index(i))
        for i in Numerical_col:
            num_col_index.append(data.columns.tolist().index(i))

        #Factorize categorical feature
        for i in range(len(Categorical_col)):
            data[Categorical_col[i]] = pd.factorize(data[Categorical_col[i]])[0]

        data = data.replace(-1, np.nan)

        num_data_for_corr = data.copy()
        imputer1 = SimpleImputer(missing_values=np.nan, strategy='mean')
        imputer1 = imputer1.fit(num_data_for_corr.values[:,num_col_index])
        
        num_data_for_corr = imputer1.transform(num_data_for_corr.values[:,num_col_index])
        num_data_for_corr = pd.DataFrame(num_data_for_corr,columns = Numerical_col)
        
        if len(cat_col_index) > 0:
            Cat_data_for_corr = data.copy()
            imputer2 = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
            imputer2 = imputer2.fit(Cat_data_for_corr.values[:,cat_col_index])
            Cat_data_for_corr = imputer2.transform(Cat_data_for_corr.values[:,cat_col_index])
            Cat_data_for_corr = pd.DataFrame(Cat_data_for_corr,columns=Categorical_col)
            data_for_corr = pd.concat([num_data_for_corr, Cat_data_for_corr], axis=1)
        
        else:
            data_for_corr = num_data_for_corr
            
        data_for_corr = data_for_corr.astype(float)
    else:
        data_for_corr = data

    return data_for_corr


def most_correlated_columns(data,corr_coef):
    data = prepare_data_for_corr(data)
    corr_table = data.corr('pearson')
    corr_table = pd.DataFrame(corr_table)
    corr_table = corr_table.rename_axis().reset_index()

    correlated_features = {}
    for i in range(1,corr_table.shape[0]+1):
        a=[]
        for j in range(corr_table.shape[0]):

            if i != j:

                if abs(corr_table[corr_table.columns[i]][j]) > corr_coef:

                    a.append(corr_table['index'][j])
        
        correlated_features[corr_table.columns[i]] = a
    return correlated_features

def mutual_information(Data,n_nearest_features):
    Data = prepare_data_for_corr(Data)
    mic_ordered = {}
    seed(21)
    for col1 in Data.columns:
        HighInformation = {}
        for col2 in Data.columns:
            
            if col1 != col2:
                col1_val = Data[col1].values
                col2_val = Data[col2].values

                score = mutual_info_regression(col1_val.reshape(-1, 1), col2_val)
                #if score[0] >= miScore:
                    #HighInformation.append(col2)

                HighInformation[col2] = score[0]
                   
        sorted_mic = {k: v for k, v in sorted(HighInformation.items(),reverse=True, key=lambda item: item[1])}
        selected_col = list(sorted_mic.keys())[:n_nearest_features]
        mic_ordered[col1] = selected_col

    return mic_ordered


def fill_nan_numeric_cols(data,regression_estimator,S):
    
    data_org = data.copy()
    Numerical_col = seperate_cat_num_cols(data)[0]
    Categorical_col = seperate_cat_num_cols(data)[1]
    correlated_features = S
    
    for i in range(len(Categorical_col)):
        data[Categorical_col[i]] = pd.factorize(data[Categorical_col[i]])[0]

    data = data.replace(-1, np.nan)
    
    for i in Numerical_col:
        
        columns = []
        m = correlated_features[i]
        #m.remove(i)
        
        num_data = data[m].copy()
        
        label = data[i]
        """need_scale = []  #If you need to scale the numerical columns you can uncomment this and work a little on it. 
        for j in correlated_features[i]:
            if j in Numerical_col:
                need_scale.append(j)"""
        
        #Scaler = scaler    
        #num_data[need_scale] = Scaler.fit_transform(num_data[need_scale])
        

        #print(data)
        nan=[]
        fill = [] 
        for k in range(data.shape[0]):
            
            if data[i].isnull()[k] ==True: 
                nan.append(k)
            else:
                fill.append(k)
        
        #Fill nan in num_data with SimpleImputer
        
        imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        imputer = imputer.fit(num_data.values)
        transform = imputer.transform(num_data.values)
        num_data = pd.DataFrame(transform)
        num_data[i] = label
        
        #Train regression model
        
        X = num_data.values[fill,:-1]
        Y = num_data.values[fill,-1]
    
        X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2, random_state=0)
        reg = regression_estimator
        reg.fit(X_train, y_train)  
        predicted = reg.predict(num_data.values[nan,:-1])
        
        k = 0
        for t in nan:
            data[i][t] = predicted[k]
            k = k+1

    data.drop(Categorical_col,axis=1,inplace=True)
    
    for i in Categorical_col:
        data[i] = data_org[i]
        
    return data


def fill_nan_categoric_cols(data,classifiation_estimator,S):
    
    Numerical_col = seperate_cat_num_cols(data)[0]
    Categorical_col = seperate_cat_num_cols(data)[1]
    correlated_features = S
    
    for i in range(len(Categorical_col)):
        data[Categorical_col[i]] = pd.factorize(data[Categorical_col[i]])[0]
        
    data = data.replace(-1, np.nan)
    data_org = data.copy()
   
    for i in Categorical_col:
        columns = []
        m = correlated_features[i]
        #m.remove(i)
       
        cat_data = data[m].copy()
        label = data[i]
        
        nan=[]
        fill = [] 
        for k in range(data.shape[0]):
        
            if data[i].isnull()[k] ==True:
                nan.append(k)
            else:
                fill.append(k)
        
        imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        imputer = imputer.fit(cat_data.values)
        transform = imputer.transform(cat_data.values)
        cat_data = pd.DataFrame(transform)
        cat_data[i] = label
        
        X = cat_data.values[fill,:-1]
        Y = cat_data.values[fill,-1]

        X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2, random_state=0)
        classify = classifiation_estimator
        classify.fit(X_train, y_train)  
        predicted = classify.predict(cat_data.values[nan,:-1])
        
        
      #Fill nan for i column with predicted valus
        
        k = 0
        for t in nan:
            data[i][t] = predicted[k]
            k = k+1

    return data

def Imputer(train,label,similarity,regression_estimator,classifiation_estimator):
    
    All = train
    Label = All[label]
    Label = Label.reset_index()
    All.drop([label],axis=1,inplace=True)
    All = All.reset_index(drop=True)
    #All.drop(['index'],axis=1,inplace=True)
    All_org = All.copy()
    #if the user choose a number < 1 for similarity, it considers as a correlation coeficient
    if similarity < 1:
        s = most_correlated_columns(All,similarity)
    
    #if the user choose a integer number >=1 for similarity, it considers as the n_nearest_features
    else:
        s = mutual_information(All,similarity)
    numeric = fill_nan_numeric_cols(All_org,regression_estimator(),s)
    categoric = fill_nan_categoric_cols(numeric,classifiation_estimator(),s)
    categoric[label] = Label[label]

    return categoric