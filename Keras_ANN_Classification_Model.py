#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#### USER INPUT SECTION ###

# Global Variables #
global_source_name = "C:/Users/debas/Desktop/My Docs/Keras Code Development/Udemy/1_ANN/Classification_Model_Dataset.csv" 
global_id_var = 'Phone_Number'
global_dep_var = 'Churn'
global_postive_class = 'Yes'
global_test_split = 0.2
global_prob_cutoff = 0.75

param_k_fold_cv = 10
param_drop_out = 0.2
param_ann_optimizer = ['adam', 'rmsprop', 'sgd', 'nadam', 'adamax', 'adadelta']
param_epochs = [100]
param_batch_size = [50, 75, 100]


# In[ ]:


### IMPORT ALL NECCESSARY PACKAGES ###

from time import *
import numpy as np
import pandas as pd
import math
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import average_precision_score, roc_curve, roc_auc_score, balanced_accuracy_score
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
import matplotlib.pyplot as plt


# In[ ]:


### USER DEFINED FUNCTION: RAW DATA IMPORT ###

def data_import(source_name, id_var, dep_var):
    
    print("\nKindly Follow The Log For Tracing The Artificial Neural Network Modelling Process\n")
    print("\nStarting Data Import Process\n")

    import_start_time = time()
    
    df = pd.read_csv(source_name)
    df_x = df[df.columns[~df.columns.isin([id_var,dep_var])]]
    df_y = df.loc[:,dep_var].astype('category')
    numeric_cols = df_x.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df_x.select_dtypes(include=['object']).columns.tolist()
    
    final_data_import = [df_x,df_y,numeric_cols,categorical_cols]
    
    import_end_time = time()
    import_elapsed_time = (import_end_time - import_start_time)
    print("\nTime To Perform Data Import: %.3f Seconds\n" % import_elapsed_time)
      
    return(final_data_import)


# In[ ]:


### USER DEFINED FUNCTION: DATA PREPROCESSING ###

def data_preprocessing(df_x, numeric_cols, categorical_cols):
    
    print("\nStarting Data Pre-Processing Process\n")

    preprocess_start_time = time()
    
    # Preprocessing Step For Numeric Variables #
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(missing_values = np.nan, strategy = 'mean')),
                                          ('scaler', StandardScaler(with_mean = True, with_std = True))])
    # Preprocessing Step For Categorical Variables #
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                              ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    # Combining All Preprocessing Step #
    final_preprocessor = ColumnTransformer(transformers=[('preprocessing_num_col', numeric_transformer, numeric_cols),
                                                         ('preprocessing_cat_col', categorical_transformer, categorical_cols)])
    
    final_df_x = pd.DataFrame(final_preprocessor.fit_transform(df_x).toarray())
    # final_df_x = pd.DataFrame(final_preprocessor.fit_transform(df_x))
    
    preprocess_end_time = time()
    preprocess_elapsed_time = (preprocess_end_time - preprocess_start_time)
    print("\nTime To Perform Data Pre-Processing: %.3f Seconds\n" % preprocess_elapsed_time)
        
    return(final_df_x)


# In[ ]:


### USER DEFINED FUNCTION: TRAIN & TEST SAMPLE CREATION USING RANDOM SAMPLING ###

def random_sampling(x, y, split, pos_class):
    
    print("\nStarting Random Sampling Process\n")
    
    sampling_start_time = time()
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = split, random_state=1000, stratify=y)
    final_sampling = [x_train, x_test, y_train, y_test]
    
    hit_rate_train = (sum(y_train==pos_class)/len(y_train))*100
    hit_rate_test = (sum(y_test==pos_class)/len(y_test))*100
    
    print("\nHit Rate In Train Sample: %.3f Percent\n" % hit_rate_train)
    print("\nHit Rate In Test Sample: %.3f Percent\n" % hit_rate_test)

    sampling_end_time = time()
    sampling_elapsed_time = (sampling_end_time - sampling_start_time)
    print("\nTime To Perform Random Sampling For Train & Test Set: %.3f Seconds\n" % sampling_elapsed_time)
       
    return(final_sampling)


# In[ ]:


### USER DEFINED FUNCTION: ANN MODEL BUILDING ###

def ann_cross_val_model(train_df_x, train_y, test_df_x, test_y, drop_out, batch, epochs, optimizer, cv, prob_cut, pos_class):
    
    print("\nStarting ANN k-Fold Cross Validation Model Training Process\n")
    
    ann_start_time = time()
    
    train_n_col = train_df_x.shape[1]
    
    def init_ann(list_optimizer):
        
        # Initializes a empty ANN
        classifier = Sequential()
        
        # Adding first hidden layer
        classifier.add(Dense(units = math.floor((train_n_col+1)/2),
                             kernel_initializer='uniform',
                             activation = 'relu', 
                             bias_initializer='zeros',
                             input_dim = train_n_col))
        classifier.add(Dropout(p = drop_out))
        
        # Adding second hidden layer
        classifier.add(Dense(units = math.floor((train_n_col+1)/2),
                             kernel_initializer='uniform',
                             bias_initializer='zeros',
                             activation = 'relu'))
        classifier.add(Dropout(p = drop_out))
        
        # Adding output layer
        classifier.add(Dense(units = 1,
                             kernel_initializer='uniform',
                             bias_initializer='zeros',
                             activation = 'sigmoid'))
        
        # Compiling ANN architecture
        classifier.compile(optimizer = list_optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
        
        return classifier
    
    # Grid Search
    ann_classifier = KerasClassifier(build_fn = init_ann)
    hyper_parameters = {'batch_size': batch,'epochs': epochs,'list_optimizer': optimizer}
    grid_search = GridSearchCV(estimator = ann_classifier,
                               param_grid = hyper_parameters,
                               scoring = 'roc_auc',
                               cv = cv,
                               n_jobs = -1)
    grid_search = grid_search.fit(train_df_x, train_y)
    
    # Model Scoring On Test Sample
    y_pred_prob = grid_search.predict_proba(test_df_x)[:,1]
    y_pred_class = np.where(y_pred_prob>=prob_cut, 1, 0)
    y_actual_class = [1 if x == pos_class else 0 for x in test_y]
    
    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_actual_class, y_pred_prob)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve On Test Sample (ANN K-Fold Cross Validation)')
    plt.show()
    
    # Model Statistics
    bal_acc = round(balanced_accuracy_score(y_actual_class, y_pred_class, adjusted=True),2)*100   
    auc = round(roc_auc_score(y_actual_class, y_pred_class),2)*100
    prec_recall_score = round(average_precision_score(y_actual_class, y_pred_class, average = 'weighted'),2)*100
    ann_model_stat = pd.DataFrame({"Model Name" : ["Atificial Neural Network"],
                                   "Balanced Accuracy(%)": bal_acc,
                                   "AUC(%)": auc, 
                                   "Precision-Recall Score(%)": prec_recall_score})

    final_result = (grid_search,ann_model_stat)
    
    ann_end_time = time()
    ann_elapsed_time = (ann_end_time - ann_start_time)
    print("\nTime To Perform Train ANN k-Fold Cross Validation Model: %.3f Seconds\n" % ann_elapsed_time)
        
    return(final_result)


# In[ ]:


# Data Importing #
result_import = data_import(global_source_name, 
                            global_id_var, 
                            global_dep_var)    

# Data Pre-Processing #
result_preprocessed = data_preprocessing(result_import[0],
                                         result_import[2],
                                         result_import[3])

# Random Sampling of Test & Train Data #
result_sampling = random_sampling(result_preprocessed, 
                                  result_import[1], 
                                  global_test_split, 
                                  global_postive_class)

# Running ANN Model
model_ann_cross_val = ann_cross_val_model(result_sampling[0], 
                                          result_sampling[2], 
                                          result_sampling[1],
                                          result_sampling[3],
                                          param_drop_out,
                                          param_batch_size, 
                                          param_epochs, 
                                          param_ann_optimizer, 
                                          param_k_fold_cv,
                                          global_prob_cutoff,
                                          global_postive_class)


# In[ ]:


model_ann_cross_val[1]

