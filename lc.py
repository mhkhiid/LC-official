import re
import json
import os
import pandas as pd
import numpy as np
import random as rand
import matplotlib.pyplot as plt
import matplotlib
import sys
from patsy import dmatrices
import sklearn.preprocessing as pp
matplotlib.style.use('ggplot')

from MLmodel import MLmodel

# Change directory on Hengde's laptop
#os.chdir('C:\\Users\\dingh\\Desktop\\data')

grade_map = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7}
status_map = {'Current':1, 'Fully Paid':2, 'In Grace Period':3, 'Charged Off':4, 
                  'Late (31-120 days)':5, 'Late (16-30 days)':6,'Default':7 }

def grade_scatplot (data, feature, expdir):
    data.plot(kind = 'scatter', x = feature, y = 'grade')
    plt.savefig(expdir+ '/' + feature + '_grade.png')
    return


def merge_data(files):
    data_merge = []
    for datafile in files:
        data_year = pd.read_csv(datafile)
        data_merge.append(data_year)

    data = pd.concat(data_merge, ignore_index = True)
    return data


def get_default_rate(data):
    num_default = len(data[data.loan_status == status_map['Default']])
    return float(num_default) / len(data)


def to_dummy(data, column):
    '''Prepare dummy variables for one column, and handles multi-colinearity by removing one of the dummies'''
    data_temp = pd.get_dummies(data[column])
    data = pd.concat([data, data_temp.iloc[:,:-1]],1)
    data = data.drop(column,1)
    return data


def test_for_nan(data):
    temp = []
    for elem in data.columns:
        if True in set(pd.isnull(data[elem])):
            temp.append(elem)
    print "The columns containing nan are ", temp
    return temp
    

def imputation_average(data):
    columns_with_nan = data.columns[pd.isnull(data).any()].tolist()
    for column in columns_with_nan:
        temp_avg = np.mean(data[column].dropna())
        data[column].fillna(temp_avg)
        
    return data
    
    
def add_polynomial_features(data, power):
    if power <= 0:
        return data
    poly = pp.PolynomialFeatures(power)
    data_remain = data.iloc[:,:3]
    temp = data.iloc[:,3:]
    temp = pd.DataFrame(poly.fit_transform(temp))
    temp = temp.drop(temp.columns[0],1)
    data = pd.concat([data_remain,temp],1)
    return data
    

def handle_verification_status(data):
    data_temp = pd.get_dummies(data['verification_status_joint'])
    data = pd.concat([data,data_temp],1)
    o_names = ["Not Verified", "Source Verified", "Verified"]
    s_names = ['Vefirication_Joint_Not','Verification_Joint_SVerified', 'Verification_Joint_Verified']
    for num in range(0,3):
       data.rename(columns = {o_names[num]:s_names[num]}, inplace = True)
    data = data.drop('verification_status_joint',1)
    return data


def reorder_data(data, reorder_list):
    column_list = data.columns.tolist()
    column_list = ist(set(column_list) - set(reorder_list))
    data = data[reorder+column_list]
    
    return data


def preprocessing(data):
    # Converting categorical grading to numerical values
    data = data.replace({'grade': grade_map, 'loan_status':status_map})
    # substitute '< 1 years' to '1', '2 years' to '2', '10+ years' to '10'
    data['emp_length'] = pd.to_numeric(data.emp_length.str.replace(r'<? ?(\d+)[+]? year[s]?', r'\1').replace('n/a', np.NaN))

    data['term'] = pd.to_numeric(data.term.str.replace(' months', ''))

    # Deleting % in int_rate
    data['int_rate'] = pd.to_numeric(data.int_rate.str.replace('%', ''))
    
    # Change home_ownership to dummy variables   note: multicollinearity
    data = to_dummy(data, 'home_ownership')
    
    # Change verification_status to dummy variables
    data = to_dummy(data, 'verification_status')
    
    # Change pymnt_plan (default: n = 0, y = 1)
    data = data.replace({'pymnt_plan':{'n': 0 , 'y': 1}})
    
    # Change purpose to multiple dummy variables
    data = to_dummy(data, 'purpose')
    
    # Delete the % in revol_util
    data['revol_util'] = pd.to_numeric(data.revol_util.str.replace('%',''))
    
    # Convert initial_list_status to multiple dummy variables (multicollinearity not controlled)
    data = to_dummy(data, 'initial_list_status')
    
    # Convert application_type to multiple dummy variables (multicollinearity not controlled)
    # Please check: we only have one application_type: INDIVIDUAL
    # data = to_dummy(data, 'application_type')
    
    # Convert verification_status_joint to multiple dummy variables 
    # (multicollinearity automatically avoided since get_dummies does not account for NaN
    # I suggest remove this feature because it's added very late -- only a small amount of data has it
    #data = handle_verification_status(data)
    
    # Replace nan with zero in specific columns (fill na with )
    columns_nantozero = ['mths_since_last_delinq', 'mths_since_last_record','collections_12_mths_ex_med',
                         'annual_inc_joint', 'dti_joint']
    for elem in columns_nantozero:
        if elem in data.columns:
            data[elem] = data[elem].fillna(0)
         
    return data

    
def deal_nan(data, imputation = None):
    if not imputation:
        data = data.dropna()
    
    elif imputation == 'average':
        data = imputation_average(data)
    
    else:
        raise RuntimeError('Imputation: this part has not been finished yet.')
    
    return data

    
def split_data(data):
    # Shuffle the data set to get training, development and test sets.
    data = data.reindex(np.random.permutation(data.index))
    
    num_data = len(data)
    batch = int(num_data / 5)

    data_train = data[:3*batch]
    data_dev = data[3*batch:4*batch]
    data_test = data[4*batch:]

    data_train.reset_index(drop=True, inplace=True)
    data_dev.reset_index(drop=True, inplace=True)
    data_test.reset_index(drop=True, inplace=True)

    return data_train, data_dev, data_test


def prepare_data():
    #files = [ "LoanStats3a.csv", "LoanStats3b.csv", "LoanStats3c.csv", "LoanStats3d.csv", "LoanStats_2016Q1.csv", "LoanStats_2016Q2.csv", "LoanStats_2016Q3.csv" ]
    files = [ "LoanStats3b.csv" ]
    data = merge_data(files)
    data_train, data_dev, data_test = split_data(data)
    # Save to csv
    data_train.to_csv("data_train.csv")
    data_dev.to_csv("data_dev.csv")
    data_test.to_csv("data_test.csv")


def read_data_file(data_file):
    date_columns = ['issue_d', 'last_pymnt_d', 'next_pymnt_d', 'last_credit_pull_d', 'earliest_cr_line']
    data = pd.read_csv(data_file, index_col = 0, parse_dates = date_columns)
    return data


def get_feat_target(data, feature_list, target):
    print "Feature list is %s" % feature_list
    print "Target is %s" % target
    x = data[feature_list]
    y = data[target]
    return x, y


def eda(data_file, param_file, exp_dir):
    '''Exploratory Data Analysis'''
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    params = json.load(open(param_file, 'r'))
    data = read_data_file(data_file)
    data = preprocessing(data)

    default_rate = get_default_rate(data)
    print "Default rate is %f" % default_rate

    for i in params['feature_list']:
        grade_scatplot(data, i, exp_dir)


def train(data_file, param_file, exp_dir):
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    params = json.load(open(param_file, 'r'))
    # save params file for record
    json.dump(params, open(exp_dir+'/params', 'w'))

    data = read_data_file(data_file)
    data = preprocessing(data)
    data = data[params['feature_list']+[params['target']]]

    data = deal_nan(data)
    
    if 'polynomial_features' in params:
        data = add_polynomial_features(data, params['polynomial_features'])
    
    x_train, y_train = get_feat_target(data, params['feature_list'], params['target'])
        
    print "\n\nPerforming %s" % params['model_type']

    model = MLmodel(params['model_type'], params['model_params'])
    model.train(x_train, y_train)
    model.save(exp_dir+'/model')


def test(data_file, exp_dir):
    param_file = exp_dir+'/params'
    if not os.path.exists(param_file):
        raise RuntimeError('Cannot find param file at %s' % param_file)
    params = json.load(open(exp_dir+'/params', 'r'))

    data = read_data_file(data_file)
    data = preprocessing(data)
    
    x_test, y_test = get_feat_target(data, params['feature_list'], params['target'])
 
    model = MLmodel(params['model_type'], params['model_params'])

    model.test(x_test, y_test)
