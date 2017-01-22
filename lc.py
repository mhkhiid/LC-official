
import re
import pandas as pd
import numpy as np
import random as rand
import matplotlib.pyplot as plt
import matplotlib
import statsmodels.api as sm
import statsmodels.formula.api as smf
import sys
from patsy import dmatrices
import os
import sklearn.preprocessing as pp
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, confusion_matrix
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier




matplotlib.style.use('ggplot')

# Change directory on Hengde's laptop
os.chdir('C:\\Users\\dingh\\Desktop\\data')


switch = {
          'imputation': 0,
          'imputation_average': 1,
          'polynomial_features': 0,
          'y_grade': 1,
          'y_status': 0,
          'scatter_plot': 0,
          'linear_regression': 1, 
          'ridge_regression': 1, 'alpha_start_ridge':0.001, 'alpha_end_ridge':1000,
          'logistic_regression': 0, 'alpha_start_log':0.001, 'alpha_end_ridge':1000,
          'decision_tree': 1,
          'random_forest': 1,
          }
          
          
          
         


def grade_scatplot (data, feature):
    data.plot(kind = 'scatter', x = feature, y = 'grade')
    plt.savefig(feature + '_grade.png')
    return


def lin_reg(y,x):
    mod = sm.OLS(exog = x,endog = y)
    res = mod.fit()
    print res.summary()
    return res
    

def merge_data(files):

    data_merge = []
    for datafile in files:
        data_year = pd.read_csv(datafile)
        data_merge.append(data_year)

    data = pd.concat(data_merge, ignore_index = True)
    data.to_csv('merged_data.csv')
    return data

def get_default_rate(data):
    num_default = len(data[data.loan_status == 4])
    return float(num_default) / len(data)

def to_dummy(data, column):
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
    
def predict_linear_model(model,endog, test_dataset):
    predicted_value = model.predict(test_dataset)
    predicted_difference = endog - predicted_value
    MAE = np.mean(np.absolute(predicted_difference))
    print "The mean absolute error is ", MAE
    print "This is ", (MAE/(np.mean(endog)))*100, " percent of the average value"
    # print "The r-squared of the prediction is ", model.predict(test_dataset).rsquared
    return
    
def ridge_reg(y, x, alpha_input):
    clf = Ridge(alpha = alpha_input)
    clf.fit(x,y)
    print "The r-squared value when alpha is ",alpha_input, " is ", clf.score(x, y)
    return clf
    
def ridge_reg_predict(train_y, train_x, test_y, test_x, alpha_start,alpha_end, alpha_step):
    optimal_alpha = 0
    max_r2 = 0
    for num in np.arange(alpha_start, alpha_end, alpha_step):
        clf = Ridge(alpha = num)
        clf.fit(train_x, train_y)
        r2 = clf.score(test_x,test_y)
        print "The r-squared for prediction when alpha is ", num, "is", r2
        if r2 > max_r2:
            max_r2 = r2
            optimal_alpha = num
            
    print "The optimal r-squared is ", max_r2, " at alpha level of ", optimal_alpha

def ridge_reg_predict_10(train_y, train_x, test_y, test_x, alpha_start,alpha_end):
    optimal_alpha = 0
    max_r2 = 0
    while alpha_start < alpha_end:
        clf = Ridge(alpha_start)
        clf.fit(train_x, train_y)
        r2 = clf.score(test_x,test_y)
        print "The r-squared for prediction when alpha is ", alpha_start, "is", r2
        if r2 > max_r2:
            max_r2 = r2
            optimal_alpha = alpha_start
        alpha_start = 10* alpha_start
    print "The optimal r-squared is ", max_r2, " at alpha level of ", optimal_alpha
    

def logistic_reg_predict_10(train_y, train_x, test_y, test_x, c_start, c_end):
    optimal_c = 0
    max_accuracy = 0
    while c_start < c_end:
        lr= LogisticRegression(C = c_start)
        res = lr.fit(train_x, train_y)
        accuracy = res.score(test_x,test_y)
        mse = mean_squared_error(test_y, res.predict(test_x))
        print "The prediction accuracy for prediction when c level is ", c_start, "is", accuracy, "mse =", mse
        if  accuracy > max_accuracy:
            max_accuracy = accuracy
            optimal_c = c_start
        c_start = 10* c_start
    print "The optimal prediction accuracy is ", max_accuracy, " at c level of ", optimal_c
    
    
def imputation_average(data):
    columns_with_nan = data.columns[pd.isnull(data).any()].tolist()
    for column in columns_with_nan:
        temp_avg = np.mean(data[column].dropna())
        data[column].fillna(temp_avg)
        
    return data
    
    
def add_polynomial_features(data, power):
    poly = pp.PolynomialFeatures(power)
    data_remain = data.iloc[:,:3]
    temp = data.iloc[:,3:]
    temp = pd.DataFrame(poly.fit_transform(temp))
    temp = temp.drop(temp.columns[0],1)
    data = pd.concat([data_remain,temp],1)
    return data
    

    
def preprocessing(data):
    # Converting categorical grading to numerical values
    grade_map = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7}
    status_map = {'Current':1, 'Fully Paid':2, 'In Grace Period':3, 'Charged Off':4, 
                  'Late (31-120 days)':5, 'Late (16-30 days)':6,'Default':7 }
    data = data.replace({'grade': grade_map, 'loan_status':status_map})
    data = data.drop('sub_grade',1)
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
    data = to_dummy(data, 'application_type')
    
    
    # Convert verification_status_joint to multiple dummy variables (multicollinearity automatically avoided since get_dummies does not account for NaN
    data_temp = pd.get_dummies(data['verification_status_joint'])
    data = pd.concat([data,data_temp],1)
    o_names = ["Not Verified", "Source Verified", "Verified"]
    s_names = ['Vefirication_Joint_Not','Verification_Joint_SVerified', 'Verification_Joint_Verified']
    for num in range(0,3):
       data.rename(columns = {o_names[num]:s_names[num]}, inplace = True)
    data = data.drop('verification_status_joint',1)
       
    # Drop insignificant columns
    columns = ['emp_title', 'issue_d',
                'title', 'zip_code', 'addr_state', 'earliest_cr_line', 'last_pymnt_d',
                'next_pymnt_d', 'last_credit_pull_d']
    for elem in columns:
        data = data.drop(elem, 1)
        
        
    
    # Replace nan with zero in specific columns (fill na with )
    columns_nantozero = ['mths_since_last_delinq', 'mths_since_last_record','collections_12_mths_ex_med',
                         'annual_inc_joint', 'dti_joint']
    for elem in columns_nantozero:
        data[elem] = data[elem].fillna(0)
        
        
    # Drop 'Unnamed: 0', 'id', 'member_id'
    temp = ['Unnamed: 0', 'id', 'member_id']
    data = data.drop(temp,1)
        
    
    # Rearrange the order for convinience
    column_list = data.columns.tolist()
    reorder = ['grade','loan_status']
    for elem in reorder:
        column_list.remove(elem)
    
    data = data[reorder+column_list]
        
    
    return data

    
    

def deal_nan(data, switch):
    if switch['imputation'] == 0:
        
        # Drop the later-added features
    
        temp = ['chargeoff_within_12_mths', 'delinq_amnt', 'pub_rec_bankruptcies', 'tax_liens']
        for elem in temp:
            data[elem + "_new"] = data[elem]
        start = data.columns.tolist().index('tot_coll_amt')
        end = data.columns.tolist().index('total_il_high_credit_limit')
        drop_list = data.columns.tolist()[start:end]
        data = data.drop(drop_list, 1)
            
        # Drop nan
        data = data.dropna()
    
    else:
        if switch['imputation_average'] == 1:
            data = imputation_average(data)
            
        if switch['imputation_average'] != 1:
            print "This part has not been finished yet."
    
    return data
            
    
    
    
def split_data(data):
    # Shuffle the data set to get training, development and test sets.
    data = data.reindex(np.random.permutation(data.index))
    data.reset_index(drop=True, inplace=True)
    data = data.drop('Unnamed: 0',1)
    
    num_data = len(data)
    batch = int(num_data / 5)

    data_test = data[0:batch]
    data_dev = data[batch:2*batch]
    data_train = data[2*batch:]
    data_dev.reset_index(drop=True, inplace=True)
    data_train.reset_index(drop=True, inplace=True)

    # Save to csv
    data_test.to_csv("data_test.csv")
    data_dev.to_csv("data_dev.csv")
    data_train.to_csv("data_train.csv")
    data.to_csv("overall sorted data.csv")
    return

    
def run(switch):
    
    #files = [ "LoanStats3a.csv", "LoanStats3b.csv", "LoanStats3c.csv", "LoanStats3d.csv", "LoanStats_2016Q1.csv", "LoanStats_2016Q2.csv", "LoanStats_2016Q3.csv" ]
    #files = [ "LoanStats3a.csv" ]
    #data = merge_data(files)
    #data = preprocessing(data)
    #split_data(data)

    data_train = pd.read_csv("data_train.csv")
    data_dev = pd.read_csv('data_dev.csv')
    # data_test = pd.read_csv('data_test.csv')
    data_train = data_train.drop('Unnamed: 0',1)
    data_dev = data_dev.drop('Unnamed: 0' ,1)
    
    # Create back-up untouched data for other manipulation to avoid reading again
    data_train_backup = data_train
    data_dev_backup = data_dev
    # data_test_backup = data_test
    
    # Reload data for diffrent manipulation
    # data_train = data_train_backup
    # data_dev = data_dev_backup
    # data_test = data_test_backup
    
    
    ''' Want to do something like below but obviously cannot do it this way
    data_with_nan = [data_train, data_dev]
    for elem in data_with_nan:
        elem = deal_nan(elem)
    '''
    
    data_train = deal_nan(data_train,switch)
    data_dev = deal_nan(data_dev,switch)
    #data_test = deal_nan(data_test)
    
    default_rate = get_default_rate(data_train)
    print "Default rate is %f" % default_rate
    
    if switch['polynomial_features'] > 0:
        data_train = add_polynomial_features(data_train,switch['polynomial_features'])
        data_dev = add_polynomial_features(data_dev,switch['polynomial_features'])
        # data_test = add_polynomial_features(data_test,switch['polynomial_features])
        
    
    x_train = data_train.iloc[:,3:]
    x_dev = data_dev.iloc[:,3:]
    
    if switch['y_grade'] == 1:
        y_train = data_train['grade']
        y_dev = data_dev['grade']
        if switch['y_status'] == 1:
            sys.exit("Can only set one target featuer at each time.")
    elif switch['y_status']:
        y_train = data_train['loan_status']
        y_dev = data_dev['loan_status']
    else:
        sys.exit("The other target features are not available. Please modify your switch.")

    
    
    
    if switch['scatter_plot'] == 1:
        print "The scatter plots are: "
        # Exploratory data analysis
        features = [ 'avg_cur_bal', 'chargeoff_within_12_mths', 'dti', 'num_bc_sats', 
                    'pub_rec', 'delinq_2yrs', 'emp_length', 'il_util', 'inq_fi', 
                    'inq_last_12m', 'max_bal_bc', 'mths_since_last_delinq', 
                    'mths_since_last_major_derog', 'num_accts_ever_120_pd', 
                    'num_actv_bc_tl', 'num_bc_sats', 'num_tl_120dpd_2m', 
                    'num_tl_30dpd', 'open_il_6m', 'pct_tl_nvr_dlq', 'pub_rec', 
                    'tot_cur_bal', 'verification_status' ]
                    
        for i in features:
            grade_scatplot(data_train, i)

    
    if switch['linear_regression'] == 1:
        print"\n\n\nThe linear regression performance is "                            
        # Linear regression of grade with regard to all variables
        model = lin_reg(y_train,x_train)
        
        # Predict using dev set
        predict_linear_model(model,y_dev, x_dev)
    
        
    if switch['ridge_regression'] == 1:
        print "\n\n\nThe ridge regression performance is "
        # Ridge regression exploration
        ridge_reg_predict_10(train_y = y_train, train_x = x_train,test_y=y_dev, test_x=x_dev,alpha_start = switch['alpha_start_ridge'], 
                          alpha_end = switch['alpha_end_ridge'])
    
    
   
    if switch['logistic_regression'] == 1:
        
        # Default status prediction using Logistic Regression
        logistic_reg_predict_10(y_train, x_train, y_dev, x_dev, switch['alpha_start_log'], switch['alpha_end_log'])

        
    if switch['decision_tree'] == 1:
        print " \n\n\n The decision tree performance is "
        # Decision Tree algorithm
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(x_train, y_train)
        accuracy_tree = clf.score(x_dev, y_dev)
        print "The MSE is ", mean_squared_error(clf.predict(x_dev),y_dev), "and the accuracy is ", accuracy_tree
        print confusion_matrix(y_dev, clf.predict(x_dev))

    if switch['random_forest'] == 1:
        
        # Random Forest Exploration
        print "\n\n\n The random forest performance is "
        clf = RandomForestClassifier(n_estimators = 25)
        clf = clf.fit(x_train,y_train)    
        accuracy_forest = clf.score(x_dev,y_dev)        
        print "The MSE is ", mean_squared_error(clf.predict(x_dev),y_dev), "and the accuracy is ", accuracy_forest
        print confusion_matrix(y_dev, clf.predict(x_dev))
    
    
'''
Findings:
 
 inq_fi                          No obvious relationship
 inq_last_12m                    No obvious relationship
 max_bal_bc                      quite obvious positive relationship
 mths_since_last_delinq 
 mths_since_last_major_derog     Not so obvious
 num_accts_ever_120_pd           Obvious positive relationship???
 num_actv_bc_tl                  No obvious relationship
 num_bc_sats                     good relationship
 num_tl_120dpd_2m                good negative relationship, but sample space is too small
 num_tl_30dpd                    No relationship 
 open_il_6m                      positive relationship 
 pct_tl_nvr_dlq                  WHY MOST OF THE SCATTER PLOTS ARE > SHAPED??
 pub_rec                         counter-intuitive shape. 
 
 
 
Logistic Regresison:
The prediction accuracy for prediction when c level is  0.001 is 0.966742366894 mse = 0.461990074859
The prediction accuracy for prediction when c level is  0.01 is 0.966876945075 mse = 0.460425603499
The prediction accuracy for prediction when c level is  0.1 is 0.966961056439 mse = 0.460661115317
The prediction accuracy for prediction when c level is  1.0 is 0.966119942804 mse = 0.462191942131
The prediction accuracy for prediction when c level is  10.0 is 0.966742366894 mse = 0.46205736395
The prediction accuracy for prediction when c level is  100.0 is 0.966170409622 mse = 0.461569518042
The optimal prediction accuracy is  0.966961056439  at c level of  0.1
 


Decision Tree:
The MSE is  0.914912944739 and the accuracy is  0.930944570611

confusion_matrix(y_grade_dev, clf.predict(test_x))
Out[105]: 
array([[36295,     7,   702, ...,  1249,   290,     6],
       [    8, 15029,     0, ...,     0,     0,     0],
       [  460,     0,    14, ...,    34,     6,     1],
       ..., 
       [  994,     0,    40, ...,   103,    14,     0],
       [  210,     0,    10, ...,    22,     4,     0],
       [   12,     0,     3, ...,     1,     0,     0]])



Random Forest with 25 trees:
The MSE is  0.457986373959 and the accuracy is  0.967045167802

confusion_matrix(y_grade_dev, clf.predict(test_x))
Out[121]: 
array([[38537,     7,     0, ...,     5,     0,     0],
       [    0, 15058,     0, ...,     0,     0,     0],
       [  512,     0,     0, ...,     3,     0,     0],
       ..., 
       [ 1142,     0,     0, ...,     9,     0,     0],
       [  246,     0,     0, ...,     0,     0,     0],
       [   15,     0,     0, ...,     1,     0,     0]])    
 
 
 
 
'''
"""
if __name__ == '__main__':
    run()
"""     

     
     
     
     
     
     
     
     
     
