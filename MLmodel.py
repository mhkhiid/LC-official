import os
import pickle
import logging
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix, accuracy_score

class MLmodel(object):
    def __init__(self, model_type, model_params):
        self.model_type = model_type
        self.model_params = model_params
        self.model = None


    def train(self, x, y):
        if self.model_type in ['LogisticRegression', 'RidgeRegression']:
            if 'alpha' not in self.model_params:
                raise RuntimeError('Require alpha be in params file for ridge regression')
        if self.model_type in ['LogisticRegressionCV', 'RidgeRegressionCV']:
            if 'alphas' not in self.model_params:
                raise RuntimeError('Require alpha be in params file for ridge regression')

        if self.model_type == 'LinearRegression':
            self.model = sm.OLS(exog = x,endog = y).fit()
            logging.info(self.model.summary())
        elif self.model_type == 'LogisticRegression':
            self.model = LogisticRegression(C = self.model_params['alpha']).fit(x, y)
        elif self.model_type == 'LogisticRegressionCV':
            self.model = LogisticRegressionCV(Cs = self.model_params['alphas']).fit(x,y)
        elif self.model_type == 'RidgeRegression':
            self.model = Ridge(alpha = self.model_params['alpha']).fit(x, y)
        elif self.model_type == 'RidgeRegressionCV':
            self.model = RidgeCV(alphas = self.model_params['alphas']).fit(x,y)
        elif self.model_type == 'DecisionTree':
            self.model = tree.DecisionTreeClassifier().fit(x,y)
        elif self.model_type == 'RandomForest':
            self.model = RandomForestClassifier(n_estimator = self.model_params['number_of_trees'])
            clf = RandomForestClassifier(n_estimators = 25)
        else:
            raise RuntimeError('Train not supported for %s yet' % self.model_type)


    def test(self, x, y):
        if not self.model:
            raise RuntimeError('No model loaded. Please read model before testing')

        if self.model_type in ['LinearRegression', 'RidgeRegression', 'RidgeRegressionCV']:
            y_pred = self.model.predict(x)
            y_diff = y - y_pred
            mae = np.mean(np.absolute(y_diff))
            logging.info("MAE is %f, %f %% of avg value", mae, mae/(np.mean(y))*100)
            print y_diff
            mse = np.mean(y_diff**2)
            ## accuracy = accuracy_score(y, np.around(y_pred))
            logging.info("MSE is %f, %f %% of avg value", mse, mse/(np.mean(y**2))*100)
            ## logging.info("Prediction accuracy is ", accuracy)
        elif self.model_type in ['LogisticRegression', 'LogisticRegressionCV']:
            accuracy = self.model.score(x, y)
            y_pred = self.model.predict(x)
            mse = mean_squared_error(y, y_pred)
            logging.info("Prediction accuracy is %f, MSE is %f", accuracy, mse)
        else:
            raise RuntimeError('Test not supported for %s yet' % self.model_type)

        return


    def save(self, model_file):
        if not self.model:
            raise RuntimeError('No model to save. Please train/read model before saving')

        if self.model_type == 'linear regression':
            self.model.save(model_file)
        else:
            pickle.dump(self.model, open(model_file, 'w'))


    def read(self, model_file):
        if not os.path.exists(model_file):
            raise RuntimeError('Cannot find model file %s' % model_file)
        if self.model_type == 'linear regression':
            self.model = sm.regression.linear_model.OLSResults.load(model_file)
        else:
            self.model = pickle.load(open(model_file, 'r'))

