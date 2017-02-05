import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, confusion_matrix
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle


class MLmodel(object):
    def __init(self, model_type, model_params):
        self.model_type = model_type
        self.model_params = model_params
        self.model = None

    def train(self, x, y):
        if self.model_type == 'linear regression':
            model = sm.OLS(exog = x, endog = y)
            self.model = model.fit()
            
        elif self.model_type == 'logistic_regression':
            logmodel = LogisticRegression(c = self.model_param['alpha'])
            self.model = logmodel.fit(x,y)
            

        elif self.model_type == 'ridge regression':
            clf = Ridge(alpha = self.model_params['alpha']) # Feedback outcome at optimal alpha?
            clf.fit(x,y)
            self.model = clf

    def test(self, x, y):
        if not self.model:
            raise RuntimeError('No model loaded. Please read model before testing')

        if self.model_type == 'linear regression':
            predicted_value = self.model.predict(x)
            predicted_difference = y - predicted_value
            MAE = np.mean(np.absolute(predicted_difference))
            print "The mean absolute error is ", MAE
            print "This is ", (MAE/(np.mean(y)))*100, " percent of the average value"
            

        if self.model_type == 'ridge regression':
            predicted_value = self.model.predict(x)
            predicted_difference = y - x # Not used
            print "The ridge regression model's r squared is ", self.model.score(x,y)
            
        if self.model_type == 'logistic regression':
            accuracy = self.model.score(x,y)
            print "The logistic regresision model's accuracy is ", accuracy
        
            
        return

    def save(self, model_file):
        modelout = open(model_file, "wb")
        pickle.dump(self.model, modelout)
        
        
    def read(self, model_file):
        self.model = pickle.load(model_file, "rb")


'''    elif params['model_type'] == 'decision_tree']:
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(x_train, y_train)
        accuracy_tree = clf.score(x_dev, y_dev)
        print "The MSE is ", mean_squared_error(clf.predict(x_dev),y_dev), "and the accuracy is ", accuracy_tree
        print confusion_matrix(y_dev, clf.predict(x_dev))

    elif params['model_type'] == switch['random_forest']:
        clf = RandomForestClassifier(n_estimators = 25)
        clf = clf.fit(x_train,y_train)    
        accuracy_forest = clf.score(x_dev,y_dev)        
        print "The MSE is ", mean_squared_error(clf.predict(x_dev),y_dev), "and the accuracy is ", accuracy_forest
        print confusion_matrix(y_dev, clf.predict(x_dev))
    
 '''
