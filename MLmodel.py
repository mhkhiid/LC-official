import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, confusion_matrix
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

class MLmodel(object):
    def __init__(self, model_type, model_params):
        self.model_type = model_type
        self.model_params = model_params
        self.model = None

    def train(self, x, y):
        if self.model_type == 'linear regression':
            self.model = sm.OLS(exog = x,endog = y).fit()
            print self.model.summary()
        
        elif self.model_type == 'logistic_regression':
            self.model = logistic_reg_predict_10(y_train, 
                                x_train, 
                                y_dev, 
                                x_dev, 
                                switch['alpha_start_log'], 
                                switch['alpha_end_log'])

        elif self.model_type == 'ridge regression':
            clf = Ridge(alpha = self.model_params['alpha'])

    def test(self, x, y):
        if not self.model:
            raise RuntimeError('No model loaded. Please read model before testing')

        if self.model_type == 'linear regression':
            y_pred = self.model.predict(x)
            y_diff = y - y_pred
            MAE = np.mean(np.absolute(y_diff))
            print "MAE is %f, %f %% of avg value" % (MAE, MAE/(np.mean(y))*100)
            MSE = np.mean(y_diff ** 2)
            print "MSE is %f, %f %% of avg value" % (MSE, MSE/(np.mean(y**2))*100)

        return

    def save(self, model_file):
        if self.model_type == 'linear regression':
            self.model.save(model_file)

    def read(self, model_file):
        if self.model_type == 'linear regression':
            self.model = sm.regression.linear_model.OLSResults.load(model_file)


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
