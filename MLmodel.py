import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, confusion_matrix
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

class MLmodel(object):
    def __init(self, model_type, model_params):
        self.model_type = model_type
        self.model_params = model_params
        self.model = None

    def train(self, x, y):
        if self.model_type == 'linear regression':
            self.model = lin_reg(y_train,x_train)
        
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

        if model_type == 'linear regression':
            predicted_value = model.predict(test_dataset)
            predicted_difference = endog - predicted_value
            MAE = np.mean(np.absolute(predicted_difference))
            print "The mean absolute error is ", MAE
            print "This is ", (MAE/(np.mean(endog)))*100, " percent of the average value"

        return

    def save(self, model_file):
        if self.model_type == 'linear regression':
            self.model.save(model_file)

    def read(self, model_file):
        if self.model_type == 'linear regression':
            self.model = OLSResults.load(model_file)


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
