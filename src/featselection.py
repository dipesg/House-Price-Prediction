import pandas as pd
import numpy as np
import logger
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel

class Selection:
    def __init__(self):
        self.log_writer = logger.App_Logger()
        self.file_object = open("./logs/featureselection.txt", 'a+')
        
    def select(self):
        self.dataset=pd.read_csv('./data/X_train.csv')
        
        ## Capture the dependent feature
        self.Y_train=self.dataset[['SalePrice']]
        
        ## drop dependent feature from dataset
        self.X_train=self.dataset.drop(['Id','SalePrice'],axis=1)
        
        ### Apply Feature Selection
        # first, I specify the Lasso Regression model, and I
        # select a suitable alpha (equivalent of penalty).
        # The bigger the alpha the less features that will be selected.

        # Then I use the selectFromModel object from sklearn, which
        # will select the features which coefficients are non-zero

        feature_sel_model = SelectFromModel(Lasso(alpha=0.0005, random_state=0)) # remember to set the seed, the random state in this function
        feature_sel_model.fit(self.X_train, self.Y_train)
        
        # let's print the number of total and selected features
        # this is how we can make a list of the selected features
        selected_feat = self.X_train.columns[(feature_sel_model.get_support())]

        # let's print some stats
        print('total features: {}'.format((self.X_train.shape[1])))
        print('selected features: {}'.format(len(selected_feat)))
        print('features with coefficients shrank to zero: {}'.format(
            np.sum(feature_sel_model.estimator_.coef_ == 0)))
        
        self.X_train=self.X_train[selected_feat]
        return self.X_train
    
if __name__ == "__main__":
        x_train = Selection().select()
        print(x_train.shape)

    
    