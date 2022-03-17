import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import logger


class Engineering:
    def __init__(self):
        self.log_writer = logger.App_Logger()
        self.file_object = open("./logs/featengineering.txt", 'a+')
        #self.dataset=pd.read_csv('./data/train.csv')
    def feature_engineering(self):
            self.data = pd.read_csv('./data/train.csv')
            self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.data,self.data['SalePrice'],test_size=0.1,random_state=0)
            #print(self.data.head())
            self.log_writer.log(self.file_object, "Entered the cat_missing_values function.")
            self.features_nan=[feature for feature in self.data.columns if self.data[feature].isnull().sum()>1 and self.data[feature].dtypes=='O'] 
            self.data[self.features_nan]=self.data[self.features_nan].fillna('Missing')
            # Dealing with Numerical missing values.
            self.log_writer.log(self.file_object, "Dealing with missing values in numerical features.")
            self.numerical_with_nan=[feature for feature in self.data.columns if self.data[feature].isnull().sum()>1 and self.data[feature].dtypes!='O']
            self.log_writer.log(self.file_object, "Entering for loop.")
            for feature in self.numerical_with_nan:
                ## We will replace by using median since there are outliers
                median_value=self.data[feature].median()
                self.data[feature+'nan']=np.where(self.data[feature].isnull(),1,0)
                self.data = self.data[feature].fillna(median_value,inplace=True) 
            # Handling a temporal feature
            self.log_writer.log(self.file_object, "Dealing with temporal features.")
            for feature in ['YearBuilt','YearRemodAdd','GarageYrBlt']:
                self.data[feature]=self.data['YrSold']-self.data[feature]
                
            # Doing log normal transformation
            self.log_writer.log(self.file_object, "Performing log normal transformation.")
            # As we have seen from our EDA some numerical variable are skewed so we have to convert it into SND.
            num_features=['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea', 'SalePrice']
            for feature in num_features:
                self.data[feature]=np.log(self.data[feature])
                
            # Handling rare values
            self.log_writer.log(self.file_object, "Handling rare values.")
            # We will remove categorical variables that are present less than 1% of the observations
            self.categorical_features=[feature for feature in self.data.columns if self.data[feature].dtype=='O']
            
            for feature in self.categorical_features:
                temp=self.data.groupby(feature)['SalePrice'].count()/len(self.data)
                temp_df=temp[temp>0.01].index
                self.data[feature]=np.where(self.data[feature].isin(temp_df),self.data[feature],'Rare_var')
                
            for feature in self.categorical_features:
                labels_ordered=self.data.groupby([feature])['SalePrice'].mean().sort_values().index
                labels_ordered={k:i for i,k in enumerate(labels_ordered,0)}
                self.data[feature]=self.data[feature].map(labels_ordered)
                
            # Doing scaling operation'
            self.log_writer.log(self.file_object, "EDoing scaling operation.")
            feature_scale=[feature for feature in self.data.columns if feature not in ['Id','SalePrice']]
            scaler=MinMaxScaler()
            scaler.fit(self.data[feature_scale])
            scaler.transform(self.data[feature_scale])
            
            # transform the train and test set, and add on the Id and SalePrice variables
            self.data_refined = pd.concat([self.data[['Id', 'SalePrice']].reset_index(drop=True),
                                pd.DataFrame(scaler.transform(self.data[feature_scale]), columns=feature_scale)],
                                axis=1)
            self.data.to_csv('X_train.csv',index=False)
        
if __name__ == "__main__":
    Engineering().feature_engineering()
        