from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xg
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import logger
from featselection import Selection
import pandas as pd
import pickle

class Model:
    def __init__(self):
        self.log_writer = logger.App_Logger()
        self.file_object = open("./logs/model.txt", 'a+')
        
    def train(self):
        dataset = pd.read_csv("./data/X_train.csv")
        self.X_train = Selection().select()
        ## Capture the dependent feature
        self.Y_train=dataset[['SalePrice']]
        # Performing train_test_split
        x_train, x_test, y_train, y_test = train_test_split(self.X_train, self.Y_train, test_size=0.3, random_state=0)
        dict = {"linreg": LinearRegression(), "svr": SVR(), "knn": KNeighborsRegressor(), "tree": DecisionTreeRegressor(),
                "random": RandomForestRegressor(), "xgboost": xg.XGBRegressor(), "ada": AdaBoostRegressor(), "grad": GradientBoostingRegressor()}
        self.log_writer.log(self.file_object, "Fitting a Model.")
        for mod in dict.values():
            model = mod
            model.fit(x_train,y_train)
            model.predict(x_test)
            print("Model {} give {} score.".format(model,model.score(x_test, y_test)))
            
if __name__ == "__main__":
    Model().train()
    # Model RandomForestRegressor(max_leaf_nodes=100, n_estimators=500) give 0.8856559526357313 score.
    # Model GradientBoostingRegressor(learning_rate=0.1098891866898283, max_leaf_nodes=20, n_estimators=200) give 0.8919182144066403 score.