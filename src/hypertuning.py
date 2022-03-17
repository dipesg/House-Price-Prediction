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
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform
import logger
from featselection import Selection
import pandas as pd
import pickle

class Tune:
    def __init__(self):
        self.log_writer = logger.App_Logger()
        self.file_object = open("./logs/tune.txt", 'a+')
        dataset = pd.read_csv("./data/X_train.csv")
        self.X_train = Selection().select()
        ## Capture the dependent feature
        self.Y_train=dataset[['SalePrice']]
        # Performing train_test_split
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X_train, self.Y_train, test_size=0.3, random_state=0)
        
    def hypertune_randomforest(self):
        self.log_writer.log(self.file_object, "Tuning Random Forest Model.")
        param_distributions = {"n_estimators": [1, 2, 5, 10, 20, 50, 100, 200, 500],"max_leaf_nodes": [2, 5, 10, 20, 50, 100]}
        search1_cv = RandomizedSearchCV(
            RandomForestRegressor(n_jobs=2), param_distributions=param_distributions,
            scoring="accuracy", n_iter=10, random_state=0, n_jobs=2,
        )
        search1_cv.fit(self.x_train, self.y_train)
        #print(f"Best score: {search_cv.score(x_test,y_test)}")

        columns = [f"param_{name}" for name in param_distributions.keys()]
        columns += ["mean_test_error", "std_test_error"]
        cv_results = pd.DataFrame(search1_cv.cv_results_)
        cv_results["mean_test_error"] = -cv_results["mean_test_score"]
        cv_results["std_test_error"] = cv_results["std_test_score"]
        cv_results[columns].sort_values(by="mean_test_error")
        
        # Findinding best tuned parameter
        search1_cv.best_params_
        
        # Fitting RandomForest model Using best parameter
        model = RandomForestRegressor(n_estimators=500, max_leaf_nodes=100)
        model.fit(self.x_train,self.y_train)
        pred = model.predict(self.x_test)
        print("Score is {}".format(mean_absolute_error(self.y_test, pred)))
        print("Model {} give {} score.".format(model,model.score(self.x_test, self.y_test)))
        
    def hypertune_grad(self):
        self.log_writer.log(self.file_object, "Tuning GradientBoosting Model.")
        param_distributions = {"n_estimators": [1, 2, 5, 10, 20, 50, 100, 200, 500],"max_leaf_nodes": [2, 5, 10, 20, 50, 100],"learning_rate": loguniform(0.01, 1),}
        search_cv = RandomizedSearchCV(GradientBoostingRegressor(), param_distributions=param_distributions,scoring="neg_mean_absolute_error", n_iter=20, random_state=0, n_jobs=2)
        search_cv.fit(self.x_train, self.y_train)

        columns = [f"param_{name}" for name in param_distributions.keys()]
        columns += ["mean_test_error", "std_test_error"]
        cv_results = pd.DataFrame(search_cv.cv_results_)
        cv_results["mean_test_error"] = -cv_results["mean_test_score"]
        cv_results["std_test_error"] = cv_results["std_test_score"]
        cv_results[columns].sort_values(by="mean_test_error")
        
        #selectinig a best tuned parameter
        search_cv.best_params_
        
        model = GradientBoostingRegressor(learning_rate=0.1098891866898283, max_leaf_nodes=20, n_estimators=200)
        model.fit(self.x_train, self.y_train)
        pickle.dump(model, open('model.pkl', 'wb'))
        pred = model.predict(self.x_test)
        print("Score is {}".format(mean_absolute_error(self.y_test, pred)))
        print("Model {} give {} score.".format(model,model.score(self.x_test, self.y_test)))
        
if __name__ == "__main__":
    #Tune().hypertune_randomforest()
    Tune().hypertune_grad()

        
        
        
        
        