"""
I want do a project on MLR using oops concepts
"""
from _pyrepl import __main__

import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error,r2_score
import sys
import warnings
warnings.filterwarnings("ignore")
import pickle
import sys



class MLR:
    def __init__(self, path):
        try:
                self.path = path
                self.df = pd.read_csv(path)
               # print(self.df)
                t={}
                c=0
                for i in self.df['city']:
                    if i not in t:
                        t[i]=1
                        c=c+1
                self.df['city']=self.df['city'].map(t).astype(int)
                self.df['country']=self.df['country'].map({"USA":0}).astype(int)
                #print(self.df)
                self.x = self.df.iloc[:, 1:]
                self.y = self.df.iloc[:, 0]
                self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(self.x,self.y,test_size=0.2,random_state=42)
                print(f"Training Data Size : {len(self.x_train)}: {len(self.y_train)}")
                print(f"testing Data Size : {len(self.x_test)}: {len(self.y_test)}")
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f"error in line no:{er_line.tb_lineno}:due to:{er_type}:and reason was:{er_msg}")

    def training(self):
        try:
                self.reg=LinearRegression()
                self.reg.fit(self.x_train,self.y_train)
                self.y_train_prediction=self.reg.predict(self.x_train)
                print(f"Training Accuracy : {r2_score(self.y_train,self.y_train_prediction)}")
                print(f"Training Loss : {root_mean_squared_error(self.y_train, self.y_train_prediction)}")
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f"error in line no:{er_line.tb_lineno}:due to:{er_type}:and reason was:{er_msg}")
    def testing(self):
        try:
                self.y_test_prediction=self.reg.predict(self.x_test)
                print(f"testing Accuracy : {r2_score(self.y_train, self.y_train_prediction)}")
                print(f"testing Loss : {root_mean_squared_error(self.y_train, self.y_train_prediction)}")
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f"error in line no:{er_line.tb_lineno}:due to:{er_type}:and reason was:{er_msg}")
    def check_own_data(self):
        try:

            bedrooms = 3
            bathrooms = 1.50
            sqft_living = 1340
            sqft_lot = 7912
            floors = 1.5
            waterfront = 0
            view = 0
            condition = 3
            sqft_above = 1340
            sqft_basement = 0
            yr_built = 1955
            yr_renovated = 2005
            city = 1
            country = 0
            print(f"test point prediction:{self.reg.predict([[bedrooms,bathrooms,sqft_living,sqft_lot,floors,
            waterfront,view,condition,sqft_above,sqft_basement,yr_built,yr_renovated,city,country]])[0]}")
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f"error in line no:{er_line.tb_lineno}:due to:{er_type}:and reason was:{er_msg}")

    def saving_model(self):
      try:
           with open('MODEL.pkl', 'wb') as f:
                pickle.dump(self.reg, f)

                print("---------------------------------load and check---------------------------------")
           with open("MODEL.pkl", "rb") as t:
                model = pickle.load(t)
                bedrooms = 3
                bathrooms = 1.50
                sqft_living = 1340
                sqft_lot = 7912
                floors = 1.5
                waterfront = 0
                view = 0
                condition = 3
                sqft_above = 1340
                sqft_basement = 0
                yr_built = 1955
                yr_renovated = 2005
                city = 1
                country = 0
                print(f"test point prediction:{model.predict([[bedrooms, bathrooms, sqft_living, sqft_lot, floors,
                waterfront, view, condition, sqft_above, sqft_basement, yr_built, yr_renovated, city, country]])[0]}")
      except Exception as e:
          er_type, er_msg, er_line = sys.exc_info()
          print(f"error in line no:{er_line.tb_lineno}:due to:{er_type}:and reason was:{er_msg}")

if __name__ == "__main__":
    try:
        path = 'data (1).csv'
        reg = MLR(path)
        reg.training()
        reg.testing()
        reg.check_own_data()
        reg.saving_model()
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        print(f"error in line no:{er_line.tb_lineno}:due to:{er_type}:and reason was:{er_msg}")


