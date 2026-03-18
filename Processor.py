import math
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVR
from datetime import datetime as dt
class Processor:

    def __init__(self):

        self.kawhi_leonard = pd.read_csv('data/game-log/kawhi_leonard.csv').dropna(subset=['G'])
        self.trae_young = pd.read_csv('data/game-log/trae_youngBIG.csv').dropna(subset=['G'])
        self.trae_young2 = pd.read_csv('data/game-log/trae_young2.csv').dropna(subset=['G'])
        self.bam_adebayo = pd.read_csv('data/game-log/bam_adebayo.csv').dropna(subset=['G'])
        self.derrick_white = pd.read_csv('data/game-log/derrick_white.csv').dropna(subset=['G'])
        self.jose_alvarado = pd.read_csv('data/game-log/jose_alvarado.csv').dropna(subset=['G'])

        self.kawhi_leonard = self.kawhi_leonard.drop(columns=['Date', 'Age', 'Rk', 'FG', '3P', 'FT'])
        self.trae_young = self.trae_young.drop(columns=['Date', 'Age', 'Rk', 'FG', '3P', 'FT'])
        self.trae_young2 = self.trae_young2.drop(columns=['Date', 'Age', 'Rk', 'FG', '3P', 'FT'])
        self.bam_adebayo = self.bam_adebayo.drop(columns=['Date', 'Age', 'Rk', 'FG', '3P', 'FT'])
        self.derrick_white = self.derrick_white.drop(columns=['Date', 'Age', 'Rk', 'FG', '3P', 'FT'])
        self.jose_alvarado = self.jose_alvarado.drop(columns=['Date', 'Age', 'Rk', 'FG', '3P', 'FT'])
    def random_forest_boosted_points(self, data):
        y_d = data.iloc[:, -3]
        y = pd.Series.to_numpy(y_d)

        data = data.drop(columns = ['Unnamed: 7', 'PTS', 'FGA', '3PA', 'FTA'])
        X = pd.DataFrame.to_numpy(data)

        new_team, new_opp, new_home = self.convert_to_ints(X)

        X[:, 1] = new_team
        X[:, 2] = new_home
        X[:, 3] = new_opp

        # convert minutes to float
        for k in range(y.shape[0]):
            X[k, 5] = self.convert_minutes(X[k, 5])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=19)
        rfr = RandomForestRegressor(random_state=13)
        rfr.fit(X_train, y_train)
        yp = rfr.predict(X_test)
        mean_1 = mean_absolute_error(yp, y_test)
        mean_2 = mean_squared_error(yp, y_test)
        r = r2_score(yp, y_test)

        # print(f"R2 Score: {r}")
        # print(f"Mean Absolute Error: {mean_1}")
        # print(f"Mean Squared Error: {mean_2}")
        # print("predictions: ")
        # print(yp)
        # print("ground truth: ")
        # print(y_test)

        return mean_1, mean_2, r
    def random_forest_boosted_spread(self, data):
        
        y_d = data.iloc[:, 4]
        y = pd.Series.to_numpy(y_d)

        data = data.drop(columns=['Unnamed: 7'])

        X_prime = data.iloc[:, 0:28] #[0: 7] + data.iloc[:, 8:28]
        X = pd.DataFrame.to_numpy(X_prime)

        new_team, new_opp, new_home = self.convert_to_ints(X)

        X[:, 1] = new_team
        X[:, 2] = new_home
        X[:, 3] = new_opp

        for i in range(y.shape[0]):
            y[i] = int(y[i][3:-1])

        # convert minutes to floats [minutes is column 5]
        for k in range(y.shape[0]):
            X[k, 5] = self.convert_minutes(X[k, 5])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=19)
        rfr = RandomForestRegressor(random_state=13)
        rfr.fit(X_train, y_train)
        yp = rfr.predict(X_test)
        mean_1 = mean_absolute_error(yp, y_test)
        mean_2 = mean_squared_error(yp, y_test)
        r = r2_score(yp, y_test)

        # print(f"R2 Score: {r}")
        # print(f"Mean Absolute Error: {mean_1}")
        # print(f"Mean Squared Error: {mean_2}")
        # print("predictions: ")
        # print(yp)
        # print("ground truth: ")
        # print(y_test)
    # TODO: Make teams into numbers (0-30)
    # TODO: Convert Away/Home to numbers (0-1)
    def convert_to_ints(self, data):
        teams = {'ATL':0, 'BOS':1, 'BRK':2, 'CHI':3, 'CHO':4, 'CLE':5, 'DAL':6, 'DEN':7, 'DET':8, 'GSW':9, 'HOU':10, 'IND':11, 'LAC':12, 'LAL':13, 'MEM':14, 'MIA':15, 'MIL':16, 'MIN':17, 'NOP':18, 'NYK':19, 'ORL':20, 'PHI':21, 'PHO':22, 'POR':23, 'SAC':24, 'SAS':25, 'TOR':26, 'UTA':27, 'WAS':28, 'OKC': 29}

        team = data[:, 1]
        home = data[:, 2]
        opp = data[:, 3]

        FG = data[:, 6]
        TP = data[:, 7]
        FT = data[:, 8]
        for i in range(len(FT)):
            if type(FT[i]) != type('str'):
                FT[i] = 0
            if type(TP[i]) != type('str'):
                TP[i] = 0
            if type(FG[i]) != type('str'):
                FG[i] = 0

        # home = np.nan_to_num(home, nan=1)
        home[home == '@'] = 0
        home[home != 0] = 1

        new_team = np.array([teams[x] for x in team])
        new_opp = np.array([teams[x] for x in opp])

        return new_team, new_opp, home
    def convert_minutes(self, x):

        ind = x.index(":")

        minutes = float(x[0:ind])
        seconds = float(x[ind+1:ind+3])
        return minutes + (seconds / 60)
    def convert_date(self, data):
        date = data[:, 1]
        for i, d in enumerate(date):
            ind = d.index("/")
            month = int(d[0:ind])
            day = int(d[ind+1: -5])
            year = int(d[-4:])
            datey = dt(year, month, day)
            date[i] = datey
        one = date[0]
        for i, d in enumerate(date):
            date[i] = (d - one).days
        return date
    def gradient_tree(self, data):
        y_d = data.iloc[:, -3]
        y = pd.Series.to_numpy(y_d)

        data = data.drop(columns = ['Unnamed: 7', 'PTS', 'FGA', '3PA', 'FTA'])
        X = pd.DataFrame.to_numpy(data)

        new_team, new_opp, new_home = self.convert_to_ints(X)

        X[:, 1] = new_team
        X[:, 2] = new_home
        X[:, 3] = new_opp

        # paramgrid = {
        #     'n_estimators':[100, 200, 300],
        #     'learning_rate':[0.01, 0.1, 0.2],
        #     'max_depth':[3, 4, 5],
        #     'min_samples_split':[2, 3, 4],
        #     'min_samples_leaf':[1, 2, 3],
        # }
        # convert minutes to float
        for k in range(y.shape[0]):
            X[k, 5] = self.convert_minutes(X[k, 5])

        # print(X)
        # print(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=19)
        # print(X_train)
        # random_state=13
        GBR = GradientBoostingRegressor()
        GBR.fit(X_train, y_train)
        yp = GBR.predict(X_test)
        # GBRCV = GridSearchCV(GBR, paramgrid, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
        # GBRCV.fit(X_train, y_train)
        # yp = GBRCV.predict(X_test)
        mean_1 = mean_absolute_error(yp, y_test)
        mean_2 = mean_squared_error(yp, y_test)
        r = r2_score(yp, y_test)

        # print(f"R2 Score: {r} \n")
        # print(f"Mean Absolute Error: {mean_1} \n")
        # print(f"Mean Squared Error: {mean_2} \n")
        # print("predictions: ")
        # print(yp)
        # print("ground truth: ")
        # print(y_test)

        return mean_1, mean_2, r
    
    def create_avg(self, X):
        X = X.astype(float)
        avg = X[0]
        for i in range(X.shape[0] -  1):
            avg = avg + X[i + 1]
        avg = avg / 5
        avg[0] = X[-1, 0] + 1
        avg = np.reshape(avg, (1, -1))
        return avg
    def linearSVR(self, data):
        y_d = data.iloc[:, -3]
        y = pd.Series.to_numpy(y_d)

        data = data.drop(columns = ['Unnamed: 7', 'PTS', 'FGA', '3PA', 'FTA'])
        X = pd.DataFrame.to_numpy(data)
        # y_new = self.trae_young2.iloc[:, -3]
        # print(y_new)
        # new = self.trae_young2.drop(columns = ['Unnamed: 7', 'PTS', 'FGA', '3PA', 'FTA'])
        # y_new = pd.Series.to_numpy(y_new)
        # new = pd.DataFrame.to_numpy(new)

        new_team, new_opp, new_home = self.convert_to_ints(X)
        X[:, 1] = new_team
        X[:, 2] = new_home
        X[:, 3] = new_opp

        # new_team, new_opp, new_home = self.convert_to_ints(new)
        # new[:, 1] = new_team
        # new[:, 2] = new_home
        # new[:, 3] = new_opp

        for k in range(y.shape[0]):
            X[k, 5] = self.convert_minutes(X[k, 5])
        # for k in range(y_new.shape[0]):
        #     new[k, 5] = self.convert_minutes(new[k, 5])
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.99)
        X_test = X[-6: -1]
        X_test = np.append(X_test, (self.create_avg(X[-6:-1])))
        X_train, y_train, y_test = X, y, np.array(y[-6:])
        SVR = LinearSVR(dual=True, max_iter=1000000)
        SVR.fit(X_train, y_train)
        yp = SVR.predict(X_test)
        mean_1 = mean_absolute_error(yp, y_test)
        mean_2 = mean_squared_error(yp, y_test)
        r = r2_score(yp, y_test)
        # print(f"R2 Score: {r}")
        # print(f"Mean Absolute Error: {mean_1}")
        # print(f"Mean Squared Error: {mean_2}")
        # print("predictions: ")
        # print(yp)
        # print("ground truth: ")
        # print(y_test)
        return mean_1, mean_2, r, yp