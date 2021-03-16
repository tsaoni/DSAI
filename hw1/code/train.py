import numpy as np
import pandas as pd
import os
import sys

class Model:
    def __init__(self):
        self.days = 5
        self.df = pd.DataFrame({})
        self.x = np.array([], dtype = float)
        self.y = np.array([], dtype = float)
        self.num = 0
        self.x_mean = np.array([])
        self.y_mean = np.array([])
        self.x_var = np.array([])
        self.y_var = np.array([])
        self.x_div = np.array([])
        self.y_div = np.array([])
        self.iter_num = 1000
        self.parameters = np.zeros(6 * self.days + 1)

    def train(self, df_training):
        self.df = df_training
        i = 0
        for col in self.df.columns:
            i += 1
            if i > 7:
                del(self.df[col])
#        print(self.df.columns)
#        print(self.df)

        i = 0
        cur_row = 0
        date = self.df.iloc[:, 0]

        for d in date:
            if d % 100 > self.days:
                for i in range(1, 7):
                    self.x = np.append(self.x, self.df.iloc[cur_row - self.days: cur_row, i])
                self.y = np.append(self.y, self.df.iloc[cur_row, 3])
                self.num += 1
            cur_row += 1

        print(np.size(self.y), self.num)
        self.x = np.hsplit(self.x, self.num)
        print(type(self.x))
        print(self.x)
#        print(self.y)

        self.x = np.array(self.x) 
        self.x_mean = self.x.sum(axis = 0) / self.num
        self.y_mean = self.y.sum() / self.num
        self.x_var = (self.x ** 2 / self.num).sum(axis = 0) - self.x_mean ** 2
        self.y_var = (self.y ** 2 / self.num).sum() - self.y_mean ** 2
#        self.x_mean = self.x_mean.astype(int)
        print(self.x_mean)
        self.x_div = np.sqrt(self.x_var)
        self.y_div = np.sqrt(self.y_var)
#        self.x_div = self.x_div.astype(int)
        print(self.x_div)

        for i in range(0, self.num):
            self.x[i] = np.subtract(self.x[i], self.x_mean) / self.x_div

        self.y = (self.y - self.y_mean) / self.y_div

        print(self.x)
        print(self.y)

        self.parameters[2 * self.days : 3 * self.days] = 1 / self.days
        print(self.parameters)

        self.x = np.concatenate((self.x, np.ones((self.num, 1))), axis = 1)
        np.set_printoptions(threshold = sys.maxsize)
        print(self.x)


        #linear regression implementation
        learning_rate = 0.0001
        adagrad = np.zeros(6 * self.days + 1)
        eps = 0.0000000001

        for i in range(0, self.iter_num):
            diff = np.array([])
            for j in range(0, self.num):
                diff = np.append(diff, self.y[j] - np.dot(self.x[j], self.parameters))
            loss = np.sqrt(np.sum(diff ** 2) / self.num)
            print("loss: ", loss)
            gradient = np.array([])
            for i in range(0, self.days * 6 + 1):
                gradient = np.append(gradient, -2 * np.dot(diff, np.transpose(self.x)[i]))
            adagrad += gradient ** 2
            self.parameters = self.parameters - learning_rate / np.sqrt(adagrad + eps) * gradient
            print(self.parameters)

    def predict(self, n_step):
        cur_path = os.path.dirname(__file__)
        new_path = os.path.relpath("../dataset/data1.csv", cur_path)
        df = pd.read_csv(new_path)

        pred_y = np.array([])
        for i in range(self.days, len(df.index)):
            pred_x = np.array([])
            for j in range(1, 7):
                pred_x = np.append(pred_x, df.iloc[i - self.days : i, j])
            pred_x = np.append(pred_x, 1)
            pred_y = np.append(pred_y, np.dot(pred_x, self.parameters))
        print(pred_y)
        return pred_y

