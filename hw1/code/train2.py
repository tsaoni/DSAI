from datetime import timedelta, date, datetime
import numpy as np             #for numerical computations like log,exp,sqrt etc
import pandas as pd            #for reading & storing data, pre-processing
import matplotlib.pylab as plt #for visualization
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 10, 6

class Model:
    def __init__(self):
        self.df = pd.DataFrame({})
        self.result = None

    def test_stationarity(self, timeseries):

        #Determine rolling statistics
        movingAverage = timeseries.rolling(window=14).mean()
        movingSTD = timeseries.rolling(window=14).std()

        #Plot rolling statistics
        orig = plt.plot(timeseries, color='blue', label='Original')
        mean = plt.plot(movingAverage, color='red', label='Rolling Mean')
        std = plt.plot(movingSTD, color='black', label='Rolling Std')
        plt.legend(loc='best')
        plt.title('Rolling Mean & Standard Deviation')
        #plt.show(block=False)
        plt.show()

        #Perform Dickey–Fuller test:
        print('Results of Dickey Fuller Test:')
        dftest = adfuller(timeseries["備轉容量(萬瓩)"], autolag='AIC')
        dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
        for key,value in dftest[4].items():
            dfoutput['Critical Value (%s)'%key] = value
        print(dfoutput)


    def train(self, df_training):
        # ***adjuting time fomat***
        self.df = df_training[["日期", "備轉容量(萬瓩)"]]
        year = []
        month = []
        day = []
        for x in self.df["日期"]:
            year.append(x.split("/", 2)[0])
            month.append(x.split("/", 2)[1])
            day.append(x.split("/", 2)[2])
        self.df.insert(loc = 0, column = "year", value = year)
        self.df.insert(loc = 1, column = "month", value = month)
        self.df.insert(loc = 2, column = "day", value = day)
        self.df.loc[:, "日期"] = pd.to_datetime(self.df[["year", "month", "day"]])
        self.df = self.df.drop("year", axis = 1)
        self.df = self.df.drop("month", axis = 1)
        self.df = self.df.drop("day", axis = 1)
        print(self.df)
        indexedDataset = self.df.set_index(['日期'])
        indexedDataset.head(5)
        print(indexedDataset)
        """         
        plt.xlabel("date")
        plt.ylabel("Backup capacity")
        plt.plot(indexedDataset)
        plt.show()        
        """
        rolmean = indexedDataset.rolling(window=14).mean()
        rolstd = indexedDataset.rolling(window=14).std()
        print(rolmean,rolstd)

        #Plot rolling statistics
        """
        orig = plt.plot(indexedDataset, color='blue', label='Original')
        mean = plt.plot(rolmean, color='red', label='Rolling Mean')
        std = plt.plot(rolstd, color='black', label='Rolling Std')
        plt.legend(loc='best')
        plt.title('Rolling Mean & Standard Deviation')
        #plt.show(block=False)
        plt.show()
        """

        #Perform Augmented Dickey–Fuller test:
        print('Results of Dickey Fuller Test:')
        dftest = adfuller(indexedDataset["備轉容量(萬瓩)"], autolag='AIC')

        dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
        for key,value in dftest[4].items():
            dfoutput['Critical Value (%s)'%key] = value
    
        print(dfoutput)

        # ***Data Transformation to achieve Stationarity***

        # **Log Scale Transformation**

        #Estimating trend
        indexedDataset_logScale = np.log(indexedDataset)
        """        
        plt.plot(indexedDataset_logScale)
        plt.show()
        """

        #The below transformation is required to make series stationary
        movingAverage = indexedDataset_logScale.rolling(window=14).mean()
        movingSTD = indexedDataset_logScale.rolling(window=14).std()
        """
        plt.plot(indexedDataset_logScale)
        plt.plot(movingAverage, color='red')
        plt.show()
        """

        datasetLogScaleMinusMovingAverage = indexedDataset_logScale - movingAverage
        datasetLogScaleMinusMovingAverage.head(12)

        #Remove NAN values
        datasetLogScaleMinusMovingAverage.dropna(inplace=True)
        datasetLogScaleMinusMovingAverage.head(10)

        #self.test_stationarity(datasetLogScaleMinusMovingAverage)


        # **Exponential Decay Transformation**

        exponentialDecayWeightedAverage = indexedDataset_logScale.ewm(halflife=12, min_periods=0, adjust=True).mean()
        """
        plt.plot(indexedDataset_logScale)
        plt.plot(exponentialDecayWeightedAverage, color='red')
        plt.show()
        """

        datasetLogScaleMinusExponentialMovingAverage = indexedDataset_logScale - exponentialDecayWeightedAverage
        #self.test_stationarity(datasetLogScaleMinusExponentialMovingAverage)


        # **Time Shift Transformation**

        datasetLogDiffShifting = indexedDataset_logScale - indexedDataset_logScale.shift()
        """ 
        plt.plot(datasetLogDiffShifting)
        plt.show()
        """

        datasetLogDiffShifting.dropna(inplace=True)
        #self.test_stationarity(datasetLogDiffShifting)

        decomposition = seasonal_decompose(indexedDataset_logScale) 

        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid

        """
        plt.subplot(411)
        plt.plot(indexedDataset_logScale, label='Original')
        plt.legend(loc='best')

        plt.subplot(412)
        plt.plot(trend, label='Trend')
        plt.legend(loc='best')

        plt.subplot(411)
        plt.plot(seasonal, label='Seasonality')
        plt.legend(loc='best')

        plt.subplot(411)
        plt.plot(residual, label='Residuals')
        plt.legend(loc='best')

        plt.tight_layout()
        """

        #there can be cases where an observation simply consisted of trend & seasonality. In that case, there won't be 
        #any residual component & that would be a null or NaN. Hence, we also remove such cases.
        decomposedLogData = residual
        decomposedLogData.dropna(inplace=True)
        #self.test_stationarity(decomposedLogData)

        # ***Plotting ACF & PACF***

        #ACF & PACF plots

        lag_acf = acf(datasetLogDiffShifting, nlags=20)
        lag_pacf = pacf(datasetLogDiffShifting, nlags=20, method='ols')

        """ 
        #Plot ACF:
        plt.subplot(121)
        plt.plot(lag_acf)
        plt.axhline(y=0, linestyle='--', color='gray')
        plt.axhline(y=-1.96/np.sqrt(len(datasetLogDiffShifting)), linestyle='--', color='gray')
        plt.axhline(y=1.96/np.sqrt(len(datasetLogDiffShifting)), linestyle='--', color='gray')
        plt.title('Autocorrelation Function')

        #Plot PACF
        plt.subplot(122)
        plt.plot(lag_pacf)
        plt.axhline(y=0, linestyle='--', color='gray')
        plt.axhline(y=-1.96/np.sqrt(len(datasetLogDiffShifting)), linestyle='--', color='gray')
        plt.axhline(y=1.96/np.sqrt(len(datasetLogDiffShifting)), linestyle='--', color='gray')
        plt.title('Partial Autocorrelation Function')

        plt.tight_layout()
        plt.show()
        """

        # ***Building Models***


        #AR Model
        #making order=(2,1,0) gives RSS=1.5023
        model = ARIMA(indexedDataset_logScale, order=(2,1,0))
        results_AR = model.fit(disp=-1)
        """
        plt.plot(datasetLogDiffShifting)
        plt.plot(results_AR.fittedvalues, color='red')
        plt.title('RSS: %.4f'%sum((results_AR.fittedvalues - datasetLogDiffShifting["備轉容量(萬瓩)"])**2))
        plt.show()
        print('Plotting AR model')
        """

        #MA Model
        model = ARIMA(indexedDataset_logScale, order=(0,1,2))
        results_MA = model.fit(disp=-1)
        """
        plt.plot(datasetLogDiffShifting)
        plt.plot(results_MA.fittedvalues, color='red')
        plt.title('RSS: %.4f'%sum((results_MA.fittedvalues - datasetLogDiffShifting["備轉容量(萬瓩)"])**2))
        plt.show()
        print('Plotting MA model')
        """

        # AR+I+MA = ARIMA model
        model = ARIMA(indexedDataset_logScale, order=(2,1,2))
        results_ARIMA = model.fit(disp=-1)
        """
        plt.plot(datasetLogDiffShifting)
        plt.plot(results_ARIMA.fittedvalues, color='red')
        plt.title('RSS: %.4f'%sum((results_ARIMA.fittedvalues - datasetLogDiffShifting["備轉容量(萬瓩)"])**2))
        plt.show()
        """

        # ***Prediction & Reverse transformations***


        predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
        print(predictions_ARIMA_diff.head())

        #Convert to cumulative sum
        predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
        print(predictions_ARIMA_diff_cumsum)

        predictions_ARIMA_log = pd.Series(indexedDataset_logScale["備轉容量(萬瓩)"].iloc[0], index=indexedDataset_logScale.index)
        predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)
        predictions_ARIMA_log.head()

        predictions_ARIMA = np.exp(predictions_ARIMA_log)
        """
        plt.plot(indexedDataset)
        plt.plot(predictions_ARIMA)
        plt.show()
        """

        self.result = results_ARIMA


    def predict(self, n_step):

        #self.result.plot_predict(1, 469)
        print(np.exp(self.result.forecast(120)[0]))
        target = np.exp(self.result.forecast(120)[0])
        plt.show()

        dates = []
        target = [x * 10 for x in target[97: 97 + n_step]]
        start_date = date(2021, 3, 23)

        for d in range(0, n_step):
            dates.append((start_date + timedelta(d)).strftime("%Y%m%d"))

        predict = pd.DataFrame({})
        predict["date"] = dates
        predict["operating_reserve(MW)"] = target
        print(predict)
        
        return predict
