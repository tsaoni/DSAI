# DSAI HW1 - Electricity Forecasting

###### F74072277 許郁翎

### 程式架構

主要資料夾底下有三個資料夾，分別是裝程式的code, 裝訓練資料的dataset, 和裝預測結果(也就是submission.csv)的output。

code底下有含主程式的app.py，class Model的attribute跟method的train2.py，跟裝未完成的程式資料夾unfinish。

### 程式執行

```
python app.py --training "Your Training Data" --output submission.csv
```

### 參考範例

Time Series For beginners with ARIMA
https://www.kaggle.com/freespirit08/time-series-for-beginners-with-arima


### 程式分析圖

#### 原始資料

原始訓練集的資料分布
![](https://i.imgur.com/NCOFtMi.png)

左方為ACF，右方為PACF，分別預測Q(moving avg.)跟P(auto regressive lags)的值
![](https://i.imgur.com/koGxdpA.png)


AR model預測的結果
![](https://i.imgur.com/khwsxP5.png)

MA model預測的結果
![](https://i.imgur.com/PutSyDV.png)


預測結果
![](https://i.imgur.com/ISUbmzs.png)


