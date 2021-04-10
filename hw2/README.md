# DSAI HW2 - AutoTrading

##### F74072277 許郁翎

### 程式架構

* 主要資料夾底下有三個資料夾，分別是裝程式的code, 裝訓練資料的dataset, 和裝預測最佳action(也就是output.csv)的output。

* code底下的trader.py為主程式所在檔案，DDQN.py為存放class Trader的attribute跟method檔案，裡面也包含實作DDQN的程式

* output底下的pre_output裝的是output的歷史紀錄(每一次執行程式，output結果皆會不同)

### 程式執行

```
python trader.py --training training.csv --testing testing.csv --output output.csv
```
###### 執行時間約為309.2245971789962 sec

### 參考範例

Deep Reinforcement Learning on Stock Data
https://www.kaggle.com/itoeiji/deep-reinforcement-learning-on-stock-data

### 程式碼運作
In DDQN.py...

#### Trader為預測股票的model，它的attribute有下列3個
* test_env (testing的時候會使用到的environment)
* test_pobs (testing的時候，儲存上一個action執行過後所產生的observation)
* Q (加強式學習(這裡是Q learning)中扮演actor的角色，計算每一個action被選擇的可能性)
#### 而method為下述2個
* train (主要功能為訓練DDQN的模型)
* prediction_action (回傳輸入action後預測最佳action)

#### Environment1為訓練加強式學習中的環境，其中包含下面兩個method
* reset (初始化參數)
* step (回傳輸入action後，新產生的observation跟所得到的reward，過程也會計算所得獲利)

#### train_ddqn為訓練模型的method，裡面包含Q_Network的class，以及DDQN訓練過程 (最核心重要的程式碼，也幾乎是複製kaggle的部分QQ)

#### plotting function

* plot_train_test_by_q (印出模型在股票曲線圖所會採取的action)
##### 其中，灰色為持有，藍色為買入，紅色為賣出
![](https://i.imgur.com/qU02Qr8.png)
###### 很勤勞但是獲利極低
![](https://i.imgur.com/nJnrOYV.png)
* plot_train_test (印出股票趨勢圖 ~~它不重要~~)
* plot_loss_reward (印出training過程中，loss跟reward的變化)
![](https://i.imgur.com/zgWW6Bm.png)
###### 理論上reward是要遞增的說..

### 程式歷史output所得的profit紀錄

```
4.090000000000032
1.5
1.1999999999999602
4.3799999999999955
3.3799999999999955
3.0
0 (有時候會出現完全沒有任何買賣的情況，求好心der助教幫忙多執行幾次(大誤QQ
0
1.3000000000000398
-0.8899999999999579 (arrrrr
0.4199999999999875
...
```

