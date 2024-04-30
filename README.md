**機器學習練習(不使用模型框架)**

*<開發中>* <br>
還再調整架構，預計之後嘗試提供GUI或網站去操作，是用戶能自行匯入分類資料集、選擇損失函數(RMSE/Cross-Entroy)、調整超參數等等

*<說明來源>* <br>
完成一個分類任務，並且只能使用 numpy、pandas 和 matplotlib 這三個套件。 <br>
練習定義訓練過程中的前向傳播和反向傳播。 <br>
自學練習使用OOP寫法來完成簡單分類任務。<br>
將超參數、和激活函數獨立出來，可以方便地進行未來的進行微調與更新。<br>

*<使用>* <br>
```python
import nbimporter #let file.ipynb can import to your file.py
from midLabML import NeuralNetwork

#you can fine tune hyperparameters by this dict
hyperparameters = {
    'n': 20,                 # n: number of hidden neurons
    'split_ratio': 0.5,      # split_ratio: train/test split ratio
    'epochs': 500,          # epochs: number of training epochs
    'learning_rate': 0.05   # learning_rate: learning rate
}

nn = NeuralNetwork(**hyperparameters)
nn.load_dataset()
nn.train()
```
Epoch: 1, RMSE: 0.6774 <br>
Epoch: 2, RMSE: 0.4455 <br>
Epoch: 3, RMSE: 0.3657 <br>
Epoch: 4, RMSE: 0.3371 <br>
Epoch: 5, RMSE: 0.3261 <br>
Epoch: 6, RMSE: 0.3197 <br>
Epoch: 7, RMSE: 0.3143 <br>
Epoch: 8, RMSE: 0.3092 <br>
Epoch: 9, RMSE: 0.3044 <br>
...

```python
predict_df, true_df = nn.predict() #get predicted dataframe
```
```python
nn.loss_plot()
```
![image](https://github.com/Isongzhe/courseML/assets/86179263/88aa2a49-297c-4792-96b7-829ba1e9e354)

```python
nn.predict_plot()
```
![image](https://github.com/Isongzhe/courseML/assets/86179263/36de9bb0-15cc-4c40-951f-9dd5a758c0f1)
