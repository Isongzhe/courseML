**機器學習與主動影像處理**

*<說明來源>* <br>

這個儲存庫包含了我為課程的期中作業所完成的專案。<br>
作業的要求是使用 "iris" 數據集來完成一個分類任務，並且只能使用 numpy、pandas 和 matplotlib 這三個套件。 <br>
這意味著我需要自己定義訓練過程中的前向傳播和反向傳播，以及大部分的code，不可使用其餘ML套件。 <br>
由於我之前在類似的任務中使用過 keras 和 pytorch，對於整個架構有經驗，故這次自學使用OOP寫法來完成這個簡單任務。<br>
我將超參數、繪圖和激活函數獨立出來，可以方便地進行未來的進行微調與更新。<br>
目前還是學生，這是我第一份投稿，有任何建議都請聯絡我!

*<使用>* <br>

下載所有檔案，包含`midLabML.ipynb`以及`dataset file`，並裝在同一個資料夾下。<br>
你可以選擇直接執行`midLabML.ipynb`,我已經分區並註解，相信能幫你理解其用作。
又或是你可以選擇匯入model到你本地端檔案

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
