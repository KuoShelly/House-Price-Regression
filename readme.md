# Home Price Regression
這段程式碼是針對 Kaggle 的房價預測競賽，使用 Python 實現了一個三層的集成學習模型。以下是對這個集成學習架構的簡要描述：

**Level-1 模型：Ridge、Lasso、Elastic Net**
- 在第一層，選擇了三種回歸模型，分別是 Ridge、Lasso 和 Elastic Net。這些模型被用於預測目標變數（房價）。

**Level-2 模型：Voting 和基於 GBDT 的 Stacking**
- 在第二層，使用 VotingRegressor 將前一層的三個模型整合在一起。同時，也使用了以 Gradient Boosting Decision Tree (GBDT) 為基礎的 StackingRegressor，將前一層的模型作為子模型進行集成。

**Level-3 模型：Blending**
- 在第三層，將前一層的 Voting 和 Stacking 模型進行 Blending。Blending 是一種集成學習的技術，通過線性加權將不同模型的預測結果混合在一起，以獲得最終預測結果。

這樣的三層模型架構旨在提升整體模型的效果和穩定性。使用不同層次的模型進行集成，有助於充分利用各個模型的優勢，提高對房價預測的準確性。最終的預測結果經過 Blending 的加權處理，得到一個更為綜合和強健的模型。


## EDA (Exploratory Data Analysis)

```python
# Coding block
if __name__ == "__main":
    print("Hello World")
```

