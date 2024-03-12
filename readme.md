# Home Price Regression
這段程式碼是針對 Kaggle 的房價預測競賽，使用 Python 實現了一個學習模型。以下是對這個學習架構的簡要描述：

## 資料理解
在進行資料分析及建模前，下載建模用的資料集及schema。透過閱讀schema，將變數大致分為以下類別：
#### 1. 建物面積：
   - `1stFlrSF`（一樓面積）、`GrLivArea`（一樓以上總生活空間）、`TotalBsmtSF`（地下室總面積）等。

#### 2. 裝置數量：
   - `FullBath`（完整浴室數量）、`Kitchen`（廚房數量）、`Bedroom`（臥室數量）等。

#### 3. 量表型類別特徵：
   - `ExterCond`（外部狀態）、`ExterQual`（外觀材質）、`GarageQual`（車庫品質）等。
      這類型特徵的值為Excellent/Good/Averege/Fair/Poor等，具有順序特性。

#### 4. 一般類別特徵：
   - `Neighborhood`（所在區域）、`HouseStyle`（房屋型式）等。
     
在進行探索性資料分析（EDA）和實際建模之前，理解資料型態很重要。資料本身可能隱含了豐富的領域知識，這些資訊與後續的特徵工程處理手法密切相關。

## 相關係數確認
將train_df的數值特徵跟預測變數(SalePrice)的相關係數矩陣，然後把相關性最高的幾個特徵列出來，並刪除相關性低的特徵(相關係數小於0.3)。
```python
# 相關性熱點圖
num_data_train = train_df.select_dtypes(include=['int', 'float'])

# 計算相關性
num_corr = num_data_train.corr()

# 調整圖形大小
fig, ax = plt.subplots(figsize=(15, 1))

# 繪製相關性熱點圖
sns.heatmap(num_corr.sort_values(
    by=['SalePrice'], ascending=False).head(1), cmap='Reds')

# 設置標題和軸標籤
plt.title("Correlation Matrix", weight='bold', fontsize=18)
plt.xticks(weight='bold')
plt.yticks(weight='bold', color='dodgerblue', rotation=0)

# 找出相關性小於0.3的特徵
low_corr_features = num_corr['SalePrice'][num_corr['SalePrice'] < 0.3].index

# 刪除相關性小於0.3的特徵
train_df = train_df.drop(low_corr_features, axis=1)
```
## 極端值處理
找出train_df裡的數值型特徵中屬於常態分佈的，並刪除其兩個標準差以外的值(留其95%信賴區間)
```python
# 極端值處理
num_train_data = train_df.select_dtypes(include=['int', 'float'])

# 找出常態分佈的features
normal_distributed_features = []
for feature in num_train_data:
    stat, p_value = shapiro(train_df[feature])
    if p_value > 0.05:
        normal_distributed_features.append(feature)

# 刪除於常態分佈features中95%區間外的值
for feature in normal_distributed_features: 
    mean_val = train_df[feature].mean()
    std_val = train_df[feature].std()
    lower_bound = mean_val - 1.96 * std_val
    upper_bound = mean_val + 1.96 * std_val

    # 刪除超出區間的值
    clean_train_df = train_df[(train_df[feature] >= lower_bound) & (
        train_df[feature] <= upper_bound)]
```
## 遺漏值檢查
將合併test_df及train_df的data進行遺漏值的檢查，若遺漏值太多，則刪除該特徵。
```python
# 遺漏值確認
missing_columns = data.isnull().mean().sort_values(ascending=False)
missing_columns = missing_columns[missing_columns !=
                                  0].to_frame().reset_index()

# 遺漏率繪圖
fig, ax = plt.subplots(figsize=(7, 7))
sns.barplot(x='index', y=0, data=missing_columns)
ax.set(xlabel='特徵', ylabel='遺漏率', title='特徵遺漏率')
plt.xticks(rotation=45, ha="right")  # 調整標籤旋轉角度
```
![遺漏率檢察](連結 "")

## EDA (Exploratory Data Analysis)
-
```python
## 模型預測
**Level-1 模型：Ridge、Lasso、Elastic Net**
- 在第一層，選擇了三種回歸模型，分別是 Ridge、Lasso 和 Elastic Net。這些模型被用於預測目標變數（房價）。

**Level-2 模型：Voting 和基於 GBDT 的 Stacking**
- 在第二層，使用 VotingRegressor 將前一層的三個模型整合在一起。同時，也使用了以 Gradient Boosting Decision Tree (GBDT) 為基礎的 StackingRegressor，將前一層的模型作為子模型進行集成。

**Level-3 模型：Blending**
- 在第三層，將前一層的 Voting 和 Stacking 模型進行 Blending。Blending 是一種集成學習的技術，通過線性加權將不同模型的預測結果混合在一起，以獲得最終預測結果。

這樣的三層模型架構旨在提升整體模型的效果和穩定性。使用不同層次的模型進行集成，有助於充分利用各個模型的優勢，提高對房價預測的準確性。最終的預測結果經過 Blending 的加權處理，得到一個更為綜合和強健的模型。



```

