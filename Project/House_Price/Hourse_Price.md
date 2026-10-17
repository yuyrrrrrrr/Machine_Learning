# 前言

本文主要针对**机器学习、数据科学、数据分析初学者**，提供了 Kaggle 的入门赛 House Prices 的解题思路。在学习前，需要对 **python** 的基本语法，如**变量、字符串、函数**等有一定了解。在本文中，将学习如何用 python 进行入门级的**数据清洗、特征工程、模型训练、模型融合**，从而实现数据科学目的。

Kaggle 是全球最大的数据科学竞赛与学习社区，2010 年创立、2017 年被 Google 收购。House Prices 全称：House Prices – Advanced Regression Techniques，是 Kaggle 最经典的入门回归比赛。任务目标是，用房屋的 79 个特征（面积、房龄、卧室数、装修、地段等），预测美国爱荷华州艾姆斯市的房屋售价（SalePrice）。

本文的最优成绩是  ，截至撰写日期（2026年2月27日），在 3951 支参赛队伍中排名 496 ，不算优秀，但对于初学者而言，**代码简单，思路易理解，参数可解释性较强**，可以作为入门学习参考。如果在学完本文后，能写出一个“能跑起来”的入门代码，哪怕分数很低，也算完成目标。

# 总体思路

## 题目分析

题目给出两个数据集，分为**训练集**和**测试集**。

训练集，一共 1460 行，81 列。其中，每行代表一栋房屋的相关数据。前 80 列包括各类特征，如装修质量、地上面积、车库面积、卧室数等等……以字符串和数字的形式给出。最后一列为房价，也就是我们要预测的目标。

测试集，一共 1459 行，80 列。同样，每行也代表一栋房屋的相关数据。相比训练集，少了房价那一列，因为这是我们要预测的目标（没有卷子上面会印参考答案吧！）。我们最后需要提交的是一个 1459 行， 2 列的 csv 表格，第一列为房屋ID，第二列为预测该房屋的房价（相当于答题卡）。

最终评定成绩的指标为 **RMSLE（均方根对数误差）** 。如果不了解，可以简单理解为，越接近真实房价，最终该指标越低，意味着预测结果更准确。

## 思路介绍

本文采用机器学习的思路，使用 XGBoost 和 Lasso 模型。

### XGBoost介绍

XGBoost 是一种**集成学习模型**，是**机器学习**的一种，属于梯度提升树，它会一棵接一棵地生成决策树，每一棵都去修正前面模型的误差，通过不断迭代让预测越来越准。它擅长处理表格数据，拟合能力强，效果稳定，在 Kaggle 比赛里非常常用，适合房价预测这种回归任务。

如果把一棵决策树看作一个专家，XGBoost 就像是一群专家开会，从第一个专家开始轮流发言。假如真实房价是 50 万，刚开始，XGBoost 会随便预测一个值，比如 32 万。第一个专家发言时会进行修正，比如“加 15 万！”，现在预测值达到了 47 万。第二个专家发言时，认为“加 2 万！”，从而让预测值变成 49 万，后面还有几百个专家等着发言，每个专家都会去修正前人留下来的误差，从而让预测值逼近真实值。

### Lasso介绍

Lasso 是一种**线性回归模型**，也是**机器学习**的一种，核心是加入了 L1 正则化，会**让一些不重要的特征系数直接变成 0**，从而自动筛选特征、简化模型，防止过拟合。它适合特征很多但部分特征没用的场景，解释性比较好，计算也很快。

# 数据阅读

数据阅读往往是比赛的第一个环节。在这个环节中，要通过肉眼观察，对数据集有一个大概的了解。比如，一共有多少行数据？多少个特征？预测的目标是什么？数据大致呈现什么趋势？目标数据和哪些特征关系最为紧密？

## 粗读

请你跟我来，我们一起写代码：

```haskell
# 读入数据
import numpy as np
import pandas as pd
import os
import seaborn as sns
from xgboost import XGBRegressor
from scipy.stats import skew
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import Lasso
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline

train_data = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
test_data = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

print(train_data.head())
print(train_data.shape)
print(test_data.shape)
```

前面那一大堆 `import` 语句目前无需理会，用啥取啥就行……从 `read_csv` 语句开始看。这是 pandas 库提供的读取 csv 文件的方法，可以读取 csv 表格并转换成 DataFrame 变量（可以简单理解为一种便于运算的矩阵或表格）。

接下来的三行 `print` 语句，分别打印了：训练集的前五行信息、训练集的形状、测试集的形状。

输出：

```yaml
   Id  MSSubClass MSZoning  LotFrontage  LotArea Street Alley LotShape  \
0   1          60       RL         65.0     8450   Pave   NaN      Reg   
1   2          20       RL         80.0     9600   Pave   NaN      Reg   
2   3          60       RL         68.0    11250   Pave   NaN      IR1   
3   4          70       RL         60.0     9550   Pave   NaN      IR1   
4   5          60       RL         84.0    14260   Pave   NaN      IR1   

  LandContour Utilities  ... PoolArea PoolQC Fence MiscFeature MiscVal MoSold  \
0         Lvl    AllPub  ...        0    NaN   NaN         NaN       0      2   
1         Lvl    AllPub  ...        0    NaN   NaN         NaN       0      5   
2         Lvl    AllPub  ...        0    NaN   NaN         NaN       0      9   
3         Lvl    AllPub  ...        0    NaN   NaN         NaN       0      2   
4         Lvl    AllPub  ...        0    NaN   NaN         NaN       0     12   

  YrSold  SaleType  SaleCondition  SalePrice  
0   2008        WD         Normal     208500  
1   2007        WD         Normal     181500  
2   2008        WD         Normal     223500  
3   2006        WD        Abnorml     140000  
4   2008        WD         Normal     250000  

[5 rows x 81 columns]
(1460, 81)
(1459, 80)
```

可以清晰看到前五行的数据，如 ID 、泳池面积、房价等……

最后输出的 `(1460, 81) (1459, 80)`代表这两个数据集的形状：训练集有 1460 行 81 列，测试集有 1459 行 80 列，数据量很大程度上决定我们要选怎样的模型和参数。

## 细读

我们可以使用 seaborn 来画图。这是一个基于 matplotlib 的绘图库，代码简洁、图象美观。入门学习时，基本不需要死记硬背画图的代码，需要用的时候查文档或者问 AI 怎么用就行了。

```python
# 绘制房价直方图
sns.histplot(train_data['SalePrice'], kde=True)
```

![image](https://img2024.cnblogs.com/blog/3588980/202602/3588980-20260227164033327-1452880614.png)

这张图信息很多。比如，大部分房价集中在 100000 到 300000 之间，峰值出现在 150000 左右，最低 50000 左右，最高 700000 左右，而其实 400000 以上的房子就很少了。

可以看出，房价整体是**右偏**的，有一个很长的右拖尾。

什么是“右偏”？简单来说，图像上，房价并不是以 150000 为中心，左右对称的，而是在右边有一个很长——的拖尾，甚至拉到了 700000 去。放在现实中，这是很好理解的，大部分普通人住的房子也就那个价格大差不差，不过有极个别“豪宅”价格高的离谱。

这意味着，豪宅的影响会很大。你会发现赛方用的评估指标是**RMSLE（均方根对数误差）**。小学二年级，数学老师就告诉我们，对数函数图象单调递增，但斜率是单调递减的，也就是说，x 越大，x 的变化对 y 的影响越小。这样可以尽可能抵消极个别豪宅造成的巨大影响。举个例子，在 RMSLE 眼中， 140000 和 150000 差距很大，但 700000 和 800000 差距也许甚至没那么大。

对了，我们的两个模型 XGBoost 和 Lasso 都接受不了这样右偏严重的目标数据，所以投喂给模型前，我们需要先将房价进行**对数处理**。（你看，如果我们没有进行数据阅读，也许就不会注意到这一点）

## 再读

还不够！接着读！接下来我们需要知道，**哪些数据与房价关系最大？** 我们需要着重处理这些数据。

我们可以利用 pandas 的相关性矩阵，找出所有列两两之间的相关性，挑出“与房价的相关性”那一列，并降序排序，相关性越强越靠前，然后挑出前十个打印出来。

```python
# 数字特征相关性
corr_matrix = train_data.corr(numeric_only=True) # 相关性矩阵
corr_price = corr_matrix["SalePrice"].sort_values(ascending = False) # 与房价相关性列，按降序排列
print(corr_price.iloc[:15]) # 前十个特征及其相关性
```

输出：

```yaml
SalePrice       1.000000
OverallQual     0.790982
GrLivArea       0.708624
GarageCars      0.640409
GarageArea      0.623431
TotalBsmtSF     0.613581
1stFlrSF        0.605852
FullBath        0.560664
TotRmsAbvGrd    0.533723
YearBuilt       0.522897
YearRemodAdd    0.507101
GarageYrBlt     0.486362
MasVnrArea      0.477493
Fireplaces      0.466929
BsmtFinSF1      0.386420
Name: SalePrice, dtype: float64
```

第一行，自然是房价它自己与自己相关性最强，这无需理会，我们看后面：

相关性最强的是 OverallQual 这个特征，即“整体品质”。然后是 GrLivArea ，即“地上居住面积”。再往后是车库面积、全浴室、建成年份等等等等……

# 数据清洗

## 异常值处理

根据刚才的观察，整体品质的影响是非常大的。我们画出装修品质与房价的散点图：

```haskell
# 绘制材质及装修品质与房价散点图
sns.scatterplot(x="OverallQual", y="SalePrice", data=train_data)
```

![image](https://img2024.cnblogs.com/blog/3588980/202602/3588980-20260227170437807-1360741297.png)

依旧观察一手……整体看来，房价随着装修品质的提高而上升，这很符合常识。但是，有两个点，整体品质明明为 10 级，但是价格低到了 20 万以下！那人家房地产商又不傻，哪有那么好的事情！一定是特例！

经过我的手动检查（这一步不展示代码，大家可以自行研究，用print或者画图都可以），发现这两个房子的地上面积竟然在 4000 以上，堪称超级大豪宅，所以这一定是搞错了！这种点我们称为异常点，直接删掉！

```bash
# 有两个品质10级但价格小于200000的点
print(train_data[(train_data["OverallQual"] == 10) & (train_data["SalePrice"] <= 200000)][["OverallQual", "GrLivArea", "SalePrice"]])
# 经过检查这两个点地上面积大于4000，几乎不可能低价，无参考价值
# 异常值删掉
train_data = train_data.drop(train_data[(train_data["OverallQual"] == 10) & (train_data["SalePrice"] <= 200000)].index)
```

刚才看到，房价还有一个重要决定因素：地上居住面积。我们画出对应的散点图：

```haskell
# 绘制地上面积与房价散点图
sns.scatterplot(x="GrLivArea", y="SalePrice", data=train_data)
```

![image](https://img2024.cnblogs.com/blog/3588980/202602/3588980-20260227170904224-418352017.png)

整体看来，房价随着地上居住面积上升而增大，非常合理！大致观察，没有奇怪的点，就放过它们吧。

## 缺失值处理

### 缺失值

在原始的表格中，无论是训练集还是测试集，都会有一些缺失的信息。对于缺失值，我们主要采取 **“填充”** 和 **“删除”** 两种策略。“删除”可以分为删除该行，或者删除该列。在本文中没有使用“删除”，我们重点进行“填充”。

我们先扫描一下缺失值的情况，有一个大概的了解：

```bash
# 扫描缺失值情况
missing_data = train_data.isnull().sum()
print(missing_data[missing_data > 0])
```

输出：

```yaml
LotFrontage      259
Alley           1367
MasVnrType       872
MasVnrArea         8
BsmtQual          37
BsmtCond          37
BsmtExposure      38
BsmtFinType1      37
BsmtFinType2      38
Electrical         1
FireplaceQu      690
GarageType        81
GarageYrBlt       81
GarageFinish      81
GarageQual        81
GarageCond        81
PoolQC          1452
Fence           1177
MiscFeature     1404
dtype: int64
```

“填充”主要分为以下 3 类，需要根据特征的情况讨论：

### 填充“None”或 0

有时候，缺失值本身也是一种信息：

PoolQC（泳池质量）缺失，有没有可能就是因为这个房子，它就没有泳池？

GarageType（车库类型）缺失，有没有可能是因为这个房子，它就不带车库？

TotalBsmtSF（地下室总面积）缺失，有没有可能是因为这个房子，它就没有地下室？

那你没泳池、没车库、没地下室的房子，可不就应该卖得便宜点嘛！所以对于这类信息，类别特征填充 None，数值特征填充 0，更为妥当。

### 填充中位数

LotFrontage（临街宽度）缺失，有没有可能是因为它没有临街？NO！根据常识，房子旁边总有路……所以，这类数据缺失不是因为“没有”，而是因为没有采集到，属于“随机缺失”。为了弥补它的缺失带来的影响，我们往往填充中位数更妥当。

### 填充众数

SaleType（房屋销售类型）、Electrical（电气系统类型）、MSZoning（所在区域用地分类）、Exterior1st（外立面主要材质）……这些东西，也是每个房子都有的！所以不能填 None。但是由于是类别特征，没有什么“中位数”，所以只能填充“众数”。

### 缺失值填充

我整理了所有缺失特征的分类，代码如下：

```bash
# 处理缺失值函数
def fill_missing(df):
    df = df.copy()
    # 第一类：需要填充None的字符串
    none_cols = [
        'Alley', 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
        'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu',
        'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
        'PoolQC', 'Fence', 'MiscFeature'
    ]
    df[none_cols] = df[none_cols].fillna("None")
    # 第二类：需要填充0的数值
    zero_cols =[
        'GarageYrBlt', 'GarageArea', 'GarageCars', 
        'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 
        'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'
    ]
    df[zero_cols] = df[zero_cols].fillna(0)
    # 第三类：需要填充中位数的数值
    df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].median())
    # 第四类：需要填充众数的特征
    mode_cols =[
        'Electrical', 'MSZoning', 'Exterior1st', 'Exterior2nd',
        'SaleType', "Utilities", "KitchenQual", "Functional"
    ]
    for col in mode_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    return df

train_data = fill_missing(train_data)
missing_data = train_data.isnull().sum().sum()
print(f"清洗后训练数据缺失值数量：{missing_data}")
test_data = fill_missing(test_data)
missing_data = test_data.isnull().sum().sum()
print(f"清洗后测试数据缺失值数量：{missing_data}")
```

输出：

```undefined
清洗后训练数据缺失值数量：0
清洗后测试数据缺失值数量：0
```

至此，数据清洗完成，心里暖暖的……

# 特征工程

## 对数转换

还记得之前我们说，房价是“右偏”的，需要进行 **“对数转换”** 吗？

那具体如何进行呢？

我们可以利用 numpy 中的 `log1p` 和 `expm1` 方法进行转换。

![image-20261012210935371](C:\Users\27813\AppData\Roaming\Typora\typora-user-images\image-20261012210935371.png)

两者是互逆运算，也就是说：

![image-20261012210945607](C:\Users\27813\AppData\Roaming\Typora\typora-user-images\image-20261012210945607.png)

所以我们可以先对房价进行 `log1p` 处理，让模型去预测处理之后的值，然后再将预测结果用 `expm1` 处理回来！

## 提取目标特征

Id 只是一个随机的序列号，它与房子的本质属性无关。如果模型记住了 Id 对应的价格，那就是纯粹的过拟合。所以 Id 这一列需要被去掉！

划分训练集的 X 和 y， X 记得去掉 ID 和 房价。测试集记得去掉 ID。

## Yeo-Johnson变换

很多特征是左偏/右偏的。这对于决策树模型（XGBoost）来说，影响没那么大，但是对于线性模型 Lasso 来说，是致命的打击。

大佬已经帮我们写好了办法：我们计算每个特征的“偏度”，大于 0 就说明右偏，小于 0 就说明左偏，然后将偏度绝对值大于 0.5 的特征提出来，进行一次 Yeo-Johnson 变换。

这个变换可以处理所有实数，对于右偏数据，使用近似 `log1p` 的方式转换；对于左偏数据，使用近似平方转换的方式转换；它非常强大，可以根据数据具体情况自动调整转换方式，使得数据尽可能接近正态分布。

注意，这个变换中，我们希望训练集和测试集的标准尽可能是一样的的，所以需要先合并起来再进行变换。

## 独热编码

对于类别数据，XGBoost 和 Lasso 是识别不了的，我们要将它们转换成数值型。

比如性别，如果有这样一个表格：

|   姓名   | 性别 |
| :------: | :--: |
|   暖暖   |  女  |
|   大喵   |  雄  |
|  乔万尼  |  男  |
| 吉吉罗达 |  雄  |

独热编码后就会变成这样：

|   姓名   | 性别_女 | 性别_雄 | 性别_男 |
| :------: | :-----: | :-----: | :-----: |
|   暖暖   |    1    |    0    |    0    |
|   大喵   |    0    |    1    |    0    |
|  乔万尼  |    0    |    0    |    1    |
| 吉吉罗达 |    0    |    1    |    0    |

这样，类别型就变成了数值型，便于模型学习。

注意，有的类别可能只在训练集中出现，有的只在测试集中出现：

想象，如果上表中，暖暖和大喵在测试集，乔万尼和吉吉罗达在训练集，那么独热编码之后，训练集就没有 “性别_男 ”这列，而测试集就没有 “性别_女” 这一列，导致特征名/特征数量可能出现不同。所以这一步也要在合并训练集与测试集的情况下完成。

pandas 提供了对 DataFrame 一键独热编码的方法：`get_dummies()`，非常省心。

## 特征工程代码

```python
# 特征工程

# 提取目标变量进行对数转换
y_train = np.log1p(train_data['SalePrice'])

# 提取目标特征
X_train = train_data.drop(["Id", "SalePrice"], axis = 1)
X_test = test_data.drop(["Id"], axis = 1)
ntrain = X_train.shape[0]

# 合并训练集与测试集
all_data = pd.concat([X_train, X_test], axis = 0, ignore_index = True)

# Yeo-Johnson变换
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
skewed_feats = all_data[numeric_feats].apply(skew).sort_values(ascending = False)
skewness = pd.DataFrame({'Skew':skewed_feats})
skewness = skewness[abs(skewness['Skew']) > 0.5]
skewed_features = skewness.index
print(f"{len(skewed_features)}个数值特征有拖尾...")
pt = PowerTransformer(method='yeo-johnson', standardize=True)
all_data[skewed_features] = pt.fit_transform(all_data[skewed_features])
print(f"已完成变换...")

# 独热编码
all_data = pd.get_dummies(all_data)

# 拆分训练集与测试集
X_train = all_data.iloc[:ntrain, :]
X_test = all_data.iloc[ntrain:, :]
print(f"最终 X_train 形状 (进模型前): {X_train.shape}")
print(f"最终 y_train 形状 (进模型前): {y_train.shape}")
print(f"最终 X_test 形状 (进模型前): {X_test.shape}")
```

输出：

```erlang
28个数值特征有拖尾...
已完成变换...
最终 X_train 形状 (进模型前): (1458, 301)
最终 y_train 形状 (进模型前): (1458,)
最终 X_test 形状 (进模型前): (1459, 301)
```

# 模型训练与预测

## 模型配置

```makefile
# 模型
model_xgb= XGBRegressor(
    n_estimators=1500,    # 核心：决策树的数量
    learning_rate=0.01,   # 核心：学习率（步长）
    max_depth=4,          # 核心：单棵决策树的最大深度
    subsample=0.8,        # 防过拟合：训练每棵树时抽样的样本比例
    colsample_bytree=0.3, # 防过拟合：训练每棵树时抽样的特征比例
    min_child_weight=2,   # 防过拟合：叶子节点的最小样本权重和
    random_state=42       # 通用：固定随机种子，保证结果可复现
)

model_lasso = make_pipeline(
    RobustScaler(),  # 数据预处理：鲁棒标准化
    Lasso(
        alpha=0.0005,  # 核心：正则化强度
        random_state=42 # 通用：固定随机种子
    )
)
```

XGBoost 调参核心逻辑是 “小学习率 + 多树 + 限制树复杂度” ，本次参数组合（0.01 学习率 + 1500 棵树 + 深度 4）是房价预测时我调试的比较优的方案，既能拟合数据规律，又能避免过拟合。

Lasso 的 alpha 值是太小会保留冗余特征，太大可能删掉有用特征；0.0005 是适配千级数据量、79个特征的常用参数。

## 5折交叉验证

为了确保模型效果，会先将训练集随机分成“五份”，进行五轮测试：每轮，将其中四份拿去投喂给模型，另一份用来测试，这样可以简单测试模型的泛化能力（也就是遇到新问题的解决能力）。

```python
# 5折交叉验证
print("===5折交叉验证===")
score = - cross_val_score(model_xgb, X_train, y_train, cv = 5, scoring = "neg_root_mean_squared_error")
print(f"XGB5折交叉验证RMSE：{score.mean():.4f}")
score = -cross_val_score(model_lasso, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')
print(f"Lasso5折交叉验证RMSE：{score.mean():.4f}")
```

## 正式训练并预测

验证完，将全部的训练集投喂给模型，然后让它生成“答案”。

```erlang
print("\n===正式训练===")
model_xgb.fit(X_train, y_train)
print("XGB训练完成...")
model_lasso.fit(X_train, y_train)
print("Lasso训练完成...")
predictions_xgb = model_xgb.predict(X_test)
predictions_lasso = model_lasso.predict(X_test)
y_test = np.expm1(predictions_xgb * 0.4 + predictions_lasso * 0.6)
print("预测完成...")
```

XGBoost 和 Lasso 的常用比例是  ，但在本例中，使用  得到了更好的分数（个人猜测：由于数据量过小，XGBoost仍存在过拟合的情况，而Lasso的比例增大可以形成对冲）。这个比例也不一定是最优的，因为我尝试的组合不多，大家可以多调调以上参数，融合更多模型，看看能不能取得更好的成绩。

## 生成答案文件

生成一份答案 csv 文件就可以直接提交了。

```lua
# 生成输出文件
print("\n===生成输出文件===")
output = pd.DataFrame({'Id': test_data.Id, 'SalePrice': y_test})
output.to_csv('house_price_submission.csv', index=False)
print("生成成功...")
```

输出：

```erlang
===5折交叉验证===
XGB5折交叉验证RMSE：0.1129
Lasso5折交叉验证RMSE：0.1098

===正式训练===
XGB训练完成...
Lasso训练完成...
预测完成...

===生成输出文件===
生成成功...
```