# 泰坦尼克号幸存案例

## 1 解答竞赛问题的七个步骤

1. 定义问题
2. 获取训练和测试数据
3. 整理、准备、清洗数据
4. 分析、定义模式，并且探索数据
5. 建模，预测并解决问题
6. 可视化，报告，并展示问题解决步骤和最终答案
7. 提交结果

## 2 以上步骤解决的七个主要目标

1. Classifying 分类：对样本进行分类，可能也想要了解不同类别之间的联系或含义。
2. Correlating 关联：可能可以通过[训练集](https://zhida.zhihu.com/search?content_id=112784529&content_type=Article&match_order=1&q=训练集&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3OTE5NjIzNjMsInEiOiLorq3nu4Ppm4YiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxMTI3ODQ1MjksImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.zX3G74vLPdpM_aXvOE1WJUEG_IHhcNIQXuIJPvDWQps&zhida_source=entity)获取的数据特征来解决问题。哪些特征对我们解决问题贡献最显著？从[统计学](https://zhida.zhihu.com/search?content_id=112784529&content_type=Article&match_order=1&q=统计学&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3OTE5NjIzNjMsInEiOiLnu5_orqHlraYiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxMTI3ODQ1MjksImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.wHcMw652prjsffWfB0qf6H8CbI_Oo6gRQjKwn9P583k&zhida_source=entity)的角度来说，一个特征和分析目标之间是否有显著的相关性？当某一[特征值](https://zhida.zhihu.com/search?content_id=112784529&content_type=Article&match_order=1&q=特征值&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3OTE5NjIzNjMsInEiOiLnibnlvoHlgLwiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxMTI3ODQ1MjksImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.KQMOd-LpfmPErr9JGLEq6KmANzdzHh0yzeAfpizMdTE&zhida_source=entity)改变时，目标状态是否也会改变，反之亦然？数据集中的数值变量和分类变量都可以检测以上问题。我们可能也想知道除了“生还”或其他可能的目标之外的变量之间的关系。关联某些特定的变量可能对建立、计算或[关联变量](https://zhida.zhihu.com/search?content_id=112784529&content_type=Article&match_order=1&q=关联变量&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3OTE5NjIzNjMsInEiOiLlhbPogZTlj5jph48iLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxMTI3ODQ1MjksImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.3isW8wrzURAh0Jep7pTrWQuO-d0kZoayVX3kSR_Bm88&zhida_source=entity)有帮助。
3. Converting 转变：在建模阶段需要准备数据。根据模型算法的选择，我们可能需要把所有的变量都转化为数值变量。
4. Completing 填充：数据准备可能也会要求我们估计变量的缺失值。[模型算法](https://zhida.zhihu.com/search?content_id=112784529&content_type=Article&match_order=2&q=模型算法&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3OTE5NjIzNjMsInEiOiLmqKHlnovnrpfms5UiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxMTI3ODQ1MjksImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MiwiemRfdG9rZW4iOm51bGx9.N4teJ39lA3DPcfP0atwzKo28n1GB1HVC4ILtsOKRl_U&zhida_source=entity)在无缺失值的情况下可能会运行得更好。
5. Correcting 修正：我们分析的数据中可能有错误或者潜在的异常值，需要修正这些值，或者去除包含错误值的样本。一种方法是诊断我们的样本或特征中是否有任何异常值。或者需要完全删除某一变量，如果塔对于分析没有作用或者显著地使结果偏移了。
6. Creating 创造：我们可以根据现有的特征创造新的特征，使得新特征满足相关性、转变性和完备性的目标。
7. Charting 绘图：如何选择正确的可视化图表取决于数据的构造和分析目标。

## 3 定义问题

> On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. Translated 32% survival rate.
> One of the reasons that the [shipwreck](https://zhida.zhihu.com/search?content_id=112784529&content_type=Article&match_order=1&q=shipwreck&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3OTE5NjIzNjMsInEiOiJzaGlwd3JlY2siLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxMTI3ODQ1MjksImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.s-rib1NU5ePsXFkP7mgORwcsNTQkvZNH9iQ6MO1qGdM&zhida_source=entity) led to such loss of life was that there were not enough lifeboats for the passengers and crew.
> Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
> Knowing from a training set of samples listing passengers who survived or did not survive the Titanic disaster, can our model determine based on a given test dataset not containing the survival information, if these passengers in the test dataset survived or not.

分析训练集的数据，建立模型预测训练集中的乘客是否生还。

## 4 获取训练和测试数据

[调包侠](https://zhida.zhihu.com/search?content_id=112784529&content_type=Article&match_order=1&q=调包侠&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3OTE5NjIzNjMsInEiOiLosIPljIXkvqAiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxMTI3ODQ1MjksImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.wlNp5GQ-KSfG6RcP9cLh00b-VcHkxyS47JnXllVAV1w&zhida_source=entity)启动

```python
## data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

## visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


#获取数据
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
combine = [train_data,test_data]
```

## 5 描述数据

## 5.1 提出关于数据的问题

- 哪些特征是可获取的？

```python
train_data.columns.values
---------------------------
array(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'], dtype=object)
```

- 哪些特征是无序的？
  这些值将样本分类成不同的相似的样本。是否有定类、定序、[定比](https://zhida.zhihu.com/search?content_id=112784529&content_type=Article&match_order=1&q=定比&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3OTE5NjIzNjMsInEiOiLlrprmr5QiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxMTI3ODQ1MjksImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.tcWqKK4ErHWWPKC1pLCz3q0eHdEAMNbPqDrsfxk1vA4&zhida_source=entity)或定矩变量基于目录型变量取值？根据其他相关，这能帮助我们选择合适的可视化图表。
  Categorical 无序变量: Survived, Sex, and Embarked.
  Ordinal[定序变量](https://zhida.zhihu.com/search?content_id=112784529&content_type=Article&match_order=1&q=定序变量&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3OTE5NjIzNjMsInEiOiLlrprluo_lj5jph48iLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxMTI3ODQ1MjksImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.iegqOqXC3V-PZF2_dP1MGhQcXkl1JR-813-k8BSJkDI&zhida_source=entity): Pclass.



- 哪些特征是定量的？
  这些值每个样本点都不同。在定量特征中是否有[离散](https://zhida.zhihu.com/search?content_id=112784529&content_type=Article&match_order=1&q=离散&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3OTE5NjIzNjMsInEiOiLnprvmlaMiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxMTI3ODQ1MjksImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.JmsF4AOBh_dEIpx9H5uVYRwm9ZFYmiiWlK0HPsKABBs&zhida_source=entity)、连续、或者时间序列存在？这些会帮助我们选择合适的可视化图表。
  Continous: Age, Fare.
  Discrete: SibSp, Parch.



- 哪些特征是混合数据类型？ [数值化](https://zhida.zhihu.com/search?content_id=112784529&content_type=Article&match_order=1&q=数值化&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3OTE5NjIzNjMsInEiOiLmlbDlgLzljJYiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxMTI3ODQ1MjksImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.a8oT4lnE06asMxnT4jqfM3xSYPlP0_Ykbk-v6JG-xAg&zhida_source=entity)、字母数字混合数据。这些是清洗目标的参考。
  Ticket is a mix of numeric and alphanumeric data types. Cabin is alphanumeric.



- 哪些特征可能包含错误值或记录错误？
  在一个巨大的数据集中很难检查，但是观察一些较小的样本可能会让我们明白哪些特征是需要清洗的。
  [姓名变量](https://zhida.zhihu.com/search?content_id=112784529&content_type=Article&match_order=1&q=姓名变量&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3OTE5NjIzNjMsInEiOiLlp5PlkI3lj5jph48iLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxMTI3ODQ1MjksImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.KGBhrBO-fRvJyh14K0P_Z-Sckn9Va20D8-bgrctIKGA&zhida_source=entity)中可能包含错误或打字错误，因为有好几种表述一个名字的方法，包括头衔，圆括号和再另一个或短名中使用的引号。



- 哪些特征有空值？
  这些需要清洗。
  Cabin > Age > Embarked features contain a number of null values in that order for the training dataset.
  Cabin > Age are incomplete in case of test dataset.



- 这些特征的数据类型都是什么？
  帮助我们完成清洗的目标。
  7个变量是整数或浮点数，在测试集中是六个。
  5个变量是字符串。

```python
#查看是否存在缺失值以及数据类型
train_data.info()
print("-"*40)
test_data.info()
---------------------
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
PassengerId    891 non-null int64
Survived       891 non-null int64
Pclass         891 non-null int64
Name           891 non-null object
Sex            891 non-null object
Age            714 non-null float64
SibSp          891 non-null int64
Parch          891 non-null int64
Ticket         891 non-null object
Fare           891 non-null float64
Cabin          204 non-null object
Embarked       889 non-null object
dtypes: float64(2), int64(5), object(5)
memory usage: 83.6+ KB
----------------------------------------
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 418 entries, 0 to 417
Data columns (total 11 columns):
PassengerId    418 non-null int64
Pclass         418 non-null int64
Name           418 non-null object
Sex            418 non-null object
Age            332 non-null float64
SibSp          418 non-null int64
Parch          418 non-null int64
Ticket         418 non-null object
Fare           417 non-null float64
Cabin          91 non-null object
Embarked       418 non-null object
dtypes: float64(2), int64(4), object(5)
memory usage: 36.0+ KB
```

- 样本中数值型变量的分布是什么？
  这帮助我们在早期决定这一数据集是否对现实问题有代表性意义。
  - 所有的样本有891个，是Titanic实际登船乘客的40%。
  - 生还变量是无序变量，取值为0或1.
  - 样本中大约有38%的乘客获救，在实际中是32%。
  - 大多数乘客（>75%）没有带父母或小孩。
  - 将近30%的乘客在船上有兄弟姐妹或配偶。
  - 在少数乘客中（<1%）票价显著地变化了，高达$512.
  - 有少数年龄在65-80岁的老年人（<1%）

```python
#描述性统计
train_data.describe()
# Review survived rate using `percentiles=[.61, .62]` knowing our problem description mentions 38% survival rate.
# Review Parch distribution using `percentiles=[.75, .8]`
# SibSp distribution `[.68, .69]`
# Age and Fare `[.1, .2, .3, .4, .5, .6, .7, .8, .9, .99]`

---------------------------------
PassengerId	Survived	Pclass	Age	SibSp	Parch	Fare
count	891.000000	891.000000	891.000000	714.000000	891.000000	891.000000	891.000000
mean	446.000000	0.383838	2.308642	29.699118	0.523008	0.381594	32.204208
std	257.353842	0.486592	0.836071	14.526497	1.102743	0.806057	49.693429
min	1.000000	0.000000	1.000000	0.420000	0.000000	0.000000	0.000000
25%	223.500000	0.000000	2.000000	20.125000	0.000000	0.000000	7.910400
50%	446.000000	0.000000	3.000000	28.000000	0.000000	0.000000	14.454200
75%	668.500000	1.000000	3.000000	38.000000	1.000000	0.000000	31.000000
max	891.000000	1.000000	3.000000	80.000000	8.000000	6.000000	512.329200
```

- 无序变量的分布是什么？
  - 名字在数据集中是唯一的。
  - Sex 有两个可能的取值，65%男性
  - Cabin 在样本中有许多取值。一些乘客共享一个Cabin
  - Embarked 有三个取值可能。S港口被大多数旅客使用。
  - Ticket 有很高的取值重复率（22%）（unique=681）。

```python
#描述无序变量的特征
train_data.describe(include=['O'])
----------------------------
       Name	Sex	Ticket	Cabin	Embarked
count	891	891	891	204	889
unique	891	2	681	147	3
top	Sunderland, Mr. Victor Francis	male	347082	G6	S
freq	1	577	7	4	644
```

## 5.2 基于数据分析的假定

我们根据已完成的[数据分析](https://zhida.zhihu.com/search?content_id=112784529&content_type=Article&match_order=2&q=数据分析&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3OTE5NjIzNjMsInEiOiLmlbDmja7liIbmnpAiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxMTI3ODQ1MjksImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MiwiemRfdG9rZW4iOm51bGx9.YKBs9MZIxJedEr1Jk_ipzS8lQRWLwjgwJ9-83WVUk5Y&zhida_source=entity)工作提出了以下假定。在进一步行动之间我们可能需要证实这些假定。

- Correlating 相关：我们想知道每个变量与“生还”之间的相关性有多大。我们想在项目早期完成这个工作，并且在后期的建模工作中匹配这些相关性。



- Completing 处理缺失值：

1. 因为Age与Survival显著相关，我们想处理它的缺失值。
2. 同样的写需要处理Embarked和其他变量



- Correcting 校正：

1. Ticket或许可以从分析中去除，因为它包含了大量重复的数据，并且Ticket和survival之间没有关系。
2. Cabin 可以去除，因为它在训练集和测试集中都高度缺失。
3. Passengerld 可以去除，因为它对预测survival没有帮助。
4. Name 变量不规范，可能不能直接用来做预测，因此可以去除



- Creating 新建：

1. 基于Parch和SibSp建立一个新变量Family，获得在船上的家庭成员总数。
2. 提取Name中的Title作为一个新变量。
3. 建立一个Age区间，变量将一个连续的数值型变量转变为定序变量。
4. 建立一个Fare[等级变量](https://zhida.zhihu.com/search?content_id=112784529&content_type=Article&match_order=1&q=等级变量&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3OTE5NjIzNjMsInEiOiLnrYnnuqflj5jph48iLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxMTI3ODQ1MjksImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.f1YveSqavuYkg_w2mQSqsLBj3tqiCVprg-27UDvcEyk&zhida_source=entity)。



- Classifying 分类： 我们需要基于之前提出的问题描述提出我们的假设。

1. Woman（Sex=female）更可能获救。
2. Children（Age<？）更可能获救。
3. 头等舱旅客（Pclass=1）更可能获救。

### 5.2.1变量数据透视分析

为了证实我们的一些观测和假设，我们可以快速透视数据来分析变量间的相关性。


在这一步我们只能分析没有空值的变量。这一分析只对[无序数据](https://zhida.zhihu.com/search?content_id=112784529&content_type=Article&match_order=1&q=无序数据&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3OTE5NjIzNjMsInEiOiLml6Dluo_mlbDmja4iLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxMTI3ODQ1MjksImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.P-rl0J7kNAjlLB1JoCpa_8NW5pwUg5D1YwrN2BbDnv0&zhida_source=entity)（Sex），定序变量（Pclass）或离散型（SibSp，Parch）。

- Pclass 我们观测到 Pclass=1 与Survived之间有显著的相关性（>0.5），我们决定将这一变量加入到模型中。（Classifying#3）
- Sex 我们确认了在问题定义中的观测，Sex=female 有着高达74%的生还率。（Classifying#1）
- SibSp and Parch 这两个变量没有相关性。最好从这两个变量中导出一个或创建一个变量。（Creating#1）

```python
# 查看 Pclass与Survived 的相关性
train_data[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived')
--------------------------------
      Pclass	Survived
2	3	0.242363
1	2	0.472826
0	1	0.629630  


train_data[['Sex','Survived']].groupby(['Sex'],as_index=False).mean().sort_values(by='Survived')
#女性获救概率高
------------------------------
       Sex	Survived
1	male	0.188908
0	female	0.742038  


train_data[['Embarked','Survived']].groupby(['Embarked'],as_index=False).mean().sort_values(by='Survived')
#三个港口之间没什么区别
------------------------
   Embarked	Survived
2	S	0.336957
1	Q	0.389610
0	C	0.553571
train_data[['SibSp','Survived']].groupby(['SibSp'],as_index=False).mean().sort_values(by='Survived',ascending=False)
-----------------------------------
     SibSp	Survived
1	1	0.535885
2	2	0.464286
0	0	0.345395
3	3	0.250000
4	4	0.166667
5	5	0.000000
6	8	0.000000
train_data[['Parch','Survived']].groupby(['Parch'],as_index=False).mean().sort_values(by='Survived',ascending=False)
------------------------
     Parch	Survived
3	3	0.600000
1	1	0.550847
2	2	0.500000
0	0	0.343658
5	5	0.200000
4	4	0.000000
6	6	0.000000
```

### 5.2.2 [数据可视化](https://zhida.zhihu.com/search?content_id=112784529&content_type=Article&match_order=1&q=数据可视化&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3OTE5NjIzNjMsInEiOiLmlbDmja7lj6_op4bljJYiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxMTI3ODQ1MjksImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.6xRPwF9JsNWxMGInoDTrU-SWuTNmFU7MH0kkTDxDoMU&zhida_source=entity)进行分析

现在我们要继续用[数据可视化分析](https://zhida.zhihu.com/search?content_id=112784529&content_type=Article&match_order=1&q=数据可视化分析&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3OTE5NjIzNjMsInEiOiLmlbDmja7lj6_op4bljJbliIbmnpAiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxMTI3ODQ1MjksImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.iJBhTnGHjADHbA0cfA-5EE6M6tpXNnd57uu9_29UbVk&zhida_source=entity)来证实我们的假设。

### 5.2.2.1关联数值型变量

让我们从理解数值型变量和Survived的关系开始。


直方图在分析连续型数值变量中很有用（例如Age），区间或阶段对于理解有价值的模式很有用。[直方图](https://zhida.zhihu.com/search?content_id=112784529&content_type=Article&match_order=2&q=直方图&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3OTE5NjIzNjMsInEiOiLnm7Tmlrnlm74iLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxMTI3ODQ1MjksImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MiwiemRfdG9rZW4iOm51bGx9.zWRWRmsjl-zTzXzUhOZRr9Gcgym84BofSuKgMj_ekgE&zhida_source=entity)课以用自动生成的区间或均等的区间来建立样本的分布。这帮助我们回答有关特定区间的问题。（婴儿是否有更好的生还率）


直方图的x轴代表样本或乘客数量。

- 观察
  - 婴幼儿（Age<=4）有更高的生还率
  - 最老的乘客生还了（Age=80）
  - 大量15-25岁的乘客没有生还
  - 绝大多数乘客年龄在15-35岁。



- 决定 这一分析证实了我们对于后续工作流程的假定。
  - 我们应该将Age考虑在假设内（Classifying#2）
  - 填补空缺的Age变量（Completing#1）
  - 应该将Age分为区间（Creating#3）

```python
#使用train_data 初始化对象
g = sns.FacetGrid(train_data,col='Survived')
g.map(plt.hist,'Age',bins=20)
------------------------------------------
<seaborn.axisgrid.FacetGrid at 0x22bea157438>
```

![img](https://pica.zhimg.com/v2-b2e69ba5cb813642491d92cd1faebe0a_1440w.jpg)



```python
g = sns.FacetGrid(train_data,col='Survived')
g.map(plt.hist,'Fare',bins=10)
--------------------------------
<seaborn.axisgrid.FacetGrid at 0x22bea93d710>
```

![img](https://pic3.zhimg.com/v2-2cf2f437f6acf2debac03e1ffd175e30_1440w.jpg)



```python
g = sns.FacetGrid(train_data,col='Survived')
g.map(plt.hist,'Age',bins=10)
---------------------------
<seaborn.axisgrid.FacetGrid at 0x22bea9678d0>
```

![img](https://pic4.zhimg.com/v2-1c53e9684abe58002a1fc6433cadd66d_1440w.jpg)



```python
plt.hist(train_data['Age'][train_data.Age >= 0 ],bins='auto')
------------------------
(array([40., 14., 15., 31., 79., 98., 85., 84., 73., 45., 35., 35., 29.,
        16., 13., 11.,  4.,  5.,  1.,  1.]),
 array([ 0.42 ,  4.399,  8.378, 12.357, 16.336, 20.315, 24.294, 28.273,
        32.252, 36.231, 40.21 , 44.189, 4.168, 52.147, 56.126, 60.105,
        64.084, 68.063, 72.042, 76.021, 80.   ]),
 <a list of 20 Patch objects>)
```

![img](https://pic3.zhimg.com/v2-942db0304d75933c75d47d0bbcc1a57c_1440w.jpg)



### 5.2.2.2 关联数值型和定序变量

我们可以为了识别相关性用用简单的图像结合多个变量。

- 观察
  - Pclass=3有最多的乘客，然而尚存率不是最高，证实了我们Classifying assumption#2
  - Pclass=2和Pclass=3的绝大多数婴幼儿生存下来，更加肯定了我们的Classifying assumption#2
  - Pclass=1的大多数乘客生还，证实了Classifying assumption#3
  - Pclass在不同年龄的乘客中分布不同



- 决定
  - 将Pclass考虑在建模内

```python
grid = sns.FacetGrid(train_data,hue="Survived",col='Pclass')
grid.map(plt.hist,'Age',alpha=.5,bins=20)
grid.add_legend()
------------------------
<seaborn.axisgrid.FacetGrid at 0x22beabaf438>
```

![img](https://picx.zhimg.com/v2-2ee47923ff4657352d5c23443a30a7f9_1440w.jpg)



### 5.2.2.3关联[定类变量](https://zhida.zhihu.com/search?content_id=112784529&content_type=Article&match_order=1&q=定类变量&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3OTE5NjIzNjMsInEiOiLlrprnsbvlj5jph48iLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxMTI3ODQ1MjksImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.XSPBr7e55ReaJJEmhyChQczY_cbTrE8M_otCI1FBF8Q&zhida_source=entity)

- 观察
  - 女性乘客比男性乘客存活率更高，证实了 classifying（#1）
  - Embarked=C处有异常，男性生还率更高。这可能是Pclass和Embarked之间的相关性，进而是Pclass和Survived的，不是Embarked和Survived之间必要的直接相关。
  - Pclass=3时，相较于来自C和Q港的Pclass=2，男性生还率更高 Completing（#2）
  - Pclass=3，并且乘客为男性时，登船的港口之间的生还率分布不同。Correlating（#1）



- 决定
  - 将性别特征加入模型
  - 在模型训练中加入和完善Embarked变量

```python
grid = sns.FacetGrid(train_data,col='Embarked')
grid.map(sns.pointplot,'Pclass','Survived','Sex')
grid.add_legend()
--------------------------------
D:\software\anaconda\lib\site-packages\seaborn\axisgrid.py:715: UserWarning: Using the pointplot function without specifying `order` is likely to produce an incorrect plot.
  warnings.warn(warning)
D:\software\anaconda\lib\site-packages\seaborn\axisgrid.py:720: UserWarning: Using the pointplot function without specifying `hue_order` is likely to produce an incorrect plot.
  warnings.warn(warning)





<seaborn.axisgrid.FacetGrid at 0x22beade3908>
```

![img](https://picx.zhimg.com/v2-41d5540e3fef66cde9e030118a35c4bf_1440w.jpg)



### 5.2.2.4关联 无序变量和[数值变量](https://zhida.zhihu.com/search?content_id=112784529&content_type=Article&match_order=4&q=数值变量&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3OTE5NjIzNjMsInEiOiLmlbDlgLzlj5jph48iLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxMTI3ODQ1MjksImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6NCwiemRfdG9rZW4iOm51bGx9.epnaG4vHWYQeiWyIRNHkhzo1--RhCel6seRISGfC1JY&zhida_source=entity)

我们也想要关联数值变量和无序变量。我们可以考虑关联Embarked（[categorical](https://zhida.zhihu.com/search?content_id=112784529&content_type=Article&match_order=1&q=categorical&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3OTE5NjIzNjMsInEiOiJjYXRlZ29yaWNhbCIsInpoaWRhX3NvdXJjZSI6ImVudGl0eSIsImNvbnRlbnRfaWQiOjExMjc4NDUyOSwiY29udGVudF90eXBlIjoiQXJ0aWNsZSIsIm1hdGNoX29yZGVyIjoxLCJ6ZF90b2tlbiI6bnVsbH0._3WL4xHRIy-zlRLbmso9OFgDNm9TL8CjNQ11g4w2YAQ&zhida_source=entity)，non-numeric），Sex（categorical，non-numeric），Fare（numeric，continuous）和Survived（categorical numeric）

- 观察
  - 船票更贵的乘客生还率数据更好，证实了我们creating（#4）费用等级的假设。
  - 出发港和生还率相关，证实了corrtlating（#1）和completing（#2）



- 决定
  - 考虑将费用变量划分等级

```python
grid = sns.FacetGrid(train_data,col = 'Embarked',hue='Survived', palette={0: 'k', 1: 'w'})
grid.map(sns.barplot,'Sex','Fare',ci=None,alpha=.5)
grid.add_legend()
---------------------------
D:\software\anaconda\lib\site-packages\seaborn\axisgrid.py:715: UserWarning: Using the barplot function without specifying `order` is likely to produce an incorrect plot.
  warnings.warn(warning)





<seaborn.axisgrid.FacetGrid at 0x22beaf31e48>
```

![img](https://picx.zhimg.com/v2-b254978f09807045ca821bf257132369_1440w.jpg)



## 6 清洗数据 Wrangle data

我们已经为数据集和解决问题的需求收集了好几条假设和决定。目前为止，我们没有改变任何一个变量或值，就完成了这些。让我们现在实施决定和假设，为了完成correcting，creating和completing的目标

## 6.1 通过删除变量来修正

这是好的开端。通过删除变量我们处理更少的数据点，提高notebook的速度并且简化分析。

基于我们的假设和决定，我们想要删除Cabin（correcting#2）和Ticket（correcting#1）变量。


记住在适用情况下，训练集和测试集操作相同，目的是保持一致性。

```python
print('Before',train_data.shape,test_data.shape)

train_data = train_data.drop(['Cabin','Ticket'],axis =1)
test_data = test_data.drop(['Cabin','Ticket'],axis =1)

print('After',train_data.shape,test_data.shape)
--------------------------
Before (891, 12) (418, 11)
After (891, 10) (418, 9)
```

## 6.2 从现有数据中创造新变量

在删除Name和PassengerID变量之前，我们想要分析Name变量是否可以分离出头衔（title），然后测试Title和Survival之间的相关性。


在接下来的代码中我们用正则表达式提取头衔。 (\w+.)会匹配Name变量中第一个以点结束的单词。expand=False 返回一个DataFrame

- 观察 当我们使用Title，Age，Survived 画图时，我们发现了：
  - 大多数Title精确地处于一个年龄段内。例如：Master
  - 不同Title-Age组中，Survival差距不大。
  - 有些title绝大多数都生还了（Mme，Lady，Sir），有些绝大多数没有（Don，Rev，Jonkheer）



- 决定
  - 我们决定在模型训练中保留新的Title变量

```python
# str.extract 函数 用来处理字符型数据，可以用正则表达式从字符中抽取符合条件的数据 expend :布尔型，是否返回dataframe
# crosstab 函数 返回一个交叉表，交叉表是统计分组频率的特殊透视表 前面是列后面是行
combine = [train_data,test_data]
for dataset in combine:
    dataset['Title'] =dataset.Name.str.extract('([A-Za-z]+)\.',expand =False)
pd.crosstab(train_data['Title'],train_data['Sex'])
------------------------------------------------
Sex	female	male
Title		
Capt	0	1
Col	0	2
Countess	1	0
Don	0	1
Dr	1	6
Jonkheer	0	1
Lady	1	0
Major	0	2
Master	0	40
Miss	182	0
Mlle	2	0
Mme	1	0
Mr	0	517
Mrs	125	0
Ms	1	0
Rev	0	6
Sir	0	1
```



我们可以把一些频率较少的title 转变为更常见的名字或将它们分类为Rare

```python
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'],'Rare')
    dataset['Title'] = dataset['Title'].replace(['Mlle','Ms'],'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme','Mrs')

train_data[['Title','Survived']].groupby(['Title'],as_index=False).mean()
----------------------------------------------------
    Title	Survived
0	Master	0.575000
1	Miss	0.702703
2	Mr	0.156673
3	Mrs	0.793651
4	Rare	0.347826
```

我们将无序的title 转变为 序数变量

```python
title_mapping ={"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping) #转变数据格式
    dataset['Title'] = dataset['Title'].fillna(0) #把没有title的定为0
train_data.head()
---------------------------------------
PassengerId	Survived	Pclass	Name	Sex	Age	SibSp	Parch	Fare	Embarked	Title
0	1	0	3	Braund, Mr. Owen Harris	male	22.0	1	0	7.2500	S	1
1	2	1	1	Cumings, Mrs. John Bradley (Florence Briggs Th...	female	38.0	1	0	71.2833	C	3
2	3	1	3	Heikkinen, Miss. Laina	female	26.0	0	0	7.9250	S	2
3	4	1	1	Futrelle, Mrs. Jacques Heath (Lily May Peel)	female	35.0	1	0	53.1000	S	3
4	5	0	3	Allen, Mr. William Henry	male	35.0	0	0	8.0500	S	1
```

现在我们可以从训练集和测试集中安全地删除Name变量，我们也不需要PassengerID

```python
train_data = train_data.drop(['Name','PassengerId'],axis = 1)
test_data = test_data.drop(['Name'],axis = 1)
combine = [train_data,test_data]
train_data.shape , test_data.shape
--------------------------------
((891, 9), (418, 9))
```

## 6.3 转换一个无序变量

现在我们可以将包含字符串的变量转换为数值型。这是大多数算法要求的。这样做会帮助我们完成完善变量的目标。


让我们从Sex开始，将Sex变量转换成Gender变量。female=1，male=0

```python
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map({'female':1,'male':0})
train_data.head()
---------------------------------------
   Survived	Pclass	Sex	Age	SibSp	Parch	Fare	Embarked	Title
0	0	3	0	22.0	1	0	7.2500	S	1
1	1	1	1	38.0	1	0	71.2833	C	3
2	1	3	1	26.0	0	0	7.9250	S	2
3	1	1	1	35.0	1	0	53.1000	S	3
4	0	3	0	35.0	0	0	8.0500	S	1
```



## 填补连续型数值变量的缺失值

现在我们应该开始估计和填补缺失值了。Age是第一个。


我们可以考虑三个为[连续型数值变量](https://zhida.zhihu.com/search?content_id=112784529&content_type=Article&match_order=3&q=连续型数值变量&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3OTE5NjIzNjMsInEiOiLov57nu63lnovmlbDlgLzlj5jph48iLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxMTI3ODQ1MjksImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MywiemRfdG9rZW4iOm51bGx9.jDrozqhMFOMW_CLy6BNwMuPT5oEblJdfr7G8ftbGpOc&zhida_source=entity)填补缺失值的方法。



1. 一个简单的方法是，从平均值和[标准差](https://zhida.zhihu.com/search?content_id=112784529&content_type=Article&match_order=1&q=标准差&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3OTE5NjIzNjMsInEiOiLmoIflh4blt64iLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxMTI3ODQ1MjksImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.ZQxMRes9pyz1qb92flEZTy7iC_hY-HqcBnvL_oeoPuk&zhida_source=entity)之间生成随机数字。
2. 一个猜测缺失值更精确的方法是使用其他相关变量。在我们的案例中，考虑Age，Gender和Pclass之间的相关性。使用Pclass 和Gender 不同组合中的Age中位数来猜测Age的值。即 Pclass =1 和Gender =0 ， Pclass =1 和Gender =1等等的Age中位数
3. 结合 方法1和2。即不使用[中位数](https://zhida.zhihu.com/search?content_id=112784529&content_type=Article&match_order=3&q=中位数&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3OTE5NjIzNjMsInEiOiLkuK3kvY3mlbAiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxMTI3ODQ1MjksImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MywiemRfdG9rZW4iOm51bGx9._iiR7lSkjtO9lj8rhMFnaYCG2RshYsoNLX3DVT_zWQo&zhida_source=entity)作为填充值，而是使用基于不同Pclass和Gender的平均数和标准差来生成随机数字。 方法1和3会给我们的模型引入[随机扰动](https://zhida.zhihu.com/search?content_id=112784529&content_type=Article&match_order=1&q=随机扰动&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3OTE5NjIzNjMsInEiOiLpmo_mnLrmibDliqgiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxMTI3ODQ1MjksImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.jjQw-mRXzkLkbuAT_HSyxKWd6C-zhabd8xrP1tpx6JI&zhida_source=entity)。多次执行的结果可能不同。我们更偏好方法2。

```python
# 观察Pclass Gender 和 Age 的关系
grid = sns.FacetGrid(train_data,row ='Pclass',col = 'Sex',height=2.2,aspect=1.6)
grid.map(plt.hist,'Age',alpha=.5,bins=20)
grid.add_legend()
------------------------
<seaborn.axisgrid.FacetGrid at 0x22bec052630>
```

![img](https://pic2.zhimg.com/v2-3fead2180bec45e29a6005e72af7251d_1440w.jpg)



让我们从准备一个放置基于不同Pclass 和 Gender组合的Age猜测值的空array

```python
guess_ages = np.zeros((2,3))
guess_ages
----------------------
array([[0., 0., 0.],
       [0., 0., 0.]])
```

现在我们对Sex（0,1）和Pclass（1，2，3）进行迭代来计算6个组合的Age猜测值。

```python
for dataset in combine:
    for i in range(0,2):
        for j in range(1,4):
            guess_df = dataset[( dataset.Sex == i ) & ( dataset.Pclass ==j )].Age.dropna()
            age_guess = guess_df.median()
            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean +age_std)

            guess_ages[i,j-1] = age_guess

            # 将随机的age小数转化到最近的.5
            # guess_ages[i,j-1] = int(age_guess/0.5 +0.5)*0.5
    for i in range(0,2):
        for j in range(1,4):
            dataset.loc[(dataset.Age.isnull())&(dataset.Sex==i)&(dataset.Pclass ==j),'Age'] = guess_ages[i,j-1]
    dataset.Age = dataset.Age.astype('int')
train_data.head()
-------------------------------------
    Survived	Pclass	Sex	Age	SibSp	Parch	Fare	Embarked	Title
0	0	3	0	22	1	0	7.2500	S	1
1	1	1	1	38	1	0	71.2833	C	3
2	1	3	1	26	0	0	7.9250	S	2
3	1	1	1	35	1	0	53.1000	S	3
4	0	3	0	35	0	0	8.0500	S	1
```

让我们新建Age区间并且找出和Survived的相关性

```python
train_data['Ageband'] = pd.cut(train_data['Age'],5)

train_data[['Ageband','Survived']].groupby(['Ageband'],as_index = False).mean().sort_values(by='Ageband',ascending = True)
----------------------------------------------
         Ageband	Survived
0	(-0.08, 16.0]	0.550000
1	(16.0, 32.0]	0.337374
2	(32.0, 48.0]	0.412037
3	(48.0, 64.0]	0.434783
4	(64.0, 80.0]	0.090909
```



让我们以基于这些区间的顺序来替换Age

```python
for dataset in combine:
    dataset.loc[dataset.Age <=16,'Age'] = 0
    dataset.loc[(dataset.Age >16) & (dataset.Age <= 32),'Age'] = 1
    dataset.loc[(dataset.Age >32) & (dataset.Age <= 48),'Age'] = 2
    dataset.loc[(dataset.Age >48) & (dataset.Age <= 64),'Age'] = 3
    dataset.loc[dataset.Age >64,'Age'] = 4

train_data.head()
------------------------------------
Survived	Pclass	Sex	Age	SibSp	Parch	Fare	Embarked	Title	Ageband
0	0	3	0	1	1	0	7.2500	S	1	(16.0, 32.0]
1	1	1	1	2	1	0	71.2833	C	3	(32.0, 48.0]
2	1	3	1	1	0	0	7.9250	S	2	(16.0, 32.0]
3	1	1	1	2	1	0	53.1000	S	3	(32.0, 48.0]
4	0	3	0	2	0	0	8.0500	S	1	(32.0, 48.0]      
```

我们现在可以移除Age区间变量

```python
train_data = train_data.drop(['Ageband'],axis = 1)
combine = [train_data,test_data]
train_data.head()
------------------------------------------
Survived	Pclass	Sex	Age	SibSp	Parch	Fare	Embarked	Title
0	0	3	0	1	1	0	7.2500	S	1
1	1	1	1	2	1	0	71.2833	C	3
2	1	3	1	1	0	0	7.9250	S	2
3	1	1	1	2	1	0	53.1000	S	3
4	0	3	0	2	0	0	8.0500	S	1
```

## 6.5 通过结合现有变量来创造新变量

我们可以创造一个新变量——FamilySize，结合了Parch和SibSp。这回帮助我们将Parch和SibSp变量从数据中移除。

```python
for dataset in combine:
    dataset['FamilySize'] = dataset['Parch']+dataset['SibSp']+1

train_data[['FamilySize','Survived']].groupby(['FamilySize'],as_index = False).mean().sort_values(by = 'Survived',ascending = False)
---------------------------------------
FamilySize	Survived
3	4	0.724138
2	3	0.578431
1	2	0.552795
6	7	0.333333
0	1	0.303538
4	5	0.200000
5	6	0.136364
7	8	0.000000
8	11	0.000000
```

我们可以创造另一个变量IsAlone

```python
for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset.FamilySize == 1,'IsAlone'] =1

train_data[['IsAlone','Survived']].groupby(['IsAlone'],as_index=False).mean()
---------------------------------------
IsAlone	Survived
0	0	0.505650
1	1	0.303538 
```

让我们删除Parch，SibSp和FamilySize变量，因为更偏好于IsAlone

```python
train_data = train_data.drop(['Parch','SibSp','FamilySize'],axis = 1)
test_data = test_data.drop(['Parch','SibSp','FamilySize'],axis = 1)
combine =[train_data,test_data]

train_data.head()
-----------------------------------
	Survived	Pclass	Sex	Age	Fare	Embarked	Title	IsAlone
0	0	3	0	1	7.2500	S	1	0
1	1	1	1	2	71.2833	C	3	0
2	1	3	1	1	7.9250	S	2	1
3	1	1	1	2	53.1000	S	3	0
4	0	3	0	2	8.0500	S	1	1   
```

我们还可以创造一个结合Pclass和Age的变量。

```python
for dataset in combine:
    dataset['Age*Pclass'] = dataset.Age * dataset.Pclass

train_data[['Age*Pclass','Age','Pclass']].head(10)
------------------------------------
Age*Pclass	Age	Pclass
0	3	1	3
1	2	2	1
2	3	1	3
3	2	2	1
4	6	2	3
5	3	1	3
6	3	3	1
7	0	0	3
8	3	1	3
9	0	0	2   
```

## 6.6 填补无序变量

基于登船的港口，Embarked变量有三个可能的取值S,Q,C。我们的训练集有两个缺失值。我们简单地以最常见的值填充它。

```python
# mode function 返回众数
freq_port = train_data.Embarked.dropna().mode()[0]
freq_port
'S'


for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

train_data[['Embarked','Survived']].groupby(['Embarked'],as_index =False).mean()
------------------------
Embarked	Survived
0	C	0.553571
1	Q	0.389610
2	S	0.339009  
```

## 6.7 将无序变量转化为数字型变量

我们现在可以将EmbarkedFill变量转化为一个新的数字型变量Port

```python
for dataset in combine:
    dataset['Embarked'] =dataset['Embarked'].map({'S':0,'C':1,'Q':2}).astype(int)

train_data.head()
----------------------------------
Survived	Pclass	Sex	Age	Fare	Embarked	Title	IsAlone	Age*Pclass
0	0	3	0	1	7.2500	0	1	0	3
1	1	1	1	2	71.2833	1	3	0	2
2	1	3	1	1	7.9250	0	2	1	3
3	1	1	1	2	53.1000	0	3	0	2
4	0	3	0	2	8.0500	0	1	1	6
```

## 6.8 迅速填补和转换一个[数字型变量](https://zhida.zhihu.com/search?content_id=112784529&content_type=Article&match_order=3&q=数字型变量&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3OTE5NjIzNjMsInEiOiLmlbDlrZflnovlj5jph48iLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxMTI3ODQ1MjksImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MywiemRfdG9rZW4iOm51bGx9.7z7UKb8KEjujga4w75QCH1zel5xj9shp7RvMari59Pg&zhida_source=entity)

我们现在可以使用众数填补数据集中缺失的Fare变量，使用一行代码就可以完成。

因为只有一个缺失值，我们不创造一个新的中间变量或进一步分析相关性来填补缺失值。填补是因为算法中不能有空值的要求。

我们也想将Fare四舍五入到两位小数，因为它代表现金。

```python
test_data['Fare']=test_data['Fare'].fillna(test_data['Fare'].dropna().median())
test_data.head()
test_data.info()
------------------
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 418 entries, 0 to 417
Data columns (total 9 columns):
PassengerId    418 non-null int64
Pclass         418 non-null int64
Sex            418 non-null int64
Age            418 non-null int32
Fare           418 non-null float64
Embarked       418 non-null int32
Title          418 non-null int64
IsAlone        418 non-null int64
Age*Pclass     418 non-null int64
dtypes: float64(1), int32(2), int64(6)
memory usage: 26.2 KB
```

我们现在可以建立Fare区间。

```python
# qcut 根据值的频率确定区间
train_data['Fareband'] = pd.qcut(train_data['Fare'],4)
train_data[['Fareband','Survived']].groupby(['Fareband'],as_index =False).mean().sort_values('Fareband',ascending=True)
---------------------------------------------------------------
Fareband	Survived
0	(-0.001, 7.91]	0.197309
1	(7.91, 14.454]	0.303571
2	(14.454, 31.0]	0.454955
3	(31.0, 512.329]	0.581081
```

基于Fareband 将Fare变量转换为定序变量

```python
for dataset in combine:
    dataset.loc[dataset.Fare <= 7.91,'Fare'] = 0
    dataset.loc[(dataset.Fare>7.91)&(dataset.Fare <=14.454),'Fare'] = 1
    dataset.loc[(dataset.Fare>14.454)&(dataset.Fare <=31),'Fare'] = 2
    dataset.loc[dataset.Fare>31,'Fare'] = 3
    dataset.Fare = dataset.Fare.astype(int)

train_data = train_data.drop(['Fareband'],axis =1 )
combine = [train_data,test_data]

train_data.head()
---------------------------------------------------------
   Survived	Pclass	Sex	Age	Fare	Embarked	Title	IsAlone	Age*Pclass
0	0	3	0	1	0	0	1	0	3
1	1	1	1	2	3	1	3	0	2
2	1	3	1	1	1	0	2	1	3
3	1	1	1	2	3	0	3	0	2
4	0	3	0	2	1	0	1	1	6
```

以及测试集

```python
test_data.head()
------------------------------------------
PassengerId	Pclass	Sex	Age	Fare	Embarked	Title	IsAlone	Age*Pclass
0	892	3	0	2	0	2	1	1	6
1	893	3	1	2	0	0	3	0	6
2	894	2	0	3	1	2	1	1	6
3	895	3	0	1	1	0	1	1	3
4	896	3	1	1	1	0	3	0	3
```



## 7 [建模](https://zhida.zhihu.com/search?content_id=112784529&content_type=Article&match_order=5&q=建模&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3OTE5NjIzNjMsInEiOiLlu7rmqKEiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxMTI3ODQ1MjksImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6NSwiemRfdG9rZW4iOm51bGx9.RywlMaquE8hut9SrUZiyrGf0qDqNbY9Jj88FGzMRuxQ&zhida_source=entity)，预测以及解决

现在我们准备好训练一个模型以及预测结果。有60+种预测建模算法可以选择。我们必须理解问题的类型和结果要求来缩小我们可以估计的模型的范围。 我们的问题是一个分类和[回归问题](https://zhida.zhihu.com/search?content_id=112784529&content_type=Article&match_order=1&q=回归问题&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3OTE5NjIzNjMsInEiOiLlm57lvZLpl67popgiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxMTI3ODQ1MjksImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.g98w-BOLpa7wgQMwwJsV0hdx2MVJEJDHEBc9u31mgnw&zhida_source=entity)。我们想要了解输出（Survived与否）以及其他变量（Gender，Age，Port）等的关系。我们也在运用机器学习的一种——有监督 学习，因为我们使用一个给定数据集来训练模型。基于这两个条件——有监督学习+分类和回归，我们可以将我们的选择范围缩小，包括：

- Logistic Regression
- KNN or k-Nearest Neighbors
- Support Vector Machines
- Naive Bayes Classifier
- Decision Tree
- Random Forrest
- Perceptron
- Artificial neural network
- RVM or Relevance Vector Machine

```python
X_train = train_data.drop('Survived',axis = 1)
Y_train = train_data.Survived
X_test = test_data.drop('PassengerId',axis =1)
X_train.shape, Y_train.shape,X_test.shape
---------------------------------------
((891, 8), (891,), (418, 8))
```

在工作流运行早期，Logistic Regression是一个有用的模型。Logistic [regression](https://zhida.zhihu.com/search?content_id=112784529&content_type=Article&match_order=1&q=regression&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3OTE5NjIzNjMsInEiOiJyZWdyZXNzaW9uIiwiemhpZGFfc291cmNlIjoiZW50aXR5IiwiY29udGVudF9pZCI6MTEyNzg0NTI5LCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.AMmY3uSWqQ1BRawsDYtKoMTNzl0fDNlRMC0hS8tm3sk&zhida_source=entity) 通过运用logistic function估计概率来衡量无序因变量和一个或多个[自变量](https://zhida.zhihu.com/search?content_id=112784529&content_type=Article&match_order=1&q=自变量&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3OTE5NjIzNjMsInEiOiLoh6rlj5jph48iLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxMTI3ODQ1MjksImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.soPAnmp1jlQm2_bKb9CjZ5VX-Ehrv1W6w0gYK-H1knI&zhida_source=entity)之间的关系。

计算基于我们数据集所建立模型的精确度

```python
# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train,Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train,Y_train)*100,2)
acc_log
-------------------------------------
D:\software\anaconda\lib\site-packages\sklearn\linear_model\logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
  FutureWarning)





81.26
```

我们可以使用Logistic Regression来证实我们对创造变量和填补缺失值的假设和决定。这可以通过计算在决定函数中变量的系数来完成。

正系数使生还的[log-odds](https://zhida.zhihu.com/search?content_id=112784529&content_type=Article&match_order=1&q=log-odds&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3OTE5NjIzNjMsInEiOiJsb2ctb2RkcyIsInpoaWRhX3NvdXJjZSI6ImVudGl0eSIsImNvbnRlbnRfaWQiOjExMjc4NDUyOSwiY29udGVudF90eXBlIjoiQXJ0aWNsZSIsIm1hdGNoX29yZGVyIjoxLCJ6ZF90b2tlbiI6bnVsbH0.hRVbQIMcaLDjHWqqQ8NNIABfHE8Ayi2-_qK49buzC7o&zhida_source=entity)（logistic Regression的link function相关）提高（从而提高概率），[负系数](https://zhida.zhihu.com/search?content_id=112784529&content_type=Article&match_order=1&q=负系数&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3OTE5NjIzNjMsInEiOiLotJ_ns7vmlbAiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxMTI3ODQ1MjksImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.FJ4Aj6cH-hjO1iDOlM_fCl5oZIdfKMZqvueDjlLejq4&zhida_source=entity)降低生还的log-odds（从而降低概率）。

- Sex是最高的[正系数](https://zhida.zhihu.com/search?content_id=112784529&content_type=Article&match_order=2&q=正系数&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3OTE5NjIzNjMsInEiOiLmraPns7vmlbAiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxMTI3ODQ1MjksImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MiwiemRfdG9rZW4iOm51bGx9._eDzc0gsMLpGbhdC3pJXvs3LRjx9QlJjho4A9dUGegQ&zhida_source=entity)，说明Sex值越大（male：0 到 female：1），Survived =1的概率越大。
- 相反的，当Pclass增加时，Survived=1的概率减少的最多。
- Age*Class对于模型来说是一个良好的人造变量，因为它与Survived有第二高的[负相关关系](https://zhida.zhihu.com/search?content_id=112784529&content_type=Article&match_order=1&q=负相关关系&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3OTE5NjIzNjMsInEiOiLotJ_nm7jlhbPlhbPns7siLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxMTI3ODQ1MjksImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.IPcn-7Dhvv0DNMOeXqyJIQpfAZFIDIIY7xhxVyMJKGU&zhida_source=entity)。
- Title有第二正相关的关系。

```python
coeff_df = pd.DataFrame(train_data.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df['Correlation'] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by = 'Correlation',ascending = False)
--------------------------------------------------------------
     Feature	Correlation
1	Sex	2.200978
5	Title	0.414362
4	Embarked	0.281026
6	IsAlone	0.262084
3	Fare	-0.022062
7	Age*Pclass	-0.085959
2	Age	-0.371647
0	Pclass	-1.083515
```

接着我们使用Support Vector Machines来建模，这是有监督学习模型，相关学习算法为分类和回归分析数据。给定一组训练样本集，每一个属于两个结果的其中一个，SVM训练算法建立一个模型将新的测试样本分配给两个结果其中一个，使其成为非概率二元线性分类器。

SVM的精确度高于Logistics Regression模型。

```python
# Support Vector Machines
svc = SVC()
svc.fit(X_train,Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train,Y_train)*100,2)
acc_svc
------------------------------------------------
D:\software\anaconda\lib\site-packages\sklearn\svm\base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
  "avoid this warning.", FutureWarning)





83.5
```

在模式识别中，k-Nearest Neighbors 算法（或简写为k-NN）是一个用于分类和回归的[非参方法](https://zhida.zhihu.com/search?content_id=112784529&content_type=Article&match_order=1&q=非参方法&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3OTE5NjIzNjMsInEiOiLpnZ7lj4Lmlrnms5UiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxMTI3ODQ1MjksImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.TP8qJe1zb11TiQGJ7lI5lH6ESQyJfh8zSzJ7SR5Qg4Q&zhida_source=entity)。一个样本通过一个其邻居的多数票决来分类，即实例被分配到距离它最近的k个邻居中最常见的那个类中（k是一个正整数，通常较小）。如果k=1，那么实例简单的被分配到单独最近邻的类中。

KNN精确度比Logistics Regression高但是比SVM低。

```python
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train,Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train,Y_train)*100,2)
acc_knn
----------------------------------------------------------------
84.06
```

在机器学习中，naive Bayes classifiers（[朴素贝叶斯分类器](https://zhida.zhihu.com/search?content_id=112784529&content_type=Article&match_order=1&q=朴素贝叶斯分类器&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3OTE5NjIzNjMsInEiOiLmnLTntKDotJ3lj7bmlq_liIbnsbvlmagiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxMTI3ODQ1MjksImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.TsgQZu7TkWHJ8m9pTHR1nqX1tnp5YWO-2h0gNnQfrYQ&zhida_source=entity)）是一组基于Bayes[理论变量](https://zhida.zhihu.com/search?content_id=112784529&content_type=Article&match_order=1&q=理论变量&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3OTE5NjIzNjMsInEiOiLnkIborrrlj5jph48iLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxMTI3ODQ1MjksImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.XebdNgtAD6Qtr7bT1NMqjmvLq2uf5yfvs3-1Vlzjrx8&zhida_source=entity)间有强独立性假设的简单[概率性](https://zhida.zhihu.com/search?content_id=112784529&content_type=Article&match_order=1&q=概率性&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3OTE5NjIzNjMsInEiOiLmpoLnjofmgKciLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxMTI3ODQ1MjksImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.S03y2t_Z4aZVRvpV2hpUNM__VOcSbQQ_O1QdHZIRRdY&zhida_source=entity)分类器。Naive Bayes Classifiers 高度可扩展，需要学习问题中一些参数与变量的数量呈线性关系。

这个模型的精确性在我们已估计的模型中是最低的。

```python
# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train,Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian =round(gaussian.score(X_train,Y_train)*100,2)
acc_gaussian
----------------------------------------------------------------
76.88
```

Perceptron(感知器)是一种二元分类器的有监督学习算法（分类器是一种函数，可以决定一个由一个向量或多个数字代表的输入属于某些特定类与否）。这是一种[线性分类器](https://zhida.zhihu.com/search?content_id=112784529&content_type=Article&match_order=2&q=线性分类器&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3OTE5NjIzNjMsInEiOiLnur_mgKfliIbnsbvlmagiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxMTI3ODQ1MjksImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MiwiemRfdG9rZW4iOm51bGx9.-8s6iVyOWF09Wk6fbVLIYUFwoDGNbqgRzJL_va_uQ6Y&zhida_source=entity)，也就是一种使其基于一线性预测函数的预测结果结合变量[向量](https://zhida.zhihu.com/search?content_id=112784529&content_type=Article&match_order=2&q=向量&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3OTE5NjIzNjMsInEiOiLlkJHph48iLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxMTI3ODQ1MjksImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MiwiemRfdG9rZW4iOm51bGx9.5c2sjrAoyfvTyEomIzad19amT8w3lK_df95gwenxe5I&zhida_source=entity)权重的分类算法。算法允许在线学习，每次处理训练集中的一个元素。

```python
# Percept
ron
perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
acc_perceptron
----------------------------------------------------
D:\software\anaconda\lib\site-packages\sklearn\linear_model\stochastic_gradient.py:166: FutureWarning: max_iter and tol parameters have been added in Perceptron in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
  FutureWarning)





78.79




# Linear SVC
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
acc_linear_svc
--------------------------------------------------
D:\software\anaconda\lib\site-packages\sklearn\svm\base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  "the number of iterations.", ConvergenceWarning)





79.46



# Stochastic Gradient Descent
sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd
--------------------------------------------------------
D:\software\anaconda\lib\site-packages\sklearn\linear_model\stochastic_gradient.py:166: FutureWarning: max_iter and tol parameters have been added in SGDClassifier in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
  FutureWarning)





77.44
```

此模型使用了decision tree决策树，决策树将变量（树枝）映射到目标值的决策上（树叶）。目标变量可以取有限组值的树模型称为分类树;在这些树结构中，叶子代表类标签，而分支代表导致这些类标签的特征连接。目标变量可以取连续值（通常真实数字）的决策树被称为回归树。

这个模型的精确度在已估计的模型中最高。

```python
# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree
---------------------------
86.64
```

接下来的模型Random Forests 随机森林 是最受欢迎的模型之一。[随机森林](https://zhida.zhihu.com/search?content_id=112784529&content_type=Article&match_order=2&q=随机森林&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3OTE5NjIzNjMsInEiOiLpmo_mnLrmo67mnpciLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxMTI3ODQ1MjksImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MiwiemRfdG9rZW4iOm51bGx9.A4_XIEp7rRYfqGB09_K6A6UmnnqLIYNpbL1qgUdVUys&zhida_source=entity)或[随机决策](https://zhida.zhihu.com/search?content_id=112784529&content_type=Article&match_order=1&q=随机决策&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3OTE5NjIzNjMsInEiOiLpmo_mnLrlhrPnrZYiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxMTI3ODQ1MjksImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.bMJD9Lxiu9rFI7VttI4lATdyaGoV4ARbO9wSwcrP40E&zhida_source=entity)森林是一种用于分类回归或其他任务的集成学习方法，在训练时构造大量[决策树](https://zhida.zhihu.com/search?content_id=112784529&content_type=Article&match_order=4&q=决策树&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3OTE5NjIzNjMsInEiOiLlhrPnrZbmoJEiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxMTI3ODQ1MjksImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6NCwiemRfdG9rZW4iOm51bGx9.BxlC818LwiV_RKv9aS6YIyDD5A7ZG2QVPadMD9dqpNA&zhida_source=entity)（n_estimators=100）并且输出所有单独树所提供分类的众数或平均数预测值。

此模型在已有估计的模型中精确度最高，我们决定使用此模型的输出（Y_pred）来作为我们竞赛提交结果。

```python
#Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest
-----------------------------------------------
86.64
```

## 7.1 模型评价

我们现在可以对所有模型进行评价，选出对我们问题来说最好的那个。Decision Tree 和Random Forest精确度相同，我们选择使用Random Forest 因为他们修正了Decision Tree有可能过度拟合训练集的问题。

```python
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)
----------------------------------------------------
Model	Score
3	Random Forest	86.64
8	Decision Tree	86.64
1	KNN	84.06
0	Support Vector Machines	83.50
2	Logistic Regression	81.26
7	Linear SVC	79.46
5	Perceptron	78.79
6	Stochastic Gradient Decent	77.44
4	Naive Bayes	76.88




#提交数据
submission = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('submission.csv', index=False)
```