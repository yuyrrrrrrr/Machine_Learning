# -*- coding: utf-8 -*-
"""Titanic 优化流程：复现原始 notebook 基线，并与改进的特征工程/模型对比。
评估统一用 5 折交叉验证重复 4 次（同一组 fold 供所有模型共用），保证可比。
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, VotingClassifier)
from lightgbm import LGBMClassifier

SEED = 42
DIR = r"D:\code\Machine Learning\Project\Titanic"

raw_train = pd.read_csv(DIR + r"\train.csv")
raw_test = pd.read_csv(DIR + r"\test.csv")


def baseline_features(tr, te):
    """完全复现 notebook 的清洗/特征流程"""
    tr, te = tr.copy(), te.copy()
    tr = tr.drop(['Cabin', 'Ticket'], axis=1)
    te = te.drop(['Cabin', 'Ticket'], axis=1)
    combine = [tr, te]
    for ds in combine:
        ds['Title'] = ds.Name.str.extract(r'([A-Za-z]+)\.', expand=False)
    for ds in combine:
        ds['Title'] = ds['Title'].replace(
            ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev',
             'Sir', 'Jonkheer', 'Dona'], 'Rare')
        ds['Title'] = ds['Title'].replace(['Mlle', 'Ms'], 'Miss').replace('Mme', 'Mrs')
        ds['Title'] = ds['Title'].map(
            {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}).fillna(0)
    tr = tr.drop(['Name', 'PassengerId'], axis=1)
    te = te.drop(['Name'], axis=1)
    combine = [tr, te]
    for ds in combine:
        ds['Sex'] = ds['Sex'].map({'female': 1, 'male': 0})
    # Age：按 Sex x Pclass 中位数填充，再按固定区间分箱（等价于 notebook 的 .loc 赋值）
    for ds in combine:
        for i in range(2):
            for j in range(1, 4):
                med = ds[(ds.Sex == i) & (ds.Pclass == j)].Age.dropna().median()
                ds.loc[(ds.Age.isnull()) & (ds.Sex == i) & (ds.Pclass == j), 'Age'] = med
        ds['Age'] = pd.cut(ds['Age'], [-1, 16, 32, 48, 64, 200],
                           labels=[0, 1, 2, 3, 4]).astype(int)
        ds['FamilySize'] = ds['Parch'] + ds['SibSp'] + 1
        ds['IsAlone'] = (ds['FamilySize'] == 1).astype(int)
    for ds in combine:
        ds.drop(['Parch', 'SibSp', 'FamilySize'], axis=1, inplace=True)
        ds['Age*Pclass'] = ds.Age * ds.Pclass
    freq_port = tr.Embarked.dropna().mode()[0]
    for ds in combine:
        ds['Embarked'] = ds['Embarked'].fillna(freq_port).map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
    te['Fare'] = te['Fare'].fillna(te['Fare'].dropna().median())
    for ds in combine:
        ds.loc[ds.Fare <= 7.91, 'Fare'] = 0
        ds.loc[(ds.Fare > 7.91) & (ds.Fare <= 14.454), 'Fare'] = 1
        ds.loc[(ds.Fare > 14.454) & (ds.Fare <= 31), 'Fare'] = 2
        ds.loc[ds.Fare > 31, 'Fare'] = 3
        ds['Fare'] = ds.Fare.astype(int)
    return tr.drop('Survived', axis=1), tr.Survived, te.drop('PassengerId', axis=1)


def improved_features(tr, te):
    """改进版：补齐被丢弃的 Cabin/Ticket 信息，加入家庭与称谓等派生特征"""
    df = pd.concat([tr.drop('Survived', axis=1), te], ignore_index=True, sort=False)
    y = tr.Survived.reset_index(drop=True)

    # 称谓：Mlle/Ms/Mme 归一化，其余罕见称谓合并为 Rare
    df['Title'] = df.Name.str.extract(r'([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss').replace('Mme', 'Mrs')
    df['Title'] = df['Title'].where(
        df['Title'].isin(['Mr', 'Miss', 'Mrs', 'Master']), 'Rare')

    # 舱位：首字母代表甲板，缺失单列 U；同舱人数反映同行家庭规模
    df['Deck'] = df.Cabin.str[0].fillna('U')
    df['HasCabin'] = df.Cabin.notna().astype(int)
    df['CabinShared'] = df.groupby('Cabin').Cabin.transform('size').fillna(1)

    # 票号：前缀代表购票渠道/舱区，同票号人数反映同行团体规模
    pre = df.Ticket.str.extract(r'^([^\d\s]*)', expand=False).str.replace(
        r'[^A-Za-z]', '', regex=True).str.upper()
    pre = pre.mask(pre.isna() | (pre == ''), 'NUM')
    df['TicketPrefix'] = pre.where(pre.map(pre.value_counts()) >= 10, 'RARE')
    df['TicketGroupSize'] = df.groupby('Ticket').Ticket.transform('size')

    # 家庭：姓+票号 才能把同姓的不同家庭区分开
    df['Surname'] = df.Name.str.split(',').str[0].str.strip()
    df['FamilyGroupSize'] = df.groupby(['Surname', 'Ticket']).Name.transform('size')
    df['FamilySize'] = df.Parch + df.SibSp + 1
    df['IsAlone'] = (df.FamilySize == 1).astype(int)

    df['Sex'] = df.Sex.map({'female': 1, 'male': 0})

    # Age：按 Sex x Pclass x Title 中位数填充（比原流程的 Sex x Pclass 更细），保留连续值
    df['AgeMissing'] = df.Age.isna().astype(int)
    df['Age'] = df.Age.fillna(
        df.groupby(['Sex', 'Pclass', 'Title']).Age.transform('median'))
    df['Age'] = df.Age.fillna(df.Age.median())

    # Fare：缺失按 Pclass 中位数；取对数压缩长尾，并按人头摊分
    df['Fare'] = df.Fare.fillna(df.groupby('Pclass').Fare.transform('median'))
    df['FareLog'] = np.log1p(df.Fare)
    df['FarePerPerson'] = df.Fare / df.FamilySize

    df['WomanOrChild'] = ((df.Sex == 1) | (df.Age < 15)).astype(int)
    df['Embarked'] = df.Embarked.fillna('S')

    num = ['Pclass', 'Sex', 'Age', 'AgeMissing', 'SibSp', 'Parch', 'FamilySize',
           'IsAlone', 'FamilyGroupSize', 'TicketGroupSize', 'FareLog',
           'FarePerPerson', 'HasCabin', 'CabinShared', 'WomanOrChild']
    cat = ['Title', 'Deck', 'TicketPrefix', 'Embarked']
    X = pd.get_dummies(df[num + cat], columns=cat).astype(float)
    n = len(tr)
    return (X.iloc[:n].reset_index(drop=True), y,
            X.iloc[n:].reset_index(drop=True))


def cv_eval(make_model, X, y, folds, name):
    accs = []
    for tr_idx, va_idx in folds:
        m = make_model()
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        accs.append((m.predict(X.iloc[va_idx]) == y.iloc[va_idx].values).mean())
    return {"model": name, "cv_mean": np.mean(accs), "cv_std": np.std(accs)}


# ---------------- 统一 fold ----------------
folds = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=4,
                                     random_state=SEED).split(raw_train, raw_train.Survived))

Xb, yb, Xb_test = baseline_features(raw_train, raw_test)
Xi, yi, Xi_test = improved_features(raw_train, raw_test)

MODELS = {
    "LR": lambda: Pipeline([("sc", StandardScaler()),
                            ("m", LogisticRegression(max_iter=1000, C=1.0, random_state=SEED))]),
    "SVC": lambda: Pipeline([("sc", StandardScaler()),
                             ("m", SVC(C=2, gamma="scale", probability=True, random_state=SEED))]),
    "RF": lambda: RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                                         random_state=SEED, n_jobs=1),
    "ET": lambda: ExtraTreesClassifier(n_estimators=500, min_samples_leaf=2,
                                       random_state=SEED, n_jobs=1),
    "GBM": lambda: GradientBoostingClassifier(random_state=SEED),
    "LGBM": lambda: LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=8,
                                   subsample=0.8, colsample_bytree=0.8, verbose=-1,
                                   random_state=SEED),
}

rows = []


class ConstRule:
    """硬规则：女性或男孩(Master)判定生还"""
    def __init__(self, X): self.X = X
    def fit(self, X, y): return self


def rule_accuracy(X, y, folds, name):
    accs = []
    for tr_idx, va_idx in folds:
        Xv, yv = X.iloc[va_idx], y.iloc[va_idx].values
        kid = (Xv["Title_Master"] == 1) if "Title_Master" in Xv.columns else (Xv["Title"] == 4)
        pred = ((Xv["Sex"] == 1) | kid).astype(int).values
        accs.append((pred == yv).mean())
    return {"model": name, "cv_mean": np.mean(accs), "cv_std": np.std(accs)}


print("=" * 72)
print("原始流程（notebook 特征 + 相同模型）")
print("=" * 72)
for k, f in MODELS.items():
    r = cv_eval(f, Xb, yb, folds, k)
    r["feature"] = "baseline"; rows.append(r)
    print(f"{k:<6} CV={r['cv_mean']:.4f} +- {r['cv_std']:.4f}")
r = rule_accuracy(Xb, yb, folds, "硬规则(女+男孩)"); r["feature"] = "baseline"; rows.append(r)
print(f"规则   CV={r['cv_mean']:.4f} +- {r['cv_std']:.4f}")

print("=" * 72)
print("改进流程（新增 Cabin/Ticket/家庭特征）")
print("=" * 72)
for k, f in MODELS.items():
    r = cv_eval(f, Xi, yi, folds, k)
    r["feature"] = "improved"; rows.append(r)
    print(f"{k:<6} CV={r['cv_mean']:.4f} +- {r['cv_std']:.4f}")

# 软投票集成
ENSEMBLE = {
    "Vote(LR,SVC,RF,ET,GBM,LGBM)": lambda: VotingClassifier(
        [(k, f()) for k, f in MODELS.items()], voting="soft", n_jobs=1),
    "Vote(RF,ET,GBM,LGBM)": lambda: VotingClassifier(
        [(k, MODELS[k]()) for k in ["RF", "ET", "GBM", "LGBM"]], voting="soft", n_jobs=1),
}
for k, f in ENSEMBLE.items():
    r = cv_eval(f, Xi, yi, folds, k)
    r["feature"] = "improved"; rows.append(r)
    print(f"{k:<28} CV={r['cv_mean']:.4f} +- {r['cv_std']:.4f}")

res = pd.DataFrame(rows).pivot_table(index="model", columns="feature",
                                     values="cv_mean").dropna()
res["提升"] = res["improved"] - res["baseline"]
print("=" * 72)
print("同模型下：原始流程 vs 改进流程")
print(res.sort_values("improved", ascending=False).round(4).to_string())

pd.DataFrame(rows).round(4).to_csv(DIR + r"\cv_results.csv", index=False, encoding="utf-8-sig")
print("\n特征数：baseline =", Xb.shape[1], " improved =", Xi.shape[1])


# ==================== 第二部分：调参、集成与最终提交 ====================
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedGroupKFold
from sklearn.ensemble import StackingClassifier

# 同姓 + 同票号视为一个家庭组，用于更保守的评估
groups = (raw_train.Name.str.split(",").str[0].str.strip() + "_" + raw_train.Ticket).values
gfold = list(StratifiedGroupKFold(n_splits=5, shuffle=True,
                                  random_state=SEED).split(Xi, yi, groups))

print("\n" + "=" * 72)
print("家庭组切分 vs 随机切分（LGBM）")
print("=" * 72)
g = cv_eval(MODELS["LGBM"], Xi, yi, gfold, "LGBM-groupCV")
print(f"随机切分 CV={0.8305:.4f}    家庭组切分 CV={g['cv_mean']:.4f} +- {g['cv_std']:.4f}")

# 小网格调参（随机切分调参是常规做法；家庭组 CV 用于看提升是否稳健）
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
grids = {
    "LGBM": (LGBMClassifier(verbose=-1, random_state=SEED),
             {"n_estimators": [300, 600], "learning_rate": [0.02, 0.05],
              "num_leaves": [4, 8, 15], "min_child_samples": [5, 20]}),
    "RF": (RandomForestClassifier(random_state=SEED, n_jobs=1),
           {"n_estimators": [500], "max_depth": [4, 6, None],
            "min_samples_leaf": [1, 2, 4], "max_features": ["sqrt", 0.5]}),
    "SVC": (Pipeline([("sc", StandardScaler()), ("m", SVC(probability=True, random_state=SEED))]),
            {"m__C": [0.5, 1, 2, 5], "m__gamma": ["scale", 0.05, 0.1]}),
}
print("\n" + "=" * 72)
print("网格调参（5 折 CV 选参）")
print("=" * 72)
best = {}
for name, (est, pg) in grids.items():
    gs = GridSearchCV(est, pg, cv=skf, scoring="accuracy", n_jobs=1)
    gs.fit(Xi, yi)
    best[name] = gs.best_estimator_
    print(f"{name:<5} 最优 CV={gs.best_score_:.4f}  参数={gs.best_params_}")

# 集成：软投票 与 LR 元学习器堆叠
final_candidates = {
    "LGBM-tuned": best["LGBM"],
    "Vote(4模型)": VotingClassifier([(k, best[k]) for k in ["LGBM", "RF", "SVC"]]
                                    + [("LR", MODELS["LR"]())], voting="soft", n_jobs=1),
    "Stack(4模型->LR)": StackingClassifier(
        [(k, best[k]) for k in ["LGBM", "RF", "SVC"]] + [("LR", MODELS["LR"]())],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=StratifiedKFold(5, shuffle=True, random_state=SEED), n_jobs=1),
}
print("\n" + "=" * 72)
print("最终候选模型对比")
print("=" * 72)
final_rows = []
for name, est in final_candidates.items():
    r = cv_eval(lambda e=est: e, Xi, yi, folds, name)
    rg = cv_eval(lambda e=est: e, Xi, yi, gfold, name)
    final_rows.append({"model": name, "随机CV": r["cv_mean"], "家庭组CV": rg["cv_mean"]})
    print(f"{name:<18} 随机CV={r['cv_mean']:.4f}  家庭组CV={rg['cv_mean']:.4f}")
pd.DataFrame(final_rows).round(4).to_csv(DIR + r"\final_cv.csv", index=False, encoding="utf-8-sig")

# 用随机 CV 与家庭组 CV 均分最高者作为最终模型，写入提交
best_name = max(final_rows, key=lambda d: d["随机CV"] + d["家庭组CV"])["model"]
final_model = final_candidates[best_name]
print(f"\n最终模型：{best_name}")
final_model.fit(Xi, yi)

import shutil, os
if os.path.exists(DIR + r"\submission.csv") and not os.path.exists(DIR + r"\submission_0758_backup.csv"):
    shutil.copy(DIR + r"\submission.csv", DIR + r"\submission_0758_backup.csv")
sub = pd.DataFrame({"PassengerId": raw_test.PassengerId,
                    "Survived": final_model.predict(Xi_test).astype(int)})
sub.to_csv(DIR + r"\submission.csv", index=False)
print("已写入 submission.csv，预测生还率 =", round(sub.Survived.mean(), 4))
print("空值检查:", int(sub.isna().sum().sum()), " 生还数量:", int(sub.Survived.sum()), "/", len(sub))

# 特征重要度（LGBM），确认新特征确实被用上
imp = pd.Series(best["LGBM"].feature_importances_, index=Xi.columns).sort_values(ascending=False)
print("\nTop 15 特征重要度:\n", imp.head(15).round(1).to_string())
