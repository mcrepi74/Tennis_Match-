# imports
import itertools
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.io.parsers import read_csv
import seaborn as sns
from datetime import date, timedelta
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
import scipy
import scipy.stats

# read in data
df_all = pd.read_csv("ATP_tweaked.csv", parse_dates=['tourney_date'])

# see the columns'             
df_all.columns

features_match = ["tourney_id", "tourney_level", "surface", "tourney_date", "round", "best_of"]

features_player = ["id", "seed", "entry", "hand", "ht", "age", "rank", "rank_points"]
features_player = [("p1_"+f) for f in features_player] + [("p2_"+f) for f in features_player]
outcome = "p1_won"

df_data = df_all[features_match + features_player + [outcome]].copy()

min_match = 20
times = pd.value_counts(df_data["p1_id"].tolist() + df_data["p2_id"].tolist())
ids = times.index[times >= min_match].values
flag = np.full(len(df_data), False)

keep = [r[1]["p1_id"] in ids and r[1]["p2_id"] in ids for r in df_data.iterrows()]
df_data = df_data.iloc[keep]
print(len(df_data))


# replacing missing entry feature with "unknown"
df_data[["p1_entry", "p2_entry"]] = df_data[["p1_entry", "p2_entry"]].fillna("unknown")

# replacing missing seed value with 50
df_data[["p1_seed", "p2_seed"]] = df_data[["p1_seed", "p2_seed"]].fillna(50)

# simple imputation of height data
df_data[["p1_ht", "p2_ht"]] = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(df_data[["p1_ht", "p2_ht"]])

# drop rows still having missing data (these are missing rank data)
df_data.dropna(inplace=True)

df_data_mirror = df_data.copy()

# get the name changing mapping
mapping = np.array([(c, c.replace("p2", "p1")) for c in df_data_mirror.columns if c.startswith("p2")])
mapping = dict(np.concatenate((mapping, mapping[:,::-1]), axis=0))

# rename columns
df_data_mirror.rename(mapping, axis="columns", inplace=True)

# flip p1_won as now players 1 and 2 have been fliped
df_data_mirror["p1_won"] = 1 - df_data_mirror["p1_won"]

# concatenate
df_data = pd.concat((df_data, df_data_mirror), axis=0).reset_index(drop=True)

# sort by the match date
df_data.sort_values(by="tourney_date", inplace=True)

def match_statistics(player_id: str, until_date: datetime.date, Δ_days: int, min_match: int=5) -> float:
    """Calculate the recent match winning rate of a player over a past time period
    
    Args:
        - player_id: the player id
        - until_date: the ending date of the period
        - Δ_days: the number of days looking into the past
        - min_match: the minimum number of matches the player has played recently. Less than this number, no winning rate will be computed and simply 0 is returned.

    Returns: the winning rate (0 to 1) of won matches
    """
    df_player = df_data.loc[df_data["p1_id"] == player_id]
    diff = (until_date - df_player["tourney_date"]).dt.days
    df_player = df_player.loc[(0 < diff) & (diff <= Δ_days)]
    if len(df_player) > min_match:
        return df_player["p1_won"].mean()
    else:
        return 0

Δ_days = 365
p1_stats, p2_stats = np.full(len(df_data), np.nan), np.full(len(df_data), np.nan)
for index in range(len(df_data)):    
    p1_id, p2_id, match_date = df_data["p1_id"].iloc[index], df_data["p2_id"].iloc[index], df_data["tourney_date"].iloc[index]
    p1_stats[index], p2_stats[index] = match_statistics(p1_id, match_date, Δ_days), match_statistics(p2_id, match_date, Δ_days)

df_data["p1_stat"] = p1_stats
df_data["p2_stat"] = p2_stats

df_data["Δ_seed"] = df_data["p1_seed"] - df_data["p2_seed"]
df_data["Δ_ht"] = df_data["p1_ht"] - df_data["p2_ht"]
df_data["Δ_age"] = df_data["p1_age"] - df_data["p2_age"]
df_data["Δ_stat"] = df_data["p1_stat"] - df_data["p2_stat"]
df_data["Δ_rank"] = df_data["p1_rank"] - df_data["p2_rank"]
df_data["Δ_rank_points"] = df_data["p1_rank_points"] - df_data["p2_rank_points"]

y = df_data[outcome]

# remove tournament and player ids
X = df_data.drop([outcome, "tourney_id", "p1_id", "p2_id"], axis="columns").set_index("tourney_date")

# dummie coding of categorical variables
X = pd.get_dummies(X, columns=["tourney_level","surface","round","p1_entry","p1_hand","p2_entry","p2_hand"])


years = np.array([x.year for x in X.index])

indices_train = (years==2015) | (years==2016)
indices_validate = years==2017
indices_test = (years==2018) | (years==2019)

Xtrain, ytrain = X.iloc[indices_train], y.iloc[indices_train]
Xvalidate, yvalidate = X.iloc[indices_validate], y.iloc[indices_validate]
Xtest, ytest = X.iloc[indices_test], y.iloc[indices_test]

print(f"training set size: {len(ytrain)}; validate set size: {len(yvalidate)}; test set size: {len(ytest)}")

# random forest model, the parameters could be tuned using the performance on the validation data. However, for the sake of time we will not do it.
model = RandomForestClassifier(n_estimators=1024, criterion="gini", min_samples_split=16, n_jobs=4, random_state=2022)

# fit model
model.fit(Xtrain, ytrain)

# predict on the validate data
yscore = model.predict_proba(Xvalidate)[:,1]

# compute AUC
auc = roc_auc_score(yvalidate, yscore)
print(f"auc on validation data: {auc}")

T = np.arange(0, 1, 0.01)
TP = np.full(len(T), np.nan)
FP = np.full(len(T), np.nan)
TN = np.full(len(T), np.nan)
FN = np.full(len(T), np.nan)

for (index, t) in enumerate(T):
    ypred = np.array([1 if s>=t else 0 for s in yscore])
    TP[index] = sum((ypred==1)&(yvalidate==1))
    FP[index] = sum((ypred==1)&(yvalidate==0))
    TN[index] = sum((ypred==0)&(yvalidate==0))
    FN[index] = sum((ypred==0)&(yvalidate==1))

TPR, FPR, FNR = TP/(TP+FN), FP/(FP+TN), FN/(FN+TP)

fig, axes = plt.subplots(1, 2, gridspec_kw={'width_ratios': [2, 3]}, figsize=(10,4), dpi=100)
axes[0].plot(FPR, TPR, color="green")
axes[0].plot([0, 1], [0, 1], ls="--", color="black", lw=1)
axes[0].grid(color="lightgray", lw=0.5)
axes[0].set_xlabel("False positive rate")
axes[0].set_ylabel("True positive rate")
axes[0].set_title("ROC Curve (AUC={:.3f})".format(auc))
axes[1].plot(T, FPR, label="False Positive Rate")
axes[1].plot(T, FNR, label="False Negative Rate")
axes[1].legend()
axes[1].grid(color="lightgray", lw=0.5)
axes[1].set_xlabel("threshold")
axes[1].set_title("FPR/FNR vs. threshold")

fig.tight_layout()

# get feature importances
importances = permutation_importance(model, Xvalidate, yvalidate, scoring="roc_auc", n_repeats=10, n_jobs=4, random_state=2022)

# sort the features according to descending order of importance
# Convert importances["importances_mean"] to a NumPy array
importances_mean = np.array(importances["importances_mean"])

# Sort the features according to descending order of importance
idx = np.argsort(importances_mean)[::-1]
importances_std = importances["importances_std"][idx]
feature_names = X.columns[idx]


# plot the feature importances
fig, ax = plt.subplots(1, 1, figsize=(15,6), dpi=90)
ax.bar(feature_names, importances_mean, yerr=importances_std, width=0.6)
ax.xaxis.set_ticks(list(np.arange(0, len(importances_mean))))
ax.set_xticklabels(feature_names, rotation = 90)
ax.grid(color="lightgray", lw=0.5)
ax.set_title("Permutation feature importance")
ax.set_ylabel("Feature importance")
fig.tight_layout()

# names of features to keep
features_keep = ["Δ_stat", "Δ_rank_points", "Δ_rank", "p1_stat", "p2_stat", "p1_rank_points", "p2_rank_points"] +\
    [c for c in Xtrain.columns if c.startswith("surface")]

model.fit(Xtrain[features_keep], ytrain)
auc = roc_auc_score(yvalidate, model.predict_proba(Xvalidate[features_keep])[:,1])
print(f"auc on validation data: {auc}")

features_keep = ["Δ_stat", "Δ_rank_points", "Δ_rank", "p1_stat", "p2_stat", "p1_rank_points", "p2_rank_points"]

# concatenate train and validate data
df_sns = pd.concat((
    pd.concat((Xtrain[features_keep], Xvalidate[features_keep]), axis=0).reset_index(drop=True), 
    pd.concat((ytrain, yvalidate), axis=0).reset_index(drop=True)
    ), axis="columns")

# some of the features are highly skewed. For these ones we will use log scale
for xname in features_keep:
    if scipy.stats.skew(df_sns[xname]) > 2:
        df_sns[xname] = np.log10(df_sns[xname]+1)
        df_sns.rename({xname: xname+"(log)"}, axis="columns", inplace=True)

features_keep = df_sns.columns[:-1]

df_sns[""] = ""
fig, axes = plt.subplots(nrows=2, ncols=int(np.ceil(len(features_keep)/2)), figsize=(12,8), dpi=80)
axes = list(itertools.chain.from_iterable(axes))

for (index, xname) in enumerate(features_keep):
    ax = axes[index]
    sns.violinplot(data=df_sns, y=xname, x="", hue="p1_won", split=True, orient="v", cut=0, inner="quartile", ax=ax)
    #ax.legend(loc="upper center")
    ax.set_title(xname)

fig.tight_layout()

features_keep = ["Δ_stat", "Δ_rank_points", "Δ_rank", "p1_stat", "p2_stat", "p1_rank_points", "p2_rank_points"] +\
    [c for c in Xtrain.columns if c.startswith("surface")]

# Final model
model = RandomForestClassifier(n_estimators=1024, criterion="gini", min_samples_split=16, n_jobs=4, random_state=2022)

# fit model
model.fit(pd.concat((Xtrain[features_keep], Xvalidate[features_keep]), axis=0), pd.concat((ytrain, yvalidate), axis=0))

# predict on the validate data
yscore = model.predict_proba(Xtest[features_keep])[:,1]

# compute AUC
auc = roc_auc_score(ytest, yscore)
print(f"auc on test data: {auc}")
