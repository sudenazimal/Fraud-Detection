import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from memory_profiler import memory_usage
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

print(
    "###------------------------------------creditcard.csv-------------------------------------------------###"
)

loaded_data = joblib.load("./data/preprocessed/creditcard_preprocessed.pkl")

X_train = loaded_data["X_train"]
X_test = loaded_data["X_test"]
y_train = loaded_data["y_train"]
y_test = loaded_data["y_test"]

scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    use_label_encode=False,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]
print(
    "---------------------------Creditcard Without Smote---------------------------------"
)
print("confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("\nclassification report:\n", classification_report(y_test, y_pred))
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC-AUC : {roc_auc}")
pr_auc = average_precision_score(y_test, y_pred_proba)
print(f"PR-AUC : {pr_auc}")
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"ROC curve (ares = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("false positive rate")
plt.ylabel("true positive rate")
plt.title("ROC curve")
plt.legend(loc="lower right")
plt.grid()
plt.show()

precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
plt.figure(figsize=(7, 5))
plt.plot(precision, recall, label=f"PR curve (area = {pr_auc:.4f}) ")
plt.xlabel("recall")
plt.ylabel("precision")
plt.title("precision-recall curve")
plt.legend(loc="lower left")
plt.grid()
plt.show()

### --------------------------------------with SMOTE-------------------------------------

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print(f"Original train class distribution: {y_train.value_counts()}")
print(f"Resampled train class distribution: {y_train_resampled.value_counts()}")

modelS = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
)
modelS.fit(X_train_resampled, y_train_resampled)

y_pred = modelS.predict(X_test)
y_pred_proba = modelS.predict_proba(X_test)[:, 1]

print(
    "---------------------------Creditcard With Smote---------------------------------"
)
print("confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("\nclassification report:\n", classification_report(y_test, y_pred))
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC-AUC : {roc_auc}")
pr_auc = average_precision_score(y_test, y_pred_proba)
print(f"PR-AUC : {pr_auc}")
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"ROC curve (ares = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("false positive rate")
plt.ylabel("true positive rate")
plt.title("ROC curve")
plt.legend(loc="lower right")
plt.grid()
plt.show()

precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
plt.figure(figsize=(7, 5))
plt.plot(precision, recall, label=f"PR curve (area = {pr_auc:.4f}) ")
plt.xlabel("recall")
plt.ylabel("precision")
plt.title("precision-recall curve")
plt.legend(loc="lower left")
plt.grid()
plt.show()

print(
    "###------------------------------------fraudTest-Train.csv-------------------------------------------------###"
)

loaded_data = joblib.load("./data/preprocessed/fraud_preprocessed.pkl")

raw_data2_test = pd.read_csv("data/fraudTest.csv")

X_train2 = loaded_data["X_train"]
X_test2 = loaded_data["X_test"]
y_train2 = loaded_data["y_train"]
y_test2 = loaded_data["y_test"]


scale_pos_weight = y_train2.value_counts()[0] / y_train2.value_counts()[1]

model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    use_label_encode=False,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
)

model.fit(X_train2, y_train2)
y_pred2 = model.predict(X_test2)
y_pred_proba2 = model.predict_proba(X_test2)[:, 1]

print("---------------------------Fraud Without Smote---------------------------------")
print("confusion matrix:\n", confusion_matrix(y_test2, y_pred2))
print("\nclassification report:\n", classification_report(y_test2, y_pred2))
roc_auc = roc_auc_score(y_test2, y_pred_proba2)
print(f"ROC-AUC : {roc_auc}")
pr_auc = average_precision_score(y_test2, y_pred_proba2)
print(f"PR-AUC : {pr_auc}")
fpr, tpr, _ = roc_curve(y_test2, y_pred_proba2)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"ROC curve (ares = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("false positive rate")
plt.ylabel("true positive rate")
plt.title("ROC curve")
plt.legend(loc="lower right")
plt.grid()
plt.show()

precision, recall, _ = precision_recall_curve(y_test2, y_pred_proba2)
plt.figure(figsize=(7, 5))
plt.plot(precision, recall, label=f"PR curve (area = {pr_auc:.4f}) ")
plt.xlabel("recall")
plt.ylabel("precision")
plt.title("precision-recall curve")
plt.legend(loc="lower left")
plt.grid()
plt.show()
merged1 = np.hstack((raw_data2_test, y_pred2.reshape(-1, 1)))
### --------------------------------------with SMOTE-------------------------------------

smote = SMOTE(random_state=42)
X_train_resampled2, y_train_resampled2 = smote.fit_resample(X_train2, y_train2)
print(f"Original train class distribution: {y_train2.value_counts()}")
print(f"Resampled train class distribution: {y_train_resampled2.value_counts()}")

modelS = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
)
modelS.fit(X_train_resampled2, y_train_resampled2)

y_pred2 = modelS.predict(X_test2)
y_pred_proba2 = modelS.predict_proba(X_test2)[:, 1]

print("---------------------------Fraud With Smote---------------------------------")
print("confusion matrix:\n", confusion_matrix(y_test2, y_pred2))
print("\nclassification report:\n", classification_report(y_test2, y_pred2))
roc_auc = roc_auc_score(y_test2, y_pred_proba2)
print(f"ROC-AUC : {roc_auc}")
pr_auc = average_precision_score(y_test2, y_pred_proba2)
print(f"PR-AUC : {pr_auc}")
fpr, tpr, _ = roc_curve(y_test2, y_pred_proba2)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"ROC curve (ares = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("false positive rate")
plt.ylabel("true positive rate")
plt.title("ROC curve")
plt.legend(loc="lower right")
plt.grid()
plt.show()

precision, recall, _ = precision_recall_curve(y_test2, y_pred_proba2)
plt.figure(figsize=(7, 5))
plt.plot(precision, recall, label=f"PR curve (area = {pr_auc:.4f}) ")
plt.xlabel("recall")
plt.ylabel("precision")
plt.title("precision-recall curve")
plt.legend(loc="lower left")
plt.grid()
plt.show()
merged2 = np.hstack((raw_data2_test, y_pred2.reshape(-1, 1)))

### ---------------------ORIGINAL VS PREDICTED ------------------------------------

print(pd.DataFrame(merged1).head())
print(pd.DataFrame(merged1).describe())
print(pd.DataFrame(merged1).info())
