import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from memory_profiler import memory_usage
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

###df = pd.read_csv("data/fraudTest.csv")
###print("df.info:")
###df.info()
###print("df.describe : ")
###print(df.describe())
###print("df.head")
###print(df.head())
###i = 0
###while i < 10:
###    print(df.iloc[i])
###    i += 1

print(
    "###------------------------------------creditcard.csv-------------------------------------------------###"
)

data1 = pd.read_csv("data/creditcard.csv")
# data2 = pd.read_csv("data/creditcard_2023.csv")
#
# common_rows = pd.merge(data1, data2, "inner")
# print("number of same entries in data1 and data2",len(common_rows))
### NO SIMILARITIES BETWEEN DATA1 AND DATA2

##DATA1 PREPROCESSING
data1.describe()

data1.info()
total_samples = len(data1)
fraud_percentage = data1["Class"].mean() * 100
non_fraud_percentage = 100 - fraud_percentage

print(f"Total samples: {total_samples:,}")
print(f"Fraud percentage: {fraud_percentage:.4f}%")
print(f"Non-fraud percentage: {non_fraud_percentage:.4f}%")

class_counts = data1["Class"].value_counts()
print("\nAbsolute counts : ")
print(f"Fraud Cases : {class_counts[1]:,}")
print(f"Non Fraud Cases : {class_counts[0]:,}")

X = data1.drop("Class", axis=1)
Y = data1["Class"]
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42, stratify=Y
)

fraud_prct_train = y_train.mean() * 100
fraud_prct_test = y_test.mean() * 100
print("\nIn Split : ")
print(f"fraud percentage of Train: {fraud_prct_train:.4f}%")
print(f"fraud percentage of Test: {fraud_prct_test:.4f}% ")

print(f"number of frauds in test : {y_test.value_counts()[1]}")
print(f"number of frauds in train : {y_train.value_counts()[1]}")

### Fraud rate in test and train data is similar to both themselfs and original data. this indicates good split.
keys_dict = {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}
joblib.dump(
    keys_dict,
    "./data/preprocessed/creditcard_preprocessed.pkl",
)
### saved the processed data for model implementation in another file

pca = PCA(n_components=2)  # reduce to 2 components for visualization
X_pca = pca.fit_transform(X)

# Explained variance
print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Cumulative explained variance:", np.cumsum(pca.explained_variance_ratio_))

target_names = ["Class 0", "Class 1"]
feature_names = X.columns.tolist()
colors = ["red", "blue"]

plt.figure(figsize=(8, 6))
# for i, color in enumerate(colors):
#    plt.scatter(X_pca[Y == i, 0], X_pca[Y == i, 1], label=target_names[i], color=color)
for i, color in zip([0, 1], colors):
    plt.scatter(X_pca[Y == i, 0], X_pca[Y == i, 1], label=target_names[i], color=color)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA of Dataset")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

loadings = pd.DataFrame(pca.components_.T, columns=["PC1", "PC2"], index=feature_names)
print("\nPCA Loadings:\n", loadings)

###-------------------------------------------------------------------------------------###
print(
    "###------------------------------------fraudTest-Train.csv-------------------------------------------------###"
)

raw_data2 = pd.read_csv("data/fraudTrain.csv")
raw_data2_test = pd.read_csv("data/fraudTest.csv")

fraud_percentage2 = (raw_data2["is_fraud"].value_counts()[1] / len(raw_data2)) * 100
fraud_percentage2_test = (
    raw_data2_test["is_fraud"].value_counts()[1] / len(raw_data2_test)
) * 100

print(f"Train total samples: {len(raw_data2):,}")
print(f"Test total samples: {len(raw_data2_test):,}")
print(f"Train Fraud percentage: {fraud_percentage2:.4f}%")
print(f"Train Non-fraud percentage: {100 - fraud_percentage2:.4f}%")
print(f"Test Fraud percentage: {fraud_percentage2_test:.4f}%")
print(f"Test Non-fraud percentage: {100 - fraud_percentage2_test:.4f}%")

class_counts2 = raw_data2["is_fraud"].value_counts()
class_counts2_test = raw_data2_test["is_fraud"].value_counts()
print(f"\nTrain Absolute counts : {class_counts2}")
print(f"Train Fraud Cases : {class_counts2[1]:,}")
print(f"Train Non Fraud Cases : {class_counts2[0]:,}")
print(f"\nTest Absolute counts : {class_counts2_test}")
print(f"TestFraud Cases : {class_counts2_test[1]:,}")
print(f"Test Non Fraud Cases : {class_counts2_test[0]:,}")

X_train2 = raw_data2.drop("is_fraud", axis=1)
y_train2 = raw_data2["is_fraud"]

X_test2 = raw_data2_test.drop("is_fraud", axis=1)
y_test2 = raw_data2_test["is_fraud"]

fraud_prct_train_2 = y_train2.mean() * 100
fraud_prct_test_2 = y_test2.mean() * 100

print(f"fraud percentage of Train: {fraud_prct_train_2:.4f}%")
print(f"fraud percentage of Test: {fraud_prct_test_2:.4f}% ")

print(f"number of frauds in test : {y_test2.value_counts()[1]}")
print(f"number of frauds in train : {y_train2.value_counts()[1]}")

df_clean = raw_data2
df_clean["trans_date_trans_time"] = pd.to_datetime(df_clean["trans_date_trans_time"])
df_clean["hour"] = df_clean["trans_date_trans_time"].dt.hour
df_clean["day_of_week"] = df_clean["trans_date_trans_time"].dt.dayofweek
df_clean["month"] = df_clean["trans_date_trans_time"].dt.month
df_clean = df_clean.drop("trans_date_trans_time", axis=1)

current_year = pd.Timestamp.now().year
df_clean["age"] = current_year - pd.to_datetime(df_clean["dob"]).dt.year

df_test_clean = raw_data2_test
df_test_clean["trans_date_trans_time"] = pd.to_datetime(
    df_test_clean["trans_date_trans_time"]
)
df_test_clean["hour"] = df_test_clean["trans_date_trans_time"].dt.hour
df_test_clean["day_of_week"] = df_test_clean["trans_date_trans_time"].dt.day_of_week
df_test_clean["month"] = df_test_clean["trans_date_trans_time"].dt.month
df_test_clean = df_test_clean.drop("trans_date_trans_time", axis=1)

df_test_clean["age"] = current_year - pd.to_datetime(df_clean["dob"]).dt.year

# Remove unnecessary fields
df_clean = df_clean.drop(
    [
        "first",
        "last",  # PII
        "street",  # Address info redundant with lat/long
        "trans_num",  # Transaction identifier
        "unix_time",  # Redundant with datetime
        "dob",  # Will calculate age instead
        "zip",  # Redundant with city/state
    ],
    axis=1,
)
df_test_clean = df_test_clean.drop(
    [
        "first",
        "last",  # PII
        "street",  # Address info redundant with lat/long
        "trans_num",  # Transaction identifier
        "unix_time",  # Redundant with datetime
        "dob",  # Will calculate age instead
        "zip",  # Redundant with city/state
    ],
    axis=1,
)
# print(df_clean.head())
# print(df_clean.describe())
# print(df_clean.info())
# print(df_test_clean.head())
# print(df_test_clean.describe())
# print(df_test_clean.info())

categorical_features = ["merchant", "category", "gender", "city", "state", "job"]
numerical_features = [
    "cc_num",
    "amt",
    "lat",
    "long",
    "city_pop",
    "merch_lat",
    "merch_long",
    "hour",
    "day_of_week",
    "month",
]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numerical_features),
        ("cat", OrdinalEncoder(), categorical_features),
    ]
)

X_tr = df_clean.drop("is_fraud", axis=1)
y_tr = df_clean["is_fraud"]

X_ts = df_test_clean.drop("is_fraud", axis=1)
y_ts = df_test_clean["is_fraud"]

X_train_ready = preprocessor.fit_transform(X_tr)
X_test_ready = preprocessor.fit_transform(X_ts)
print(X_tr.columns)

X_train_ready = pd.DataFrame(
    X_train_ready, columns=numerical_features + categorical_features
)
X_test_ready = pd.DataFrame(
    X_test_ready, columns=numerical_features + categorical_features
)

print(X_train_ready.head())
print(X_train_ready.describe())
print(X_train_ready.info())
print(X_test_ready.head())
print(X_test_ready.describe())
print(X_test_ready.info())


# X_train_ready.to_csv("X_train_fraud.csv", sep=",")
# X_test_ready.to_csv("X_test_fraud.csv", sep=",")
# y_tr.to_csv("y_train.csv", sep=",")
# y_ts.to_csv("y_test.csv", sep=",")

keys_dict2 = {
    "X_train": X_train_ready,
    "X_test": X_test_ready,
    "y_train": y_tr,
    "y_test": y_ts,
}
joblib.dump(
    keys_dict2,
    "./data/preprocessed/fraud_preprocessed.pkl",
)
