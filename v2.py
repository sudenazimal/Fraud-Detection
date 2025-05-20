import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def main():
    # 1. Load data
    try:
        data1 = pd.read_csv('creditcard.csv')
        print("Data loaded successfully")
    except FileNotFoundError:
        print("Dataset not found. Make sure 'creditcard.csv' is in your working directory.")
        return

    # 2. Basic info and statistics
    print(data1.describe())
    print(data1.info())

    total_samples = len(data1)
    fraud_percentage = data1["Class"].mean() * 100
    non_fraud_percentage = 100 - fraud_percentage

    print(f"\nTotal samples: {total_samples:,}")
    print(f"Fraud percentage: {fraud_percentage:.4f}%")
    print(f"Non-fraud percentage: {non_fraud_percentage:.4f}%")

    class_counts = data1["Class"].value_counts()
    print("\nAbsolute counts:")
    print(f"Fraud Cases    : {class_counts[1]:,}")
    print(f"Non-Fraud Cases: {class_counts[0]:,}")

    # 3. Prepare features and target
    X = data1.drop("Class", axis=1)
    Y = data1["Class"]

    # 4. Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, random_state=42, stratify=Y
    )

    fraud_prct_train = y_train.mean() * 100
    fraud_prct_test = y_test.mean() * 100

    print("\nIn Split:")
    print(f"Fraud percentage in Train: {fraud_prct_train:.4f}%")
    print(f"Fraud percentage in Test : {fraud_prct_test:.4f}%")
    print(f"Number of frauds in Train: {y_train.value_counts()[1]}")
    print(f"Number of frauds in Test : {y_test.value_counts()[1]}")

    # 5. Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)
    print("\nRandom Forest trained successfully")

    # 6. Predict and evaluate
    y_pred = rf.predict(X_test)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    main()