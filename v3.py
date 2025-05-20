import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def main():
    # 1. Load data
    try:
        df = pd.read_csv('fraudTest.csv')
        print("Data loaded successfully")
    except FileNotFoundError:
        print("Dataset not found. Make sure 'fraudTest.csv' is in your working directory.")
        return

    print("Class distribution:")
    print(df['is_fraud'].value_counts())

    # 2. Convert datetime columns to features
    for col in df.select_dtypes(include='object').columns:
        try:
            df[col] = pd.to_datetime(df[col])
            print(f"Parsed datetime column: {col}")
            df[f'{col}_hour'] = df[col].dt.hour
            df[f'{col}_dayofweek'] = df[col].dt.dayofweek
            df[f'{col}_day'] = df[col].dt.day
            df[f'{col}_month'] = df[col].dt.month
            df[f'{col}_is_weekend'] = df[col].dt.dayofweek >= 5
            df = df.drop(columns=[col])  # Drop original datetime column
        except (ValueError, TypeError):
            pass  # Skip columns that aren't datetime strings

    # 3. Drop remaining non-numeric columns
    non_numeric_cols = df.select_dtypes(include=['object']).columns
    if len(non_numeric_cols) > 0:
        print(f"Dropping non-numeric columns: {list(non_numeric_cols)}")
        df = df.drop(columns=non_numeric_cols)

    # 4. Split into features and target
    X = df.drop('is_fraud', axis=1)
    y = df['is_fraud']

    # 5. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42)

    print(f"\nTraining set size: {len(X_train)}, Test set size: {len(X_test)}")

    # 6. Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    print("\nRandom Forest trained successfully")

    # 7. Predict and evaluate
    y_pred = rf.predict(X_test)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    main()