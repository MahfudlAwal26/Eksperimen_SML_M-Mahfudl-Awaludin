import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(input_path, output_path="preprocessed_data.csv"):
    # Load dataset
    df = pd.read_csv(input_path)

    # ======================
    # 1. HANDLE MISSING VALUES
    # ======================
    num_cols = df.select_dtypes(include=['int64','float64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns

    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # ======================
    # 2. REMOVE DUPLICATES
    # ======================
    df = df.drop_duplicates()

    # ======================
    # 3. ENCODING
    # ======================
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    # ======================
    # 4. SCALING
    # ======================
    scaler = StandardScaler()

    feature_cols = df.drop(columns=['Exam_Score']).columns
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # ======================
    # 5. SAVE OUTPUT
    # ======================
    df.to_csv(output_path, index=False)

    return df


# Supaya bisa langsung dijalankan
if __name__ == "__main__":
    preprocess_data("StudentPerformanceFactors.csv")