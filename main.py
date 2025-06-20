import pandas as pd
import sqlalchemy as sq
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --------------------- import data from mysql ---------------------

engine = sq.create_engine('mysql+pymysql://root:Root123root@localhost:3306/churn_db')
df = pd.read_sql_table('customer_churn', engine)
print(df.head())

# --------------------------- Data cleaning -------------------------
df.dropna(inplace=True)
df.drop("customerID", axis=1, inplace=True)

encoders = {}  # Save encoders here BEFORE converting anything

for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    le.fit(df[col])
    df[col] = le.transform(df[col])
    encoders[col] = le  # Store the encoder for later use in prediction

# ------------------------------- EDA -------------------------------

# Churn Count : to check how many customer stopped using the companies product or service
sns.countplot(x='Churn', data=df)
plt.title("Churn Distribution")
plt.show()

# Correlation Heatmap : shows relationship of the every parameter with each other.
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), cmap='coolwarm', annot=True, fmt='.2f')
plt.title("Feature Correlation")
plt.show()

# ------------------------------ Model ------------------------------

# Split
X = df.drop("Churn", axis=1)  # All columns except the target
y = df["Churn"]               # Only the target column
feature_order = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate
print("Model accuracy:", accuracy_score(y_test, y_pred))

# ---------------------------- Prediction ----------------------------

# just an example of a customer to predict will he stop being a user of the company or not('churn' or 'no churn')
input_dict = {
    'gender': 'Female',
    'SeniorCitizen': 1,
    'Partner': 'No',
    'Dependents': 'No',
    'tenure': 1,
    'PhoneService': 'Yes',
    'MultipleLines': 'No',
    'InternetService': 'Fiber optic',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'No',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'Yes',
    'StreamingMovies': 'Yes',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 95.7,
    'TotalCharges': 95.7
}

input_df = pd.DataFrame([input_dict])

# Encode categorical columns using stored encoders
for col in input_df.columns:
    if input_df[col].dtype == 'object':
        if col in encoders:
            input_df[col] = encoders[col].transform(input_df[col])
        else:
            raise ValueError(f"No encoder found for column: {col}")

# Reorder columns to match training data
input_df = input_df[feature_order]

# Predict
prediction = model.predict(input_df)[0]
print("Prediction (1 = churn, 0 = no churn):", prediction)
