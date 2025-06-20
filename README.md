# end-to-end-churn-prediction-model-with-scikit-learn
An end-to-end Machine Learning project that predicts whether a telecom customer is likely to churn (i.e., discontinue the service). This project involves handling real-world data, exploratory data analysis (EDA), model building, and prediction — all powered by Python, Pandas, scikit-learn, MySQL, and SQLAlchemy.

## Project Highlights
✅ Imports data from MySQL database using SQLAlchemy

✅ Performs data cleaning, label encoding, and exploratory data analysis

✅ Trains a Random Forest Classifier to predict churn

✅ Evaluates model using accuracy, confusion matrix, and classification report

✅ Accepts real-time input for churn prediction using model inference

✅ Stores and reuses LabelEncoders for consistent input preprocessing

## Tech Stack
1. Language: Python 3

2. Data Handling: pandas, NumPy

3. Visualisation: seaborn, matplotlib

4. Machine Learning: scikit-learn (RandomForestClassifier)

5. Database: MySQL

6. ORM: SQLAlchemy

7. Model Evaluation: Accuracy

## Requirements

1. Code editors like PyCharm, VS Code, etc.
   
2. Installation of libraries:
>pip install pandas numpy seaborn matplotlib scikit-learn sqlalchemy pymysql

3. MySQL Workbench latest version.
   
## How to run

1. Open MySQL_script.sql in Workbench. And follow instructions to run commands line-by-line.

2. Open Main.py in the Code Editor and run.

## Graphs explained

### Churn distribution
   
![churn distribution](https://github.com/user-attachments/assets/ca0c9e13-541b-44a5-8310-cd44cba372c0)

This bar chart shows the distribution of customers who churned vs. those who didn't.

- It helps identify whether the dataset is balanced or imbalanced.

- In most telco churn datasets, more customers did not churn than those who did.

- This insight is crucial for model evaluation — we should not rely on accuracy alone if the dataset is imbalanced.

### Correlation heatmap

![correlation matrix](https://github.com/user-attachments/assets/6bbbc277-d7fd-4417-a3f6-fc82b0664182)

This heatmap visualises the correlation between all numerical features in the dataset.

- Darker colours and high absolute values (closer to +1 or -1) show strong relationships.

- For example, MonthlyCharges and TotalCharges often show high correlation.

- Churn is also included, so we can spot which features are positively or negatively associated with churn.
