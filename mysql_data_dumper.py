# code to be run once that will take data from 'telco_customer_churn.csv' and dump to mysql server

import pandas as pd
from sqlalchemy import create_engine

# Load the CSV data
csv_file_path = 'data/telco_customer_churn.csv'  # make sure this file exists in your working directory
df = pd.read_csv(csv_file_path)

# clean the CSV data
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

# Format: mysql+pymysql://<username>:<password>@<host>/<database>
engine = create_engine("mysql+pymysql://root:Root123root@localhost:3306/churn_db")

# Upload data to MySQL
df.to_sql('customer_churn', con=engine, if_exists='replace', index=False)

print("✅ Data successfully uploaded into MySQL")
