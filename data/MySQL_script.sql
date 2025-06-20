-- execute this before running mysql_data_dumper.py
CREATE DATABASE IF NOT EXISTS churn_db;

-- execute this after running mysql_data_dumper.py and creating database.
USE churn_db;
select * from customer_churn;