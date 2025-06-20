-- Execute this before running mysql_data_dumper.py
CREATE DATABASE IF NOT EXISTS churn_db;
-- run mysql_data_dumper.py 

USE churn_db;
select * from customer_churn;
