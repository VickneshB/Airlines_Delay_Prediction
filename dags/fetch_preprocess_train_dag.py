import pandas as pd
import boto3
from io import BytesIO
from airflow import DAG
from airflow.operators.python_operator import PythonOperator, BranchPythonOperator
from datetime import datetime, timedelta
import h5py
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from joblib import dump, load
import logging


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 7, 15),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

s3_bucket = 'atlantadataset1'
s3_prefix_origin = 'Origin/'
s3_prefix_destination = 'Destination/'
s3_key_org = 'combined_org_data.csv'
s3_key_dest = 'combined_dest_data.csv'
preprocessed_s3_key_org = 'org_data_preprocessed.csv'
preprocessed_s3_key_dest = 'dest_data_preprocessed.csv'

def save_file_count_to_s3(bucket_name, key, count):
    s3 = boto3.client('s3')
    s3.put_object(Body=str(count), Bucket=bucket_name, Key=key)

def load_file_count_from_s3(bucket_name, key):
    s3 = boto3.client('s3')
    try:
        obj = s3.get_object(Bucket=bucket_name, Key=key)
        count = int(obj['Body'].read().decode('utf-8'))
        return count
    except s3.exceptions.NoSuchKey:
        return None

def check_for_new_files(bucket_name, prefix, file_count_key):
    s3 = boto3.client('s3')
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    file_list = [obj['Key'] for obj in response.get('Contents', [])]

    current_number_of_files = len(file_list)
    number_of_files = load_file_count_from_s3(bucket_name, file_count_key)
    if number_of_files is None or current_number_of_files != number_of_files:
        number_of_files = current_number_of_files
        save_file_count_to_s3(bucket_name, file_count_key, number_of_files)
        if prefix == s3_prefix_origin:
            return 'fetch_combine_org'
        elif prefix == s3_prefix_destination:
            return 'fetch_combine_dest'
    else:
        if prefix == s3_prefix_origin:
            return 'no_new_files_org'
        elif prefix == s3_prefix_destination:
            return 'no_new_files_dest'

def fetch_and_combine_data(bucket_name, prefix, s3_key):
    s3 = boto3.client('s3')
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    file_list = [obj['Key'] for obj in response.get('Contents', [])]

    if len(file_list) > 0:
        combined_df = pd.DataFrame()
        for file in file_list:
            obj = s3.get_object(Bucket=bucket_name, Key=file)
            file_content = obj['Body'].read()
            df = pd.read_csv(BytesIO(file_content))
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        
        combined_df_csv = combined_df.to_csv(index=False)
        s3.put_object(Body=combined_df_csv.encode('utf-8'), Bucket=bucket_name, Key=s3_key)
        
        print(f"Combined file uploaded to S3: {s3_key}")
        return True
    else:
        print(f"No new files found in {prefix}")
        return False

def custom_preprocessing_steps(data, data_type):
    if data_type == 'origin':
        keep = 'Dest'
    else:
        keep = 'Origin'
    columns_to_keep = [
        'ActualElapsedTime', 'AirTime', 'ArrDelay', 'DepDelay', 'CRSArrTime', 'CRSDepTime', 
        'CRSElapsedTime', 'Cancelled', 'DayOfWeek', 'DayofMonth', 'DepTime', keep, 
        'Distance', 'FlightNum', 'Month', 'UniqueCarrier', 'Year'
    ]
    data = data[columns_to_keep]
    data['CRSArrTime'] = data['CRSArrTime'].apply(lambda x: '0' if not str(x).isdigit() else x).fillna(0).apply(lambda x: f"{int(x):04d}")
    data['CRSArrTime_hrs'] = data['CRSArrTime'].str[:-2].astype(int)
    data['CRSArrTime_mins'] = data['CRSArrTime'].str[-2:].astype(int)
    data.drop('CRSArrTime', axis=1, inplace=True)

    data['CRSDepTime'] = data['CRSDepTime'].apply(lambda x: '0' if not str(x).isdigit() else x).fillna(0).apply(lambda x: f"{int(x):04d}")
    data['CRSDepTime_hrs'] = data['CRSDepTime'].str[:-2].astype(int)
    data['CRSDepTime_mins'] = data['CRSDepTime'].str[-2:].astype(int)
    data.drop('CRSDepTime', axis=1, inplace=True)

    data.dropna(inplace=True)

    return data

def preprocess_combined_data(s3_key, preprocessed_s3_key):
    s3 = boto3.client('s3')
    combined_obj = s3.get_object(Bucket=s3_bucket, Key=s3_key)
    combined_data = pd.read_csv(combined_obj['Body'])
    
    try:
        if preprocessed_s3_key == 'org_data_preprocessed.csv':
            data_type = 'origin'
        else:
            data_type = 'dest'
        combined_data = custom_preprocessing_steps(combined_data, data_type)
        combined_data_csv = combined_data.to_csv(index=False)
        s3.put_object(Body=combined_data_csv.encode('utf-8'), Bucket=s3_bucket, Key=preprocessed_s3_key)
        print(f"Combined data preprocessed and saved to S3: {preprocessed_s3_key}")
        return True
    except Exception as e:
        print(f"Error in preprocessing combined data: {str(e)}")
        return False

def train_for_dep_pred(preprocessed_s3_key_org):
    s3 = boto3.client('s3')

    
    org_obj = s3.get_object(Bucket=s3_bucket, Key=preprocessed_s3_key_org)
    org_data = pd.read_csv(org_obj['Body'])
    X_org = org_data.drop(columns=['DepDelay', 'ArrDelay'])  
    y_org = org_data['DepDelay']

    mlp_reg = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)

    categorical_features = ['Dest', 'UniqueCarrier']
    numerical_features = [col for col in X_org.columns if col not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='mean'), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('mlp', mlp_reg)
    ])

    X_train_org, X_temp_org, y_train_org, y_temp_org = train_test_split(X_org, y_org, test_size=0.3, random_state=42)
    X_val_org, X_test_org, y_val_org, y_test_org = train_test_split(X_temp_org, y_temp_org, test_size=0.5, random_state=42)

    logging.info(f"Features used for training: {X_train_org.columns.tolist()}")
    pipeline.fit(X_train_org, y_train_org)
    y_pred_dep = pipeline.predict(X_test_org)
    y_pred_dep = y_pred_dep.round().astype(int)
    
    mse_dep = mean_squared_error(y_test_org, y_pred_dep)
    rmse_dep = mean_squared_error(y_test_org, y_pred_dep, squared=False)
    r2_dep = r2_score(y_test_org, y_pred_dep)
    logging.info(f"Test Error metrics for Departure Delay Regression: MSE: {mse_dep}, RMSE: {rmse_dep}, r2: {r2_dep}")

    model_dep_key = 'departure_delay_regressor.joblib'
    dump(pipeline, model_dep_key)
    
    s3.upload_file(model_dep_key, s3_bucket, model_dep_key)
    print(f"Departure Delay Regression model saved to S3: {model_dep_key}")

def train_for_arr_pred(preprocessed_s3_key_dest):
    s3 = boto3.client('s3')

    dest_obj = s3.get_object(Bucket=s3_bucket, Key=preprocessed_s3_key_dest)
    dest_data = pd.read_csv(dest_obj['Body'])
    X_dest = dest_data.drop(columns=['ArrDelay'])  
    y_dest = dest_data['ArrDelay']
    
    mlp_reg = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    
    categorical_features = ['Origin', 'UniqueCarrier']
    numerical_features = [col for col in X_dest.columns if col not in categorical_features]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='mean'), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('mlp', mlp_reg)
    ])

    X_train_dest, X_temp_dest, y_train_dest, y_temp_dest = train_test_split(X_dest, y_dest, test_size=0.3, random_state=42)
    X_val_dest, X_test_dest, y_val_dest, y_test_dest = train_test_split(X_temp_dest, y_temp_dest, test_size=0.5, random_state=42)
    
    logging.info(f"Features used for training: {X_train_dest.columns.tolist()}")
    pipeline.fit(X_train_dest, y_train_dest)
    y_pred_arr = pipeline.predict(X_test_dest)

    y_pred_arr = y_pred_arr.round().astype(int)

    mse_arr = mean_squared_error(y_test_dest, y_pred_arr)
    rmse_arr = mean_squared_error(y_test_dest, y_pred_arr, squared=False)
    r2_arr = r2_score(y_test_dest, y_pred_arr)
    logging.info(f"Test Error metrics for Arrival Delay Regression: MSE: {mse_arr}, RMSE: {rmse_arr}, r2: {r2_arr}")
    
    model_arr_key = 'arrival_delay_regressor.joblib'
    dump(pipeline, model_arr_key)
    
    s3.upload_file(model_arr_key, s3_bucket, model_arr_key)
    print(f"Arrival Delay Regression model saved to S3: {model_arr_key}")


with DAG('training_dag', default_args=default_args, schedule_interval=timedelta(minutes=30), catchup=False, max_active_runs=1) as dag:
    
    check_new_files_Origin = BranchPythonOperator(
        task_id='check_new_files_Origin',
        python_callable=check_for_new_files,
        op_args=[s3_bucket, s3_prefix_origin, 'number_of_files_org.txt']
    )
    
    no_new_files_org = PythonOperator(
        task_id='no_new_files_org',
        python_callable=lambda: print("No new files in Origin.")
    )
    
    fetch_combine_org = PythonOperator(
        task_id='fetch_combine_org',
        python_callable=fetch_and_combine_data,
        op_args=[s3_bucket, s3_prefix_origin, s3_key_org]
    )
    
    preprocess_org = PythonOperator(
        task_id='preprocess_org',
        python_callable=preprocess_combined_data,
        op_args=[s3_key_org, preprocessed_s3_key_org]
    )

    train_dep_model = PythonOperator(
        task_id='train_dep_model',
        python_callable=train_for_dep_pred,
        op_args=[preprocessed_s3_key_org]
    )
    
    check_new_files_Destination = BranchPythonOperator(
        task_id='check_new_files_Destination',
        python_callable=check_for_new_files,
        op_args=[s3_bucket, s3_prefix_destination, 'number_of_files_dest.txt']
    )
    
    no_new_files_dest = PythonOperator(
        task_id='no_new_files_dest',
        python_callable=lambda: print("No new files in Destination.")
    )
    
    fetch_combine_dest = PythonOperator(
        task_id='fetch_combine_dest',
        python_callable=fetch_and_combine_data,
        op_args=[s3_bucket, s3_prefix_destination, s3_key_dest]
    )
    
    preprocess_dest = PythonOperator(
        task_id='preprocess_dest',
        python_callable=preprocess_combined_data,
        op_args=[s3_key_dest, preprocessed_s3_key_dest]
    )
    
    train_arr_model = PythonOperator(
        task_id='train_arr_model',
        python_callable=train_for_arr_pred,
        op_args=[preprocessed_s3_key_dest]
    )

    check_new_files_Origin >> [fetch_combine_org, no_new_files_org]
    fetch_combine_org >> preprocess_org >> train_dep_model
    
    check_new_files_Destination >> [fetch_combine_dest, no_new_files_dest]
    fetch_combine_dest >> preprocess_dest >> train_arr_model
