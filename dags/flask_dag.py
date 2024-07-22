from flask import Flask, request, jsonify, render_template, flash
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import boto3
from joblib import load  
import pandas as pd
import numpy as np
from io import BytesIO
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

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

s3 = boto3.client('s3')

our_airport = ['ATL']
other_airports = ['JFK', 'LAX', 'SFO', 'ORD']

model_features = {
    'departure': [
        'ActualElapsedTime', 'AirTime',
        'CRSElapsedTime', 'Cancelled', 'DayOfWeek', 'DayofMonth',
        'DepTime', 'Dest', 'Distance', 'FlightNum', 'Month',
        'UniqueCarrier', 'Year', 'CRSArrTime_hrs',  'CRSArrTime_mins', 'CRSDepTime_hrs', 'CRSDepTime_mins'
    ],
    'arrival': [
        'ActualElapsedTime', 'AirTime', 'DepDelay',
        'CRSElapsedTime', 'Cancelled', 'DayOfWeek', 'DayofMonth',
        'ArrTime', 'Dest', 'Distance', 'FlightNum', 'Month',
        'UniqueCarrier', 'Year', 'CRSArrTime_hrs',  'CRSArrTime_mins', 'CRSDepTime_hrs', 'CRSDepTime_mins'
    ]
}

def preprocess_input_data(features, prediction_type):
    features['CRSArrTime'] = features['CRSArrTime'].apply(lambda x: '0' if not str(x).isdigit() else x).fillna(0).apply(lambda x: f"{int(x):04d}")
    features['CRSArrTime_hrs'] = features['CRSArrTime'].str[:-2].astype(int)
    features['CRSArrTime_mins'] = features['CRSArrTime'].str[-2:].astype(int)
    features.drop('CRSArrTime', axis=1, inplace=True)

    features['CRSDepTime'] = features['CRSDepTime'].apply(lambda x: '0' if not str(x).isdigit() else x).fillna(0).apply(lambda x: f"{int(x):04d}")
    features['CRSDepTime_hrs'] = features['CRSDepTime'].str[:-2].astype(int)
    features['CRSDepTime_mins'] = features['CRSDepTime'].str[-2:].astype(int)
    features.drop('CRSDepTime', axis=1, inplace=True)
    
    if prediction_type == 'departure':
        features = features.drop(columns=['ArrTime'])
    else:
        features = features.drop(columns=['DepTime'])
            
    return features

def load_models_from_s3():
    try:
        
        departure_model_key = 'departure_delay_regressor.joblib'
        departure_model = load(BytesIO(s3.get_object(Bucket=s3_bucket, Key=departure_model_key)['Body'].read()))
                
        arrival_model_key = 'arrival_delay_regressor.joblib'
        arrival_model = load(BytesIO(s3.get_object(Bucket=s3_bucket, Key=arrival_model_key)['Body'].read()))
        
        return departure_model, arrival_model
    except Exception as e:
        print(f"Failed to load models from S3: {e}")
        raise

app = Flask(__name__)
app.secret_key = 'supersecretkey'

@app.route('/')
def index():
    return render_template('index.html', our_airport=our_airport[0], other_airports=other_airports)

@app.route('/predict', methods=['POST'])
def predict():
    s3_key_org = 'combined_org_data.csv'
    s3_key_dest = 'combined_dest_data.csv'

    s3 = boto3.client('s3')
    combined_org = s3.get_object(Bucket=s3_bucket, Key=s3_key_org)
    combined_org_data = pd.read_csv(combined_org['Body'])

    combined_dest = s3.get_object(Bucket=s3_bucket, Key=s3_key_dest)
    combined_dest_data = pd.read_csv(combined_dest['Body'])
    
    try:
        prediction_type = request.form.get('prediction_type')
        origin = request.form.get('origin')
        destination = request.form.get('destination')
        flight_number = request.form.get('flight_number')
        carrier_code = request.form.get('carrier_code')

        actual_time = request.form.get('actual_time')
        flight_date = request.form.get('flight_date')
        flash("Processing.... The Results should be ready in a few seconds", "info")

        app.logger.info(f"Received form data: prediction_type={prediction_type}, origin={origin}, destination={destination}, flight_number={flight_number}, actual_time={actual_time}, flight_date={flight_date}")

        if not flight_number.isdigit():
            return render_template('error.html', message="Flight number must be numeric.")

        if not actual_time.isdigit() or len(actual_time) != 4:
            return render_template('error.html', message="Actual arrival/departure time must be a 4-digit number.")

        try:
            date_object = datetime.strptime(flight_date, '%Y-%m-%d')
        except ValueError:
            return render_template('error.html', message="Invalid date format. Please use YYYY-MM-DD.")

        day = date_object.day
        month = date_object.month
        year = date_object.year
        day_of_week = date_object.weekday() + 1  
        flight_number = int(flight_number)
        departure_model, arrival_model = load_models_from_s3()

        if prediction_type == 'departure':
            ml_model = departure_model
            model_input_features = model_features['departure']
        elif prediction_type == 'arrival':
            ml_model = arrival_model
            model_input_features = model_features['arrival']
        else:
            return render_template('error.html', message="Invalid prediction type selected.")
        
        if prediction_type == 'arrival':
            departure_delay = int(request.form.get('departure_delay'))
            features = get_features(flight_number, carrier_code, prediction_type, 
                                    day_of_week, day, month, year, combined_dest_data, 
                                    departure_delay=departure_delay, origin=origin)

            desired_order_list = [ 'ActualElapsedTime', 'AirTime', 'DepDelay', 'CRSElapsedTime', 'Cancelled', 'DayOfWeek', 'DayofMonth', 'DepTime', 'Origin', 'Distance', 'FlightNum', 'Month', 'UniqueCarrier', 'Year', 'CRSArrTime_hrs', 'CRSArrTime_mins', 'CRSDepTime_hrs', 'CRSDepTime_mins']
            features = {k: features[k] for k in desired_order_list}
        else:
            features = get_features(flight_number, carrier_code, prediction_type, 
                                    day_of_week, day, month, year, combined_org_data, 
                                    destination=destination, actual_time=actual_time)

            desired_order_list = ['ActualElapsedTime', 'AirTime', 'CRSElapsedTime', 'Cancelled', 'DayOfWeek', 'DayofMonth', 'DepTime', 'Dest', 'Distance', 'FlightNum', 'Month', 'UniqueCarrier', 'Year', 'CRSArrTime_hrs', 'CRSArrTime_mins', 'CRSDepTime_hrs', 'CRSDepTime_mins']
            features = {k: features[k] for k in desired_order_list}

        
        app.logger.info(f"Input features: {features}")

        features_df = pd.DataFrame([features])
        
        predicted_delay = int(np.round(ml_model.predict(features_df)[0]))
        
        app.logger.info(f"Predicted delay: {predicted_delay}")

        return render_template('result.html', prediction_type=prediction_type, predicted_delay=predicted_delay)
    except Exception as e:
        app.logger.error(f"Error during prediction: {str(e)}")
        return render_template('error.html', message=f"Error during prediction: {str(e)}")


def get_features(flight_number, carrier_code, prediction_type, 
                day_of_week, day, month, year, csv,
                departure_delay=None, origin=None, destination=None, actual_time=None):
    
    filtered_df = csv[(csv['FlightNum'] == flight_number) & (csv['UniqueCarrier'] == carrier_code)]

    if not filtered_df.empty:
        
        CRSArrTime = int(filtered_df['CRSArrTime'].mean())
        CRSDepTime = int(filtered_df['CRSDepTime'].mean())

        CRS_ArrTime_hrs = int(CRSArrTime // 100)
        CRS_ArrTime_min = int(CRSArrTime % 100)
        CRS_DepTime_hrs = int(CRSDepTime // 100)
        CRS_DepTime_min = int(CRSDepTime % 100)
        Distance = int(filtered_df['Distance'].mean())
        CRSElapsedTime = int(filtered_df['CRSElapsedTime'].mean())
        ActualElapsedTime = int(filtered_df['ActualElapsedTime'].mean())
        AirTime = int(filtered_df['AirTime'].mean())

        features = {
            'ActualElapsedTime': ActualElapsedTime,
            'AirTime': AirTime,
            'CRSElapsedTime': CRSElapsedTime,
            'Cancelled': 0,
            'DayOfWeek': day_of_week,
            'DayofMonth': day,
            'Distance': Distance,
            'FlightNum': flight_number,
            'Month': month,
            'UniqueCarrier': carrier_code,
            'Year': year,
            'CRSArrTime_hrs': CRS_ArrTime_hrs,
            'CRSArrTime_mins': CRS_ArrTime_min,
            'CRSDepTime_hrs': CRS_DepTime_hrs,
            'CRSDepTime_mins': CRS_DepTime_min,
        }

        if prediction_type == 'arrival':
            DepTime_hrs = CRS_DepTime_hrs + (departure_delay/60)
            DepTime_min = CRS_DepTime_min + (departure_delay%60)
            DepTime = int((CRS_DepTime_hrs * 60 + CRS_DepTime_min) + (DepTime_hrs * 60 + DepTime_min))
            features.update({
                'DepDelay': departure_delay,
                'DepTime': DepTime,
                'Origin': origin,
            })
        else:
            features.update({
                'DepTime': actual_time,
                'Dest': destination,
            })

    else:
        if origin:
            destination = 'ATL'
        else:
            origin = 'ATL'
        filtered_df_2 = csv[(csv['Origin'] == origin) & (csv['Dest'] == destination)]
        app.logger.info(f"Filtered: {filtered_df_2} {origin} {destination}")
        if filtered_df_2.empty:
            Distance = 0
        else:
            Distance = int(filtered_df_2['Distance'].mean())
        features = {
            'ActualElapsedTime': 0,
            'AirTime': 0,
            'CRSElapsedTime': 0,
            'Cancelled': 0,
            'DayOfWeek': day_of_week,
            'DayofMonth': day,
            'Distance': Distance,
            'FlightNum': flight_number,
            'Month': month,
            'UniqueCarrier': carrier_code,
            'Year': year,
            'CRSArrTime_hrs': 0,
            'CRSArrTime_mins': 0,
            'CRSDepTime_hrs': 0,
            'CRSDepTime_mins': 0,
        }
        if prediction_type == 'arrival':
            features.update({
                'DepDelay': 0,
                'DepTime': 0,
                'Origin': origin,
            })
        else:
            features.update({
                'DepTime': actual_time,
                'Dest': destination,
            })
    
    return features

with DAG('prediction_dag', default_args=default_args, schedule_interval=timedelta(days=1), catchup=False) as dag:
    
    
    def run_flask_app():
        app.run(host='0.0.0.0', port=8081, debug=False)

    
    run_flask_app_task = PythonOperator(
        task_id='run_flask_app',
        python_callable=run_flask_app,
    )

run_flask_app_task