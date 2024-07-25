# Flight Delay Prediction

This project uses Flask and Airflow to create a web application for predicting flight delays based on machine learning models. The models are loaded from AWS S3, and the application provides predictions for both departure and arrival delays.

## Project Structure

- `flask_dag.py`: Defines the Flask web application and Airflow DAG for running the app.
- `docker-compose.yaml`: Configuration for running the application with Docker Compose.
- `Dockerfile`: Dockerfile for building the Airflow and Flask environment.
- `requirements.txt`: Python dependencies required for the project.
- `templates/`: Directory containing HTML templates for the Flask web application.

## Getting Started

### Prerequisites

- Docker
- Docker Compose
- AWS S3 Bucket with the following structure:
  - `your-bucket-name/Origin/`
  - `your-bucket-name/Destination/`
  - Pre-trained models stored in the S3 bucket.

### Setup

1. Clone the repository:
    ```sh
    git clone https://github.com/VickneshB/Airlines_Delay_Prediction.git
    cd Airlines_Delay_Prediction
    ```

2. Create an `.env` file in the project root directory with your AWS credentials:
    ```sh
    AWS_ACCESS_KEY_ID=<Your AWS Access Key>
    AWS_SECRET_ACCESS_KEY=<Your AWS Secret Key>
    ```

3. Build and start the Docker containers:
    ```sh
    docker-compose up -d --build
    ```

### Accessing the Application

- Flask Web Application: `http://localhost:8081`
- Airflow Web UI: `http://localhost:8082`

### Using the Application

1. Open the Flask web application in your browser.
2. Fill out the form with the required flight information.
3. Click the `Submit` button to get the predicted delay.
4. If the models are loaded and the input data is valid, the application will display the predicted delay.

### File Descriptions

#### `flask_dag.py`

Defines the Flask application and Airflow DAG for running the app. Key components:
- `app = Flask(__name__)`: Initializes the Flask application.
- `@app.route('/')`: Renders the main form for user input.
- `@app.route('/predict', methods=['POST'])`: Handles form submissions and returns the predicted delay.
- `DAG('prediction_dag')`: Defines the Airflow DAG for running the Flask app.

#### `docker-compose.yaml`

Defines the Docker services for the project:
- `postgres`: PostgreSQL database for Airflow.
- `airflow-webserver`: Airflow web server.
- `airflow-scheduler`: Airflow scheduler.
- `airflow-init`: Initializes the Airflow database.
- `my-flask-app`: Runs the Flask application.

#### `Dockerfile`

Defines the Docker image for the project:
- Based on `apache/airflow:2.6.1`.
- Installs required Python packages.
- Copies the application files into the Docker image.

### Contributing

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/fooBar`).
3. Commit your changes (`git commit -am 'Add some fooBar'`).
4. Push to the branch (`git push origin feature/fooBar`).
5. Create a new Pull Request.

### License

This project is licensed under the MIT License.
