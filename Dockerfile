FROM apache/airflow:2.6.1

# Install AWS CLI and other dependencies
RUN pip install --user --upgrade pip \
    && pip install --no-cache-dir --user awscli \
    && aws --version

# Install additional Python packages
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir --user -r /requirements.txt
RUN pip install Flask boto3
RUN pip install h5py
RUN pip install Flask-WTF

# Set the working directory for your application
WORKDIR /app

# Copy your Flask application code into the image
COPY . /app

# Expose the port on which your Flask app will run (if different from Airflow)
EXPOSE 8081
