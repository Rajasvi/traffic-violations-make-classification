#!/bin/bash

# Download data from 
wget https://s3-us-west-2.amazonaws.com/pcadsassessment/parking_citations.corrupted.csv

pip install -r requirements.txt

# Fit Model
python traffic_violations.py

# Start server on port 5000
python server.py

# Request sample json
python request.py
