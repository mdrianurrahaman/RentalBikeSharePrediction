
from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.RentalBikeSharePrediction.pipelines.prediction_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # Check if 'temp' and 'windspeed' are present in the form data
        temp_value = request.form.get('temp')
        windspeed_value = request.form.get('windspeed')

        if temp_value is None or windspeed_value is None:
            # Handle the case where 'temp' or 'windspeed' is not present
            return render_template('home.html', error_message="Temperature and windspeed are required.")
        
        data = CustomData(
            season=request.form.get('season'),
            yr=request.form.get('yr'),
            mnth=request.form.get('mnth'),
            hr=request.form.get('hr'),
            weekday=request.form.get('weekday'),
            workingday=request.form.get('workingday'),
            weathersit=request.form.get('weathersit'),
            temp=float(temp_value),
            hum=float(request.form.get('hum')),
            windspeed=float(windspeed_value)  # Convert to float only if not None
        )
        final_data=data.get_data_as_data_frame()
        
        predict_pipeline=PredictPipeline()
        
        pred=predict_pipeline.predict(final_data)
        results = round(pred[0],2)

        return render_template('result.html',final_result=results)
    

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)