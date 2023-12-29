from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import StudentData, PredictPipeline
from sklearn.preprocessing import StandardScaler
from src.logger import logging

application=Flask(__name__)

app=application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        student_data = StudentData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))
        )
        student_data_df = student_data.get_data_as_df()
        logging.info('Before Prediction')
        predict_pipeline = PredictPipeline()
        logging.info('Mid of Prediction')
        results=predict_pipeline.predict(student_data_df)
        logging.info('After Prediction')
        return render_template('home.html',results=results[0])


if __name__=="__main__":
    app.run(host='0.0.0.0')  