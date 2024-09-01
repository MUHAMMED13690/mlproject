
from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from src.pipeline.predict_pipline import predict_pipline
from src.pipeline.predict_pipline import customdata

app = Flask(__name__)

@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        return str(e), 500

@app.route('/predictdata/', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        try:
            return render_template('home.html')
        except Exception as e:
            return str(e), 500
    else:
        try:
            data = customdata(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=float(request.form.get('reading_score')),
                writing_score=float(request.form.get('writing_score'))
            )

            pred_df = data.get_data_as_data_frame()
            print(pred_df)
            print("Before Prediction")

            pipeline = predict_pipline()
            print("Mid Prediction")
            results = pipeline.predict(pred_df)
            print("After Prediction")

            return render_template('home.html', results=results[0])

        except Exception as e:
            print(f"An error occurred during prediction: {e}")
            return str(e), 500

if __name__ == "__main__":
    app.run(debug=True)
