from flask import Flask, request, render_template
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('form.html')
    else:
        data = CustomData(
            Age=float(request.form.get('Age')),
            Education_Number_of_Years=float(request.form.get('Education_Number_of_Years')),
            Final_Weight=float(request.form.get('Final_Weight')),
            Capital_gain=float(request.form.get('Capital_gain')),
            Capital_loss=float(request.form.get('Capital_loss')),
            Hours_per_week=float(request.form.get('Hours_per_week')),
            Workclass=request.form.get('Workclass'),
            Education=request.form.get('Education'),
            Marital_status=request.form.get('Marital_status'),
            Occupation=request.form.get('Occupation'),
            Relationship=request.form.get('Relationship'),
            Race=request.form.get('Race'),
            Sex=request.form.get('Sex'),
            Native_country=request.form.get('Native_country')
        )

        final_new_data = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(final_new_data)

        result = pred[0]
        output = ""
        if result == 0.0:
            output = "Earn Less than 50K"
        else:
            output = "Earn More than 50K"

        return render_template('results.html', final_result=output)


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
