from flask import Flask, render_template, request, redirect
from prediction import *

app = Flask(__name__)


@app.route('/', methods=["GET", "POST"])
def index():
    if request.method == "POST":
        error = ""
        form = request.form
        age = str(form['age'])
        careerLength = str(form['careerLength'])
        noiseExposureLevel = str(form['noiseExposureLevel'])
        smoking = str(form['smoking'])
        if age.isnumeric() == False or careerLength.isnumeric() == False or noiseExposureLevel.isnumeric() == False:
            error = "Please enter a valid value"
            return render_template('index.html', display="", error=error)
        else:
            hearingLossPredict = pipeline.predict(
                np.array([[age, careerLength, noiseExposureLevel, 14, 51, 95, 81, smoking]]))
            return render_template('index.html', display=format(hearingLossPredict[0], '.2f'), error="", hearingLossPredictVal = hearingLossPredict)
    else:
        return render_template('index.html', display="")


if __name__ == '__main__':
    app.run(debug=True)
