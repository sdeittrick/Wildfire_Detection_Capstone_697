#code implemented using the course web.py provided in SIADS 643 with modifications

import pickle

import pandas as pd
from werkzeug.utils import secure_filename

from flask_wtf import FlaskForm, Form
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import StringField
from wtforms.validators import DataRequired
from flask import Flask, request, render_template
from flask_bootstrap import Bootstrap
import tensorflow as tf
import os
import sys
sys.path.append('/'.join(os.getcwd().split('/')[:-1]) + '/wildfire_prediction_pipeline')

from wildfire_detection_predict import predict

app = Flask(__name__)
app.config['SECRET_KEY'] = 'key'  
app.debug = True
Bootstrap(app)
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.gif']
app.config['UPLOAD_PATH'] = 'uploads'

class UploadForm(Form):
    img = FileField('Image', validators=[FileRequired(), FileAllowed(['jpg', 'png'], 'Images only!')])


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = '--'  
    form = UploadForm()

    if form.validate():
        filename = secure_filename(form.img.data.filename)
        form.img.data.save(os.path.join(app.config['UPLOAD_PATH'], filename))

        prediction = predict(form.img.data)

    return render_template(
        "form.html", form=form,
        msg='Predicted Image: ' + str(prediction))


if __name__ == '__main__':
    app.run()
