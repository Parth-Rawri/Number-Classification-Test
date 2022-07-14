import os
import numpy as np

from tensorflow import keras
from tensorflow.keras.preprocessing import image
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH ='model_weights.h5'

# Load your trained model
model = keras.models.load_model(MODEL_PATH)


def model_predict(img_path, model):
    # Preprocessing the image
    img = image.load_img(img_path, target_size=(28, 28)).convert("L")
    x = image.img_to_array(img)
    x = x/255 # Scaling
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    preds=np.argmax(preds, axis=1)
    if preds==0:
        preds="The Number is 0"
    elif preds==1:
        preds="The Number is 1"
    elif preds==2:
        preds="The Number is 2"
    elif preds==3:
        preds="The Number is 3"
    elif preds==4:
        preds="The Number is 4"
    elif preds==5:
        preds="The Number is 5"
    elif preds==6:
        preds="The Number is 6"
    elif preds==7:
        preds="The Number is 7"
    elif preds==8:
        preds="The Number is 8"
    else:
        preds="The Number is 9"
    return preds


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html') # Main page


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file'] # Get the file from post request

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)