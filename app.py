from flask import Flask, render_template, request
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow.keras as keras



app = Flask(__name__)
    
model = keras.models.load_model('model1.h5')
l=['Actinic keratosis', 'Basal cell carcinoma', 'Dermatofibroma', 'Melanoma', 'Nevus', 'Pigmented benign keratosis', 'Seborrheic keratosis', 'Squamous cell carcinoma', 'Vascular lesion']
ln=['https://en.wikipedia.org/wiki/Actinic_keratosis','https://en.wikipedia.org/wiki/Basal-cell_carcinoma','https://en.wikipedia.org/wiki/Dermatofibroma','https://en.wikipedia.org/wiki/Melanoma','https://en.wikipedia.org/wiki/Nevus','https://jamanetwork.com/journals/jamadermatology/fullarticle/479104','https://en.wikipedia.org/wiki/Seborrheic_keratosis','https://en.wikipedia.org/wiki/Squamous-cell_carcinoma','https://www.ssmhealth.com/cardinal-glennon/services/pediatric-plastic-reconstructive-surgery/hemangiomas']
def preprocess_image(image):
    processed_image = image.resize((180, 180))
    return processed_image

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded image file
        file = request.files['image']

        # Read and preprocess the image
        img = Image.open(file.stream)
        processed_img = preprocess_image(img)
        processed_img = np.array(processed_img)  # Convert to numpy array

        # Perform prediction
        prediction = model.predict(np.expand_dims(processed_img, axis=0))
        pred = np.argmax(prediction)

        return render_template('index.html', prediction=l[pred], link=ln[pred])
    return render_template('index.html')

@app.route('/about', methods=['GET'])
def custom_page():
    return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=True)