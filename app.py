import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

app = Flask(__name__)

# Load the model (use the correct path)
# Correct the path to the model file
model = load_model(r'C:\Users\user\Documents\5th_Sem\ML_LAB\Leaf_Disease_ML_Project\plant_disease_detection-main\model.h5')
 # Update if the model is in a different location
labels = {0: 'Healthy', 1: 'Powdery', 2: 'Rust'}

# Define the uploads folder with the absolute path
uploads_folder = r'C:\Users\user\Documents\5th_Sem\ML_LAB\Leaf_Disease_ML_Project\plant_disease_detection-main\uploads'

# Ensure the 'uploads' folder exists
if not os.path.exists(uploads_folder):
    os.makedirs(uploads_folder)

# Function to preprocess the image
def preprocess_image(image_path, target_size=(225, 225)):
    img = load_img(image_path, target_size=target_size)
    x = img_to_array(img)
    x = x.astype('float32') / 255.
    x = np.expand_dims(x, axis=0)
    return x

# Route to the main page
@app.route('/')
def index():
    return render_template('index.html', prediction=None, image_path=None)

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'})

    image_file = request.files['image']

    if image_file.filename == '':
        return jsonify({'error': 'No selected image file'})

    if image_file:
        # Save the uploaded image in the specified 'uploads' folder
        image_path = os.path.join(uploads_folder, secure_filename(image_file.filename))
        image_file.save(image_path)

        # Preprocess the image and make predictions
        x = preprocess_image(image_path)
        predictions = model.predict(x)[0]
        predicted_label = labels[np.argmax(predictions)]

        return jsonify({'prediction': predicted_label, 'image_path': image_path})

if __name__ == '__main__':
    app.run(debug=True)
