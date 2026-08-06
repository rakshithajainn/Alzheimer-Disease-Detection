from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename
from flask import send_from_directory
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = load_model("my_model.h5")

class_labels = {
    0: "Mild Dementia",
    1: "Moderate Dementia",
    2: "Non Dementia",
    3: "Very Mild Dementia"
}

def predict_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)

    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    class_id = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    return class_labels[class_id], confidence


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return "No file uploaded"

    file = request.files["file"]

    filename = secure_filename(file.filename)

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    file.save(filepath)

    prediction, confidence = predict_image(filepath)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=round(confidence, 2),
        image=filename
    )


if __name__ == "__main__":
    app.run(debug=True)