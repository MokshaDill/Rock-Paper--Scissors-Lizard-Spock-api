from PIL import Image
import numpy as np
from fastapi import FastAPI, File, UploadFile
from io import BytesIO
import tensorflow as tf

app = FastAPI()

# Load your trained model
model = tf.keras.models.load_model('artifacts\models\model.h5')

# Define the class labels for your gesture prediction
class_labels = ["Rock", "Paper", "Scissors"]

# Preprocessing function to resize and normalize the image
def preprocess_image(image):
    image = image.resize((150, 150))  # Resize to 150x150 pixels
    image_array = np.array(image) / 255.0  # Normalize the pixel values to [0, 1]
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")

    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_label = class_labels[np.argmax(prediction)]

    return {"gesture": predicted_label}
