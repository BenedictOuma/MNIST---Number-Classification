import os
import io
import numpy as np
import tensorflow as tf
from keras.models import load_model
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

#Initializing the FastAPI app
app = FastAPI()

#Defining the path to the MNIST model and the class names
MODEL_PATH = 'mnist_cnn_model.keras'
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

try:
    #Loading the trained Keras model for MNIST
    model = load_model(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    raise

#MNIST has 10 classes (digits 0 to 9)
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

#Preprocessing function for the image with DEBUG
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('L')  # grayscale
    image = image.resize((28, 28))
    
    image_array = np.array(image).astype('float32') / 255.0  # normalize + cast to float32
    
    # If your uploads are white background, black digit → invert
    if np.mean(image_array) > 0.5:  
        image_array = 1.0 - image_array

    print("Preprocessed image shape:", image_array.shape)
    image.save("debug_input.png")
    
    image_array = np.expand_dims(image_array, axis=0)  # batch dim
    image_array = np.expand_dims(image_array, axis=-1) # channel dim
    
    print("Final input shape for model:", image_array.shape)
    return image_array


#Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the MNIST digit classification API. Upload an image to get predictions."}

#Prediction endpoint with DEBUG
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image_array = preprocess_image(contents)
        
        #Running prediction
        predictions = model.predict(image_array)
        print("Model raw predictions:", predictions)

        predicted_class_index = np.argmax(predictions[0])
        predicted_class = class_names[predicted_class_index]
        print("Predicted class index:", predicted_class_index)
        print("Predicted class:", predicted_class)

        return JSONResponse(content={"class": predicted_class})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)