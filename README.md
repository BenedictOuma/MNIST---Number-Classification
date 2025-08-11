# MNIST Digit Classification API

A FastAPI-based REST API for classifying handwritten digit images (0–9) using a Convolutional Neural Network (CNN) trained on the MNIST dataset.

## Features
- **28x28 grayscale MNIST digit classification**
- **Real-time prediction** via `/predict/` endpoint
- **Debug mode**: saves preprocessed images and logs shapes/predictions
- **Model trained in TensorFlow/Keras**

---

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/BenedictOuma/MNIST---Number-Classification.git
````

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate      #On macOS/Linux
venv\Scripts\activate         #On Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

---

## Training the Model

If you don’t already have `mnist_cnn_model.keras`, train it using the provided Jupyter Notebook:

1. Open `notebook_training.ipynb` in Jupyter or VSCode.
2. Run all cells — it will:

   * Load & preprocess MNIST dataset
   * Train the CNN model
   * Save the best model as `mnist_cnn_model.keras`

---

## Running the API

Start the FastAPI server:

```bash
uvicorn main:app --reload
```

By default, the API runs at:

```
http://127.0.0.1:8000
```

---

## API Endpoints

### **Root Endpoint**

```http
GET /
```

**Response**

```json
{
    "message": "Welcome to the MNIST digit classification API. Upload an image to get predictions."
}
```

### **Prediction Endpoint**

```http
POST /predict/
```

**Parameters**

* `file`: Image file (PNG/JPG) of a handwritten digit.

**Example using `curl`:**

```bash
curl -X POST "http://127.0.0.1:8000/predict/" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@digit.png"
```

**Response**

```json
{
    "class": "4"
}
```

---

## Image Requirements

* Must be **28x28 pixels**
* **Grayscale**
* Digit should be white on black background (like MNIST dataset)
* If using your own images, you can:

  * Resize & grayscale before upload
  * Or modify `preprocess_image()` in `main.py` to auto-invert colors

---

## Debugging

* Preprocessed images are saved as `debug_input.png` in the project directory
* Console logs show:

  * Preprocessed image shape
  * Final model input shape
  * Raw model predictions

---

## Requirements

See `requirements.txt`:

```
fastapi
uvicorn
tensorflow
pillow
numpy
```

---

## License

This project is open-source and available under the MIT License.

---

## 👨Author

**Thee Intellect**
[bogutu027@gmail.com](mailto:bogutu027@gmail.com)