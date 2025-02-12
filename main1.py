import tkinter as tk
from tkinter import filedialog, Label, Button
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

def classify_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return
    
    image = Image.open(file_path).convert("RGB")
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    
    # Predict the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]
    
    result_label.config(text=f"Class: {class_name}\nConfidence Score: {confidence_score:.2f}")

# GUI setup
root = tk.Tk()
root.title("Image Classification GUI")
root.geometry("400x300")

title_label = Label(root, text="Keras Image Classifier", font=("Arial", 14))
title_label.pack(pady=10)

select_button = Button(root, text="Select Image", command=classify_image)
select_button.pack(pady=10)

result_label = Label(root, text="", font=("Arial", 12))
result_label.pack(pady=20)

root.mainloop()
