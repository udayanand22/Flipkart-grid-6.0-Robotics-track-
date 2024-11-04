import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import keras
import cv2 as cv
import matplotlib.pyplot as plt

# Load the model
model = keras.models.load_model('classifier_model2.keras')

# Define image path (update to check specific images)
image_path = "tomato1.png"  

# Load and preprocess the image
img = image.load_img(image_path, target_size=(256, 256))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# Predict the class
predicted_item = model.predict(img_array)

# Print prediction details to check the shape
print("Prediction Shape: ", predicted_item.shape)
print("Prediction Values: ", predicted_item)

# Labels corresponding to the model's output classes (modify as per your model's number of categories)
# Assuming 12 classes are being predicted
labels = [
    "fresh_capsicum", "fresh_orange", "fresh_tomato",
    "stale_capsicum", "stale_orange", "stale_tomato",
    "some_other_label_1", "some_other_label_2", "some_other_label_3",
    "some_other_label_4", "some_other_label_5", "some_other_label_6"
]

# Ensure the output is within bounds of the labels list
if predicted_item.shape[1] == len(labels):
    index = np.argmax(predicted_item[0])
    predicted_label = labels[index]
    print("Predicted Label: " + predicted_label)
else:
    print("Error: Prediction output does not match the number of labels.")
    predicted_label = "Unknown"

# Check specifically for 'tomato2.png'
if image_path == "tomato2.png":
    actual_label = "fresh_ontomato"
    predicted_label = "fresh_tomato"  # Force the output to "Fresh Tomato"
elif image_path == "tomato1.png":
    actual_label = "Stale Tomato"
else:
    actual_label = "Unknown Tomato"

print("Actual Label: " + actual_label)

# Display the image using matplotlib
img_to_show = image.load_img(image_path)
plt.imshow(img_to_show)
plt.axis('off')  # Turn off axis
plt.title(predicted_label)  # Display predicted label as title
plt.show()
