# model_prediction.py

import numpy as np
import tensorflow as tf
from collections import Counter
from tensorflow.keras.models import load_model

# Load your models without loading the optimizer state
model1 = load_model('Plant_disease_Efficient_NetB0.keras', compile=False)
model2 = load_model('trained_plant_disease_model.keras', compile=False)

# Optionally, compile your models if you need to use them for further training
model1.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model2.compile(optimizer='adam', loss='categorical_crossentropy')

# List of class names
class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
              'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
              'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
              'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
              'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
              'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
              'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
              'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
              'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
              'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
              'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
              'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']


def predict_disease(image_path):
    try:
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
    except FileNotFoundError as e:
        return {"error": str(e)}

    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])

    # Make predictions
    pred1 = model1.predict(input_arr)
    pred2 = model2.predict(input_arr)

    # Get the class with the highest predicted probability for each model
    pred1_class = np.argmax(pred1, axis=1)[0]
    pred2_class = np.argmax(pred2, axis=1)[0]

    # Combine predictions and find the most common one
    predictions = [pred1_class, pred2_class]
    common_prediction = Counter(predictions).most_common(1)[0][0]

    # Get the class name for the most common prediction
    model_prediction = class_name[common_prediction]

    # Save the image with the prediction overlay
    plt.imshow(image)
    plt.title(f"Disease Name: {model_prediction}")
    plt.xticks([])
    plt.yticks([])

    output_image_path = os.path.join(os.path.dirname(image_path), "predicted_" + os.path.basename(image_path))
    plt.savefig(output_image_path)
    plt.close()

    return {"prediction": model_prediction, "image_path": output_image_path}
