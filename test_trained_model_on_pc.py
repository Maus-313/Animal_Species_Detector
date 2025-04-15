# This code for testing the code in PC or Laptops.

# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont

# Load the three trained models
model_0 = tf.keras.models.load_model('./models/model_0.keras')
model_1 = tf.keras.models.load_model('./models/model_1.keras')
model_2 = tf.keras.models.load_model('./models/model_2.keras')

# Define species labels
species_labels = ["Elephant", "Gorilla", "Hippo", "Monkey", "Tiger", "Zebra"]

# Function to preprocess a single image
def preprocess_image(img_path, img_height=224, img_width=224):
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    if img_height == 229:  # For InceptionResNetV2
        img_array = tf.keras.applications.inception_resnet_v2.preprocess_input(img_array)
    else:  # For DenseNet201 and MobileNetV2
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

# Function to perform ensemble prediction
def ensemble_predict(models, img_inception, img_other):
    pred_inception = model_0.predict(img_inception)
    pred_densenet = model_1.predict(img_other)
    pred_mobilenet = model_2.predict(img_other)
    predictions = np.array([pred_inception, pred_densenet, pred_mobilenet])
    final_pred = np.argmax(np.sum(predictions, axis=0) / 3, axis=1)
    confidence = np.max(np.sum(predictions, axis=0) / 3)
    return species_labels[final_pred[0]], confidence

# Function to create output image with species name
def create_annotated_image(input_path, output_path, species_name, confidence):
    # Open the input image
    img = Image.open(input_path)
    draw = ImageDraw.Draw(img)
    
    # Use a font (adjust path to a local TTF font file)
    try:
        font = ImageFont.truetype("arial.ttf", 40)  # Example font; replace with your font path
    except:
        font = ImageFont.load_default()
    
    # Add text with species name and confidence
    text = f"{species_name} (Confidence: {confidence:.2f})"
    draw.text((10, 10), text, fill="yellow", font=font)
    
    # Save the annotated image
    img.save(output_path)

# Path to sample images and output directory on your laptop
sample_dir = "./sample_images"  # Create this folder and add sample images
output_dir = "./output_images"
os.makedirs(output_dir, exist_ok=True)

# Test on sample images
for img_file in os.listdir(sample_dir):
    if img_file.endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(sample_dir, img_file)
        
        # Preprocess for InceptionResNetV2 (229x229)
        img_inception = preprocess_image(img_path, img_height=229, img_width=229)
        # Preprocess for DenseNet201 and MobileNetV2 (224x224)
        img_other = preprocess_image(img_path, img_height=224, img_width=224)
        
        # Get prediction
        species_name, confidence = ensemble_predict([model_0, model_1, model_2], img_inception, img_other)
        
        # Create annotated output image
        output_path = os.path.join(output_dir, f"output_{img_file}")
        create_annotated_image(img_path, output_path, species_name, confidence)
        print(f"Processed {img_file}: Predicted {species_name} with confidence {confidence:.2f}")

# Clean up (optional)
model_0 = None
model_1 = None
model_2 = None