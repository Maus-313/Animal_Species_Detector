# Import necessary libraries
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.applications import InceptionResNetV2, DenseNet201, MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
import numpy as np
from sklearn.metrics import accuracy_score
import os

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 1. Data Preprocessing


def preprocess_data(image_paths, labels, img_height=224, img_width=224):
    data = []
    for path in image_paths:
        img = tf.keras.preprocessing.image.load_img(
            path, target_size=(img_height, img_width))
        img = tf.keras.preprocessing.image.img_to_array(img)
        if img_height == 229:  # For InceptionResNetV2
            img = tf.keras.applications.inception_resnet_v2.preprocess_input(
                img)
        else:  # For DenseNet201 and MobileNetV2
            img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        data.append(img)
    return np.array(data), np.array(labels)


# Define the base directory in Google Drive
base_dir = "/content/drive/My Drive/animal_dataset/six_species"

# Initialize lists for image paths and labels
image_paths = []
labels = []

# Map species to labels (0 to 5)
species_to_label = {
    "Elephant": 0,
    "Gorilla": 1,
    "Hippo": 2,
    "Monkey": 3,
    "Tiger": 4,
    "Zebra": 5
}

# Populate image_paths and labels
for species, label in species_to_label.items():
    species_dir = os.path.join(base_dir, species)
    for img_file in os.listdir(species_dir):
        # Adjust file extensions as needed
        if img_file.endswith(('.jpg', '.jpeg', '.png')):
            image_paths.append(os.path.join(species_dir, img_file))
            labels.append(label)

# Verify the total number of images and labels
print(f"Total images: {len(image_paths)}")  # Should be around 12000
print(f"Total labels: {len(labels)}")       # Should be around 12000
print("Label distribution:", {k: labels.count(v)
      for k, v in species_to_label.items()})

# Split dataset: 70% train (8400), 10% validation (1200), 20% test (2400)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    image_paths, labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.125, random_state=42)  # 0.125 of 80% = 10%

# Preprocess data for different model input sizes
X_train_inception, y_train_inception = preprocess_data(
    X_train, y_train, img_height=229, img_width=229)
X_val_inception, y_val_inception = preprocess_data(
    X_val, y_val, img_height=229, img_width=229)
X_test_inception, y_test_inception = preprocess_data(
    X_test, y_test, img_height=229, img_width=229)

X_train_other, y_train_other = preprocess_data(
    X_train, y_train, img_height=224, img_width=224)
X_val_other, y_val_other = preprocess_data(
    X_val, y_val, img_height=224, img_width=224)
X_test_other, y_test_other = preprocess_data(
    X_test, y_test, img_height=224, img_width=224)

# 2. Load Pre-trained Models (without top layers)
inception_model = InceptionResNetV2(
    weights='imagenet', include_top=False, input_shape=(229, 229, 3))
densenet_model = DenseNet201(
    weights='imagenet', include_top=False, input_shape=(224, 224, 3))
mobilenet_model = MobileNetV2(
    weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze base layers
inception_model.trainable = False
densenet_model.trainable = False
mobilenet_model.trainable = False

# 3. Add Custom Top Layers


def add_top_layers(base_model, input_shape):
    base_model.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(6, activation='softmax')(x)  # 6 species
    return Model(inputs=base_model.input, outputs=output)


inception_model_with_top = add_top_layers(inception_model, (229, 229, 3))
densenet_model_with_top = add_top_layers(densenet_model, (224, 224, 3))
mobilenet_model_with_top = add_top_layers(mobilenet_model, (224, 224, 3))

# 4. Compile and Train Models
models = [inception_model_with_top,
          densenet_model_with_top, mobilenet_model_with_top]
for model in models:
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

# Train models
history_inception = inception_model_with_top.fit(
    X_train_inception,
    tf.keras.utils.to_categorical(y_train_inception),
    epochs=40,
    batch_size=16,
    validation_data=(
        X_val_inception,
        tf.keras.utils.to_categorical(y_val_inception)
    ),
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)])


history_densenet = densenet_model_with_top.fit(X_train_other, tf.keras.utils.to_categorical(y_train_other),
                                               epochs=40, batch_size=16, validation_data=(X_val_other, tf.keras.utils.to_categorical(y_val_other)),
                                               callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)])
history_mobilenet = mobilenet_model_with_top.fit(X_train_other, tf.keras.utils.to_categorical(y_train_other),
                                                 epochs=40, batch_size=16, validation_data=(X_val_other, tf.keras.utils.to_categorical(y_val_other)),
                                                 callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)])

# 5. Ensemble Voting
def ensemble_predict(models, x_inception, x_other):
    pred_inception = models[0].predict(x_inception)
    pred_densenet = models[1].predict(x_other)
    pred_mobilenet = models[2].predict(x_other)
    predictions = np.array([pred_inception, pred_densenet, pred_mobilenet])
    return np.argmax(np.sum(predictions, axis=0) / 3, axis=1)


# 6. Evaluate
y_pred = ensemble_predict(models, X_test_inception, X_test_other)
# Use y_test_inception for consistency
ensemble_accuracy = accuracy_score(y_test_inception, y_pred)
print(f"Ensemble Accuracy: {ensemble_accuracy}")

# Save models to Google Drive
save_dir = "/content/drive/My Drive/models"
os.makedirs(save_dir, exist_ok=True)
for i, model in enumerate(models):
    model.save(os.path.join(save_dir, f'model_{i}.h5'))
