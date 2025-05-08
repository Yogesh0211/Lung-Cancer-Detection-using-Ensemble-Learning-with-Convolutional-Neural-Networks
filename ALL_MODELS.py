#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 10:37:21 2025

@author: yogeshyadav

Lung Cancer Classification Pipeline
This script handles both data preparation and model training for X-ray image classification.

Combined Lung Cancer Detection Pipeline
- Data Preparation
- Model Training
- Weighted Ensemble Evaluation
- Model comparison
- Graph performance
 
"""

import numpy as np
import os
import cv2
import random
import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2, DenseNet201, InceptionV3, Xception, MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from tabulate import tabulate

# ====================== CONFIGURATION ======================
BASE_DATA_DIR = r"C:\Users\najia\Downloads\Sem8\EEE405 Machine Learning for FPGAs\FinalProject\Proj" # Update DATA Path 
TRAIN_DIR = os.path.join(BASE_DATA_DIR, "TRAIN_DIR")
TEST_DIR = os.path.join(BASE_DATA_DIR, "TEST_DIR")
VAL_DIR = os.path.join(BASE_DATA_DIR, "VAL_DIR")
OUTPUT_DIR = os.path.join(BASE_DATA_DIR, "OUTPUT_DIR")
MODEL_SAVE_DIR = os.path.join(BASE_DATA_DIR, "MODELS")
GRAPHS_DIR = os.path.join(BASE_DATA_DIR, "GRAPHS")

# Create directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)

# Data parameters
CLASSES = ["LUNG_CANCER", "NOT_LUNG_CANCER"]
IMAGE_SIZE = 224
BATCH_SIZE = 16  # Reduced batch size to handle memory constraints

# Model parameters
EPOCHS = 50
LEARNING_RATE = 0.0001
MODELS_TO_TRAIN = ['mobilenet_v2', 'xception', 'resnet50_v2', 'densenet201', 'inception_v3']
FORCE_RETRAIN = True

# ====================== DATA PROCESSING ======================
def load_and_process_image(args):
    """Load and preprocess a single image"""
    path, img_size, class_num = args
    try:
        img_array = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_array is None:
            raise ValueError(f"Could not read image {path}")
       
        if len(img_array.shape) == 2:  # Grayscale
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 4:  # RGBA
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
           
        img_array = cv2.resize(img_array, (img_size, img_size))
        return img_array, class_num
    except Exception as e:
        print(f"Skipping {path}: {str(e)}")
        return None

def create_dataset(datadir, categories, img_size):
    """Create dataset from directory"""
    data = []
    for class_num, category in enumerate(categories):
        path = os.path.join(datadir, category)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Directory not found: {path}")
           
        files = [os.path.join(path, f) for f in os.listdir(path)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
       
        print(f"Found {len(files)} images in {category}")
       
        with ThreadPoolExecutor() as executor:
            args = [(f, img_size, class_num) for f in files]
            for result in tqdm(executor.map(load_and_process_image, args),
                             total=len(files),
                             desc=f"Processing {category}"):
                if result is not None:
                    data.append(result)
   
    if not data:
        raise ValueError("No valid images found in the dataset")
   
    random.shuffle(data)
    images, labels = zip(*data)
   
    images = np.array(images).reshape(-1, img_size, img_size, 3)
    images = images / 255.0
    labels = np.array(labels)
   
    return images, labels

# ====================== MODEL TRAINING ======================
def build_model(model_name, input_shape):
    """Build specified model architecture"""
    base_models = {
        'resnet50_v2': ResNet50V2,
        'densenet201': DenseNet201,
        'inception_v3': InceptionV3,
        'xception': Xception,
        'mobilenet_v2': MobileNetV2
    }
   
    base_model = base_models[model_name](
        weights='imagenet',
        include_top=False,
        input_shape=input_shape,
        pooling='avg'
    )
   
    x = base_model.output
    x = tf.keras.layers.Dropout(0.5)(x)
    predictions = tf.keras.layers.Dense(len(CLASSES), activation='softmax')(x)
   
    model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
    model._name = model_name
   
    for layer in base_model.layers:
        layer.trainable = False
   
    return model

def train_model(model_name, train_images, train_labels, val_images, val_labels):
    """Train specified model"""
    # Calculate steps per epoch to ensure proper batching
    steps_per_epoch = len(train_images) // BATCH_SIZE
    if len(train_images) % BATCH_SIZE != 0:
        steps_per_epoch += 1
   
    validation_steps = len(val_images) // BATCH_SIZE
    if len(val_images) % BATCH_SIZE != 0:
        validation_steps += 1
   
    # Calculate class weights
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(train_labels),
        y=train_labels)
    class_weights = dict(enumerate(class_weights))
   
    model = build_model(model_name, (IMAGE_SIZE, IMAGE_SIZE, 3))
   
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
   
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6),
        ModelCheckpoint(
            os.path.join(MODEL_SAVE_DIR, f"{model_name}_best.h5"),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )
    ]
   
    print(f"\nTraining {model_name}...")
    history = model.fit(
        train_images, train_labels,
        validation_data=(val_images, val_labels),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
   
    model.save(os.path.join(MODEL_SAVE_DIR, f"{model_name}.h5"))
    np.save(os.path.join(MODEL_SAVE_DIR, f"{model_name}_history.npy"), history.history)
   
    plot_training_history(history, model_name)
   
    return model, history

# ====================== VISUALIZATION ======================
def plot_training_history(history, model_name):
    """Plot training and validation metrics"""
    plt.figure(figsize=(14, 6))
   
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='green', linewidth=2)
    plt.title(f'{model_name.upper()} - Accuracy', fontsize=14, pad=20)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
   
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', color='red', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange', linewidth=2)
    plt.title(f'{model_name.upper()} - Loss', fontsize=14, pad=20)
    plt.ylabel('Loss', fontsize=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
   
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, f'{model_name}_training.png'), dpi=300, bbox_inches='tight')
    plt.close()
   
from sklearn.metrics import ConfusionMatrixDisplay

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(GRAPHS_DIR, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
    plt.close()
   
# ====================== MODEL COMPARISON ======================

def create_model_comparison(models, histories, test_images, test_labels):
    """Generate comparison table of all models"""
    comparison_data = []

    for model, history in zip(models, histories):
        # Get final training metrics
        train_acc = history.history['accuracy'][-1]
        train_loss = history.history['loss'][-1]

        # Get validation metrics
        val_acc = history.history['val_accuracy'][-1]
        val_loss = history.history['val_loss'][-1]

        # Get test metrics
        test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)

        comparison_data.append([
            model.name,
            f"{train_acc:.2%}",
            f"{train_loss:.4f}",
            f"{val_acc:.2%}",
            f"{val_loss:.4f}",
            f"{test_acc:.2%}",
            f"{test_loss:.4f}"
        ])

    # Print comparison table
    print("\nModel Comparison:")
    print(tabulate(comparison_data,
                   headers=["Model", "Train Acc", "Train Loss", "Val Acc", "Val Loss", "Test Acc", "Test Loss"],
                   tablefmt="grid"))

    # Save to file
    with open(os.path.join(GRAPHS_DIR, 'model_comparison.txt'), 'w') as f:
        f.write(tabulate(comparison_data,
                         headers=["Model", "Train Acc", "Train Loss", "Val Acc", "Val Loss", "Test Acc", "Test Loss"],
                         tablefmt="grid"))


# ====================== MAIN PIPELINE ======================
def main():
    print("Starting classification pipeline...")
   
    # Load or process data
    print("\n=== Loading Data ===")
    try:
        train_images, train_labels = create_dataset(TRAIN_DIR, CLASSES, IMAGE_SIZE)
        val_images, val_labels = create_dataset(VAL_DIR, CLASSES, IMAGE_SIZE)
        test_images, test_labels = create_dataset(TEST_DIR, CLASSES, IMAGE_SIZE)
       
        # Ensure we have complete batches
        train_samples = (len(train_images) // BATCH_SIZE) * BATCH_SIZE
        val_samples = (len(val_images) // BATCH_SIZE) * BATCH_SIZE
        test_samples = (len(test_images) // BATCH_SIZE) * BATCH_SIZE
       
        train_images = train_images[:train_samples]
        train_labels = train_labels[:train_samples]
        val_images = val_images[:val_samples]
        val_labels = val_labels[:val_samples]
        test_images = test_images[:test_samples]
        test_labels = test_labels[:test_samples]
       
        np.save(os.path.join(OUTPUT_DIR, "train_images.npy"), train_images)
        np.save(os.path.join(OUTPUT_DIR, "train_labels.npy"), train_labels)
        np.save(os.path.join(OUTPUT_DIR, "val_images.npy"), val_images)
        np.save(os.path.join(OUTPUT_DIR, "val_labels.npy"), val_labels)
        np.save(os.path.join(OUTPUT_DIR, "test_images.npy"), test_images)
        np.save(os.path.join(OUTPUT_DIR, "test_labels.npy"), test_labels)
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return

    # Train or load models
    print("\n=== Training Models ===")
    trained_models = []
    training_histories = []
   
    for model_name in MODELS_TO_TRAIN:
        model_path = os.path.join(MODEL_SAVE_DIR, f"{model_name}.h5")
       
        if os.path.exists(model_path) and not FORCE_RETRAIN:
            print(f"Loading pre-trained {model_name}")
            model = tf.keras.models.load_model(model_path)
            history = None
        else:
            model, history = train_model(model_name, train_images, train_labels, val_images, val_labels)
       
        trained_models.append(model)
        training_histories.append(history)
   
    # === Ensemble Evaluation ===
   
    print("\n=== Ensemble Evaluation ===")
    accuracies = [model.evaluate(test_images, test_labels, verbose=0)[1] for model in trained_models]
    weights = np.array(accuracies) ** 4
    weights /= weights.sum()

    ensemble_preds = []
    for img in tqdm(test_images, desc="Ensemble Prediction"):
        probs = [model.predict(img[np.newaxis, ...], verbose=0)[0] for model in trained_models]
        avg_prob = np.average(probs, axis=0, weights=weights)
        ensemble_preds.append(np.argmax(avg_prob))

    # === Enhanced Evaluation ===
    print("\nClassification Report:")
    print(classification_report(test_labels, ensemble_preds, target_names=CLASSES, digits=4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(test_labels, ensemble_preds))

    plot_confusion_matrix(test_labels, ensemble_preds, CLASSES)

    # === Model Comparison ===
    create_model_comparison(trained_models, training_histories, test_images, test_labels)


    print("\nPipeline completed successfully!")
    print(f"Results saved to: {GRAPHS_DIR}")

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel('ERROR')
    np.random.seed(42)
    tf.random.set_seed(42)
   
    main()