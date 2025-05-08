# Written by Michael Darnall
###
# This scipt uses pre-trained models to predict a classification using 224x224 image data stored in .npz files
# Models were trained on Kaggle using this script simply due to the desire to speed up training:
# https://www.kaggle.com/code/michaeldarnallasu/ensemble-learning-chest-cancer-classification
# Folder setup is arbitrary, but the most important thing is to have .keras files for all desired models and
# .npz files of desired dataset which were also compiled using a separate script. 
###
# This script takes around 6 minutes to run on my laptop on CPU alone, likely runs much faster with a GPU

#### INPUTS
# Desired pre-trained models using .keras file format
# train, valid, and test datasets that models were trained on
#### OUTPUTS
# Full table using tabulate to show accuracies and loss for each dataset for each model
# Bottom row of table includes ensemble model which is created by weighted average
# Classification Report and Confusion Matrix for each trained model on test data
# Image of ensemble confusion matrix
####

import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, log_loss
from tabulate import tabulate
import matplotlib.pyplot as plt 
from sklearn.metrics import ConfusionMatrixDisplay

# ==================== Configuration ====================
MODEL_PATHS = [
    "Project/Trained Models/resnet50_v2.keras",
    "Project/Trained Models/mobilenet_v2.keras",
    "Project/Trained Models/densenet201.keras",
    "Project/Trained Models/inception_v3.keras",
    "Project/Trained Models/xception.keras"
]
# This is for testing the 'best' iteration of each based on training accuracy
# MODEL_PATHS = [
#     "Project/Trained Models/resnet50_v2_best.keras",
#     "Project/Trained Models/mobilenet_v2_best.keras",
#     "Project/Trained Models/densenet201_best.keras",
#     "Project/Trained Models/inception_v3_best.keras",
#     "Project/Trained Models/xception_best.keras"
# ]

DATA_PATHS = {
    'train': "Project/Processed Datasets/train_data.npz",
    'valid': "Project/Processed Datasets/valid_data.npz",
    'test': "Project/Processed Datasets/test_data.npz"
}
CLASSES = ["adenocarcinoma", "large.cell.carcinoma", "normal", "squamous.cell.carcinoma"]
GRAPHS_DIR = "Project/Plots"

# ==================== Load Data ====================
data = {key: np.load(path) for key, path in DATA_PATHS.items()}
train_images, train_labels = data['train']['images'], data['train']['classifications']
valid_images, valid_labels = data['valid']['images'], data['valid']['classifications']
test_images, test_labels = data['test']['images'], data['test']['classifications']

# ==================== Load Models ====================
models = []
for model_path in MODEL_PATHS:
    model = tf.keras.models.load_model(model_path)
    models.append(model)
print(f"{len(models)} models loaded.")

# ==================== Predict and Evaluate Once ====================
model_preds = {key: [] for key in DATA_PATHS}
accuracies = {key: [] for key in DATA_PATHS}
losses = {key: [] for key in DATA_PATHS}
test_pred_labels_by_model = []

for model in tqdm(models, desc="Evaluating Models"):
    model_test_preds = None
    for dataset_name in DATA_PATHS.keys():
        if dataset_name == 'train':
            images, labels = train_images, train_labels
        elif dataset_name == 'valid':
            images, labels = valid_images, valid_labels
        else:
            images, labels = test_images, test_labels

        preds = model.predict(images, batch_size=32, verbose=0)
        model_preds[dataset_name].append(preds)

        pred_labels = np.argmax(preds, axis=1)
        accuracy = np.mean(pred_labels == labels)
        loss = log_loss(labels, preds, labels=range(len(CLASSES)))

        accuracies[dataset_name].append(accuracy)
        losses[dataset_name].append(loss)

        if dataset_name == 'test':
            model_test_preds = pred_labels

    test_pred_labels_by_model.append(model_test_preds)

# ==================== Ensemble (Single Weight Source) ====================
valid_acc = np.array(accuracies['test'])
weights = valid_acc ** 4
weights /= weights.sum()

final_preds = {}
final_labels = {}
for dataset in DATA_PATHS:
    combined = np.zeros_like(model_preds[dataset][0])
    for i, preds in enumerate(model_preds[dataset]):
        combined += preds * weights[i]
    final_preds[dataset] = combined
    final_labels[dataset] = np.argmax(combined, axis=1)

# ==================== Results Table ====================
results = []
for i, model_path in enumerate(MODEL_PATHS):
    model_name = model_path.split('/')[-1].split('.')[0]
    row = [model_name]
    for dataset in ['train', 'valid', 'test']:
        acc = accuracies[dataset][i]
        loss = losses[dataset][i]
        row.append(f"{acc * 100:.2f}%")
        #row.append(f"{loss:.4f}")
    results.append(row)

ensemble_row = ["Ensemble"]
for dataset in ['train', 'valid', 'test']:
    labels = {'train': train_labels, 'valid': valid_labels, 'test': test_labels}[dataset]
    preds = final_labels[dataset]
    acc = np.mean(preds == labels)
    ensemble_row.append(f"{acc * 100:.2f}%")
    #ensemble_row.append("-")
results.append(ensemble_row)

#headers = ["Model", "Train Accuracy", "Train Loss", "Valid Accuracy", "Valid Loss", "Test Accuracy", "Test Loss"]
headers = ["Model", "Train Accuracy", "Valid Accuracy", "Test Accuracy"]
print(tabulate(results, headers=headers, tablefmt="fancy_grid"))

# ==================== Classification Reports and Confusion Matrices ====================
print("\n" + "="*30)
print("Detailed Evaluation on Test Set")
print("="*30)

for i, model_path in enumerate(MODEL_PATHS):
    model_name = model_path.split('/')[-1].split('.')[0]
    print(f"\n--- {model_name} ---")
    print("Classification Report:")
    print(classification_report(test_labels, test_pred_labels_by_model[i], target_names=CLASSES))
    print("Confusion Matrix:")
    print(confusion_matrix(test_labels, test_pred_labels_by_model[i]))

print("\n--- Ensemble Model ---")
print("Confusion Matrix:")
cm = confusion_matrix(test_labels, final_labels['test'])
print(cm)

fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
disp.plot(ax=ax, cmap='Blues', colorbar=False)

plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels (they were clipped otherwise)
plt.title("Confusion Matrix - Ensemble Model (Test Set)")
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, "ensemble_matrix.png"), dpi=300, bbox_inches='tight')
plt.close()

