import os
import cv2
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from ultralytics import YOLO

# Load the YOLOv8 model
detector_model = YOLO('yolo_detector.pt')
recognition_model = YOLO('yolo_classifier.pt')

# Path to the folder containing user-specific subfolders
dataset_folder = '../../data/datasets/EarVN1/Images'

# Get a list of subfolders (each corresponding to a user)
user_subfolders = [user for user in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, user))]

# Lists to store true labels and predicted probabilities
true_labels = []
predicted_probs = []

# Lists to store labels and features for train and validation sets
train_labels = []
train_features = []
val_labels = []
val_features = []

# Loop through each user's subfolder
for user_subfolder in user_subfolders:
    # Path to the subfolder corresponding to the user
    user_folder_path = os.path.join(dataset_folder, user_subfolder)

    # Get a list of image files in the user's subfolder
    image_files = [f for f in os.listdir(user_folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

    # Split the images into train and validation sets
    train_images, val_images = train_test_split(image_files, test_size=0.2, random_state=42)

    # Loop through each image in the train set
    for image_file in train_images:
        image_path = os.path.join(user_folder_path, image_file)
        frame = cv2.imread(image_path)

        # Run YOLOv8 inference on the frame for recognition
        recognition_result = recognition_model(frame)

        # Check if 'probs' attribute exists in recognition_result[0]
        if hasattr(recognition_result[0], 'probs') and recognition_result[0].probs is not None:
            # Convert 'probs' to a NumPy array and get the predicted probability for the positive class
            probs_tensor = recognition_result[0].probs.numpy()
            predicted_prob = probs_tensor[:, 1] if probs_tensor.shape[1] > 1 else probs_tensor[:, 0]

            # Add ground truth label (user's name) and predicted probability to the lists
            true_labels.append(user_subfolder)
            predicted_probs.append(np.max(predicted_prob))  # Using the maximum predicted probability

            # Add label and features to the train set
            train_labels.append(user_subfolder)
            train_features.append(frame)

    # Loop through each image in the validation set
    for image_file in val_images:
        image_path = os.path.join(user_folder_path, image_file)
        frame = cv2.imread(image_path)

        # Add label and features to the validation set
        val_labels.append(user_subfolder)
        val_features.append(frame)

# Convert the lists to NumPy arrays
true_labels = np.array(true_labels)
predicted_probs = np.array(predicted_probs)

train_labels = np.array(train_labels)
train_features = np.array(train_features)

val_labels = np.array(val_labels)
val_features = np.array(val_features)

# Perform one-hot encoding on the true labels
y_true = label_binarize(true_labels, classes=user_subfolders)

# Perform one-hot encoding on the validation set labels
y_val = label_binarize(val_labels, classes=user_subfolders)

# Compute ROC curve and ROC area for each class
fpr, tpr, _ = roc_curve(y_true.ravel(), predicted_probs)
roc_auc = auc(fpr, tpr)

# Compute ROC curve for the validation set
y_score_val = []
for val_feature in val_features:
    recognition_result = recognition_model(val_feature)
    probs_tensor = recognition_result[0].probs.numpy()
    predicted_prob = probs_tensor[:, 1] if probs_tensor.shape[1] > 1 else probs_tensor[:, 0]
    y_score_val.append(np.max(predicted_prob))

fpr_val, tpr_val, _ = roc_curve(y_val.ravel(), y_score_val)
roc_auc_val = auc(fpr_val, tpr_val)

# Plot the ROC curves
plt.figure(figsize=(12, 8))
plt.plot(fpr, tpr, label=f'Training Set ROC curve (AUC = {roc_auc:.2f})')
plt.plot(fpr_val, tpr_val, label=f'Validation Set ROC curve (AUC = {roc_auc_val:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label="Random")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for User Recognition')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
