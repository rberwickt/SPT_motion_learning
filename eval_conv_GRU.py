import cv2, os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers 
from tensorflow.keras.applications.densenet import DenseNet121
from utils.tif_dataset import create_tf_dataset, create_label_map
from utils.models import conv_GRU
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def main():
    DATA_DIR = "motion_dataset_2000_30f"  
    BATCH_SIZE = 3  
    INPUT_SHAPE = (30, 256, 256, 1)  # leave last value (channel) as 1 always
    NUM_CLASSES = 4
    CLASS_LABELS = ["ND", "DM", "AD", "CD"]
    MODEL_PATH = "best_motion_GRU_model.keras"

    print("GPUs Detected:", tf.config.list_physical_devices('GPU'), flush=True)

    print(f"Loading saved model from {MODEL_PATH}...", flush=True)
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file '{MODEL_PATH}' not found!")
        return
    model = tf.keras.models.load_model(MODEL_PATH)
    label_map = create_label_map(DATA_DIR, CLASS_LABELS)
    filenames = list(label_map.keys())
    labels = list(label_map.values())

    train_files, temp_files, train_lbls, temp_lbls = train_test_split(
        filenames, labels, test_size=0.30, stratify=labels, random_state=42
    )

    val_files, test_files, val_lbls, test_lbls = train_test_split(
        temp_files, temp_lbls, test_size=0.666, stratify=temp_lbls, random_state=42
    )

    
    val_map = dict(zip(val_files, val_lbls))
    test_map = dict(zip(test_files, test_lbls))

    
    val_dataset = create_tf_dataset(DATA_DIR, val_map, NUM_CLASSES, batch_size=BATCH_SIZE, target_frames=30, crop_size=(256,256), shuffle=False)
    test_dataset = create_tf_dataset(DATA_DIR, test_map, NUM_CLASSES, batch_size=BATCH_SIZE, target_frames=30, crop_size=(256,256), shuffle=False)
    

   
    print("\n--- Generating Confusion Matrices ---")
    
    # Extract true labels directly from the dataset
    print("Extracting true labels...")
    y_val_true = []
    for _, labels in val_dataset:
        lbl_np = labels.numpy()
        if len(lbl_np.shape) > 1:
            y_val_true.extend(np.argmax(lbl_np, axis=-1))
        else:
            y_val_true.extend(lbl_np)
    y_val_true = np.array(y_val_true)

    # Predict
    print("Predicting validation set...")
    val_preds = model.predict(val_dataset) 
    y_val_pred = np.argmax(val_preds, axis=-1)

    # same for test
    print("Extracting true labels...")
    y_test_true = []
    for _, labels in test_dataset:
        lbl_np = labels.numpy()
        if len(lbl_np.shape) > 1:
            y_test_true.extend(np.argmax(lbl_np, axis=-1))
        else:
            y_test_true.extend(lbl_np)
    y_test_true = np.array(y_test_true)

    print("Predicting test set...")
    test_preds = model.predict(test_dataset)
    y_test_pred = np.argmax(test_preds, axis=-1)
    cm_val = confusion_matrix(y_val_true, y_val_pred, labels=list(range(NUM_CLASSES)))
    cm_test = confusion_matrix(y_test_true, y_test_pred, labels=list(range(NUM_CLASSES)))

    # Plotting 
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    ConfusionMatrixDisplay(confusion_matrix=cm_val, display_labels=CLASS_LABELS).plot(ax=ax[0], cmap='Blues', values_format='d')
    ax[0].set_title('Validation Confusion Matrix')

    ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=CLASS_LABELS).plot(ax=ax[1], cmap='Greens', values_format='d')
    ax[1].set_title('Test Confusion Matrix')

    plt.tight_layout()
    output_filename = 'motion_GRU_confusion_matrices.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close() # Instantly cleans up memory allocation

if __name__ == "__main__":
    main()