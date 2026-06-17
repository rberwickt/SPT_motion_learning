import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# this was made pretty quickly with AI to just see if everything worked. eval_conv_GRU.py is more comprehensive and refined
# note that the better version uses the tif_dataset generator instead of a normal dataset
from run_multi_task import load_tiff_3d

def main():
    DATA_DIR = "motion_dataset_2000_30f"  
    TARGET_SHAPE = (30, 256, 256)
    MODEL_PATH = "best_multi_task_model.keras"
    
    file_paths, raw_labels_t1, raw_labels_t2 = [], [], []
    
    if not os.path.exists(DATA_DIR):
        print(f"ERROR: The directory '{DATA_DIR}' does not exist!", flush=True)
        return
        
    all_files = os.listdir(DATA_DIR)
    for filename in all_files:
        if filename.endswith(".tiff") or filename.endswith(".tif"):
            name_without_ext = os.path.splitext(filename)[0]
            parts = name_without_ext.split('_')
            if len(parts) >= 3:
                file_paths.append(os.path.join(DATA_DIR, filename))
                raw_labels_t1.append(parts[1])
                raw_labels_t2.append(parts[2])

    # 2. Encode Labels to ensure structural matching
    le1, le2 = LabelEncoder(), LabelEncoder()
    encoded_t1 = le1.fit_transform(raw_labels_t1)
    encoded_t2 = le2.fit_transform(raw_labels_t2)
    
    # 3. Re-split with the matching random_state to isolate validation data
    idx = np.arange(len(file_paths))
    train_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=42)
    
    # Evaluate on the validation split
    eval_indices = val_idx 
    print(f"Preparing evaluation matrix for {len(eval_indices)} validation samples...", flush=True)

    # 4. Load the Model
    print(f"Loading saved model from {MODEL_PATH}...", flush=True)
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file '{MODEL_PATH}' not found!")
        return
    model = tf.keras.models.load_model(MODEL_PATH)

    # 5. Generate Predictions
    y_true_t1, y_true_t2 = [], []
    y_pred_t1, y_pred_t2 = [], []

    for count, i in enumerate(eval_indices, 1):
        print(f"Processing sample {count}/{len(eval_indices)}...", end="\r", flush=True)
        
        X = load_tiff_3d(file_paths[i], TARGET_SHAPE)
        X_batch = np.expand_dims(X, axis=0) 
        
        y_true_t1.append(encoded_t1[i])
        y_true_t2.append(encoded_t2[i])
        
        predictions = model.predict(X_batch, verbose=0)
        
        # Handle dictionary vs list output formats smoothly
        pred_t1_idx = np.argmax(predictions[0] if isinstance(predictions, list) else predictions['task1_output'])
        pred_t2_idx = np.argmax(predictions[1] if isinstance(predictions, list) else predictions['task2_output'])
        
        y_pred_t1.append(pred_t1_idx)
        y_pred_t2.append(pred_t2_idx)

    print("\nInference complete! Plotting confusion matrices...")

    # 6. Plot Side-by-Side Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Task 1
    cm_t1 = confusion_matrix(y_true_t1, y_pred_t1)
    disp_t1 = ConfusionMatrixDisplay(confusion_matrix=cm_t1, display_labels=le1.classes_)
    disp_t1.plot(ax=axes[0], cmap='Blues', xticks_rotation=45)
    axes[0].set_title("Noise Confusion Matrix")

    # Task 2
    cm_t2 = confusion_matrix(y_true_t2, y_pred_t2)
    disp_t2 = ConfusionMatrixDisplay(confusion_matrix=cm_t2, display_labels=le2.classes_)
    disp_t2.plot(ax=axes[1], cmap='Oranges', xticks_rotation=45)
    axes[1].set_title("Motion Confusion Matrix")

    plt.tight_layout()
    plt.savefig("confusion_matrices.png", dpi=300)
    print("Matrices successfully saved to 'confusion_matrices.png'!")
    plt.show()

if __name__ == "__main__":
    main()