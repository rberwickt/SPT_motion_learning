import cv2, os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers 
from tensorflow.keras.applications.densenet import DenseNet121
from utils.tif_dataset import create_tf_dataset, create_label_map
from utils.models import convLSTM_model
from sklearn.model_selection import train_test_split

def main(): # this is the convLSTM file, uses the tif_dataset method (messy confusion matrices code)
    DATA_DIR = "motion_dataset_2000_30f"  
    BATCH_SIZE = 2  
    INPUT_SHAPE = (30, 256, 256) 
    NUM_CLASSES = 4
    CLASS_LABELS = ["ND", "DM", "AD", "CD"]
    EPOCHS = 10

    print("GPUs Detected:", tf.config.list_physical_devices('GPU'), flush=True)
    # create dataset
    label_map = create_label_map(DATA_DIR, CLASS_LABELS)
    filenames = list(label_map.keys())
    labels = list(label_map.values())

    train_files, temp_files, train_lbls, temp_lbls = train_test_split(
        filenames, labels, test_size=0.30, stratify=labels, random_state=42
    )

    # Split the remaining 30% into Val (10% total) and Test (20% total)
    # 0.333 of 30% is roughly 10%
    val_files, test_files, val_lbls, test_lbls = train_test_split(
        temp_files, temp_lbls, test_size=0.666, stratify=temp_lbls, random_state=42
    )

    train_map = dict(zip(train_files, train_lbls))
    val_map = dict(zip(val_files, val_lbls))
    test_map = dict(zip(test_files, test_lbls))

    train_dataset = create_tf_dataset(DATA_DIR, train_map, batch_size=BATCH_SIZE, target_frames=30, crop_size=(256,256))
    val_dataset = create_tf_dataset(DATA_DIR, val_map, batch_size=BATCH_SIZE, target_frames=30, crop_size=(256,256), shuffle=False)
    test_dataset = create_tf_dataset(DATA_DIR, test_map, batch_size=BATCH_SIZE, target_frames=30, crop_size=(256,256), shuffle=False)
    
    print(f"Clips partitioned -> Train: {len(train_map)}, Val: {len(val_map)}, Test: {len(test_map)}")

    # Run model
    model = convLSTM_model(INPUT_SHAPE, NUM_CLASSES)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4 ,gradient_accumulation_steps=2),
        loss=tf.keras.losses.CategoricalFocalCrossentropy(gamma=2.0),
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='best_motion_model.keras', # Name of the file to save to
        monitor='val_loss',                     # Track validation loss
        mode='min',                             # We want the lowest loss possible
        save_best_only=True,                    # Only overwrite if performance improves
        verbose=0                               # Prints a message when a new best is found
    )

    print("\n--- Starting Training ---\n")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=[checkpoint_callback]
    )

    # Generate Confusion Matrices
    print("\n--- Generating Confusion Matrices ---")
    
    print("Predicting validation set...")
    val_preds = model.predict(val_dataset)
    y_val_pred = np.argmax(val_preds, axis=1)
    
    print("Predicting test set...")
    test_preds = model.predict(test_dataset)
    y_test_pred = np.argmax(test_preds, axis=1)

    # Because shuffle=False, the generator iterates through val_map.keys() in exact order
    y_val_true = np.array(list(val_map.values()))
    y_test_true = np.array(list(test_map.values()))

    cm_val = confusion_matrix(y_val_true, y_val_pred, labels=list(range(NUM_CLASSES)))
    cm_test = confusion_matrix(y_test_true, y_test_pred, labels=list(range(NUM_CLASSES)))


    # Plotting 
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    ConfusionMatrixDisplay(confusion_matrix=cm_val, display_labels=CLASS_LABELS).plot(ax=ax[0], cmap='Blues', values_format='d')
    ax[0].set_title('Validation Confusion Matrix')

    ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=CLASS_LABELS).plot(ax=ax[1], cmap='Greens', values_format='d')
    ax[1].set_title('Test Confusion Matrix')

    plt.tight_layout()
    
    output_filename = 'motion_confusion_matrices.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close() # Instantly cleans up memory allocation


if __name__ == "__main__":
    main()