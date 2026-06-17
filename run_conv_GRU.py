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
    EPOCHS = 35

    print("GPUs Detected:", tf.config.list_physical_devices('GPU'), flush=True)
    
    label_map = create_label_map(DATA_DIR, CLASS_LABELS)
    filenames = list(label_map.keys())
    labels = list(label_map.values())

    train_files, temp_files, train_lbls, temp_lbls = train_test_split(
        filenames, labels, test_size=0.30, stratify=labels, random_state=42
    )

    val_files, test_files, val_lbls, test_lbls = train_test_split(
        temp_files, temp_lbls, test_size=0.666, stratify=temp_lbls, random_state=42
    )

    train_map = dict(zip(train_files, train_lbls))
    val_map = dict(zip(val_files, val_lbls))
    test_map = dict(zip(test_files, test_lbls))

    train_dataset = create_tf_dataset(DATA_DIR, train_map, NUM_CLASSES, batch_size=BATCH_SIZE, target_frames=30, crop_size=(256,256))
    val_dataset = create_tf_dataset(DATA_DIR, val_map, NUM_CLASSES, batch_size=BATCH_SIZE, target_frames=30, crop_size=(256,256), shuffle=False)
    test_dataset = create_tf_dataset(DATA_DIR, test_map, NUM_CLASSES, batch_size=BATCH_SIZE, target_frames=30, crop_size=(256,256), shuffle=False)
    
    print(f"Clips partitioned -> Train: {len(train_map)}, Val: {len(val_map)}, Test: {len(test_map)}")

    model = conv_GRU(INPUT_SHAPE, NUM_CLASSES)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='best_motion_GRU_model.keras',
        monitor='val_loss',                    
        mode='min',                            
        save_best_only=True,                    
        verbose=1 
    )


    steps_per_train_epoch = len(train_map) // BATCH_SIZE
    print("\n--- Starting Training ---\n")
    history = model.fit(
        train_dataset.repeat(),
        validation_data=val_dataset,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_train_epoch,
        callbacks=[checkpoint_callback]
    )

    print("\n--- Generating Confusion Matrices ---")
    
    # FIX 3: Safe Extraction of Ground Truth Labels directly from tf.data arrays
    print("Extracting true labels and predicting validation set...")
    y_val_true = []
    val_preds = []
    for images, labels in val_dataset:
        y_val_true.extend(np.argmax(labels.numpy(), axis=1)) # Assumes one-hot encoded output from dataset
        val_preds.extend(model.predict_on_batch(images))
    y_val_pred = np.argmax(np.array(val_preds), axis=1)
    y_val_true = np.array(y_val_true)
    
    print("Extracting true labels and predicting test set...")
    y_test_true = []
    test_preds = []
    for images, labels in test_dataset:
        y_test_true.extend(np.argmax(labels.numpy(), axis=1))
        test_preds.extend(model.predict_on_batch(images))
    y_test_pred = np.argmax(np.array(test_preds), axis=1)
    y_test_true = np.array(y_test_true)

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