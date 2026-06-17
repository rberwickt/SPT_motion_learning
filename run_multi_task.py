import os
import tifffile as tiff
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from utils.models_multi_task import multi_conv_3D

# AI assisted code for help with the dataset as this was before the utils file was made, model didn't work that well (the multi-task)
print("GPUs Detected:", tf.config.list_physical_devices('GPU'), flush=True)

def load_tiff_3d(file_path, target_shape=(14, 256, 256)):
    if isinstance(file_path, bytes):
        file_path = file_path.decode('utf-8')

    try:
        img_stack = tiff.imread(file_path)
    except Exception as e:
        return np.zeros(target_shape + (1,), dtype=np.float32)
        
    if img_stack.ndim == 2:
        img_stack = np.expand_dims(img_stack, axis=0)
    
    max_val = np.max(img_stack)
    min_val = np.min(img_stack)
    if max_val - min_val > 0:
        img_stack = (img_stack - min_val) / (max_val - min_val)
    else:
        img_stack = np.zeros_like(img_stack, dtype=np.float32)

    resized_frames = []
    for frame in img_stack:
        resized_frame = cv2.resize(frame, (target_shape[2], target_shape[1]))
        resized_frames.append(resized_frame)
        
    target_depth = target_shape[0]
    if len(resized_frames) >= target_depth:
        resized_frames = resized_frames[:target_depth]
    else:
        padding = [np.zeros_like(resized_frames[0]) for _ in range(target_depth - len(resized_frames))]
        resized_frames.extend(padding)
        
    data_3d = np.stack(resized_frames, axis=0)
    data_3d = np.expand_dims(data_3d, axis=-1)
    return data_3d.astype('float32')

class TIFFDataGenerator:
    def __init__(self, file_paths, y1, y2, target_shape):
        self.file_paths = file_paths
        self.y1 = y1
        self.y2 = y2
        self.target_shape = target_shape

    def __call__(self):
        for path, l1, l2 in zip(self.file_paths, self.y1, self.y2):
            X = load_tiff_3d(path, self.target_shape)
            yield X, {"task1_output": l1, "task2_output": l2}

def main():
    DATA_DIR = "motion_dataset_2000_30f"  
    BATCH_SIZE = 2  
    TARGET_SHAPE = (30, 256, 256) 
    
    file_paths = []
    raw_labels_t1 = []
    raw_labels_t2 = []
    
    print(f"Parsing dataset directory: '{DATA_DIR}'...", flush=True)
    if not os.path.exists(DATA_DIR):
        print(f"ERROR: The directory '{DATA_DIR}' does not exist relative to execution path!", flush=True)
        return

    all_files = os.listdir(DATA_DIR)
    print(f"Found {len(all_files)} total raw items in folder. Starting label matching...", flush=True)

    for filename in all_files:
        if filename.endswith(".tiff") or filename.endswith(".tif"):
            name_without_ext = os.path.splitext(filename)[0]
            parts = name_without_ext.split('_')
            if len(parts) >= 3:
                file_paths.append(os.path.join(DATA_DIR, filename))
                raw_labels_t1.append(parts[1])
                raw_labels_t2.append(parts[2])
                
    print(f"Parsing complete. Successfully processed {len(file_paths)} files", flush=True)
                
    if not file_paths:
        print("No valid matching dataset files found! Check filename structures.", flush=True)
        return

    # Create combined labels for 16-way multi-task stratification
    combined_labels = [f"{t1}_{t2}" for t1, t2 in zip(raw_labels_t1, raw_labels_t2)]

    le1, le2 = LabelEncoder(), LabelEncoder()
    encoded_t1 = le1.fit_transform(raw_labels_t1)
    encoded_t2 = le2.fit_transform(raw_labels_t2)
    
    y1_onehot = tf.keras.utils.to_categorical(encoded_t1, len(le1.classes_))
    y2_onehot = tf.keras.utils.to_categorical(encoded_t2, len(le2.classes_))

    print("\n--- CLASS INDEX MAPPING ---")
    num_classes1 = len(le1.classes_)
    num_classes2 = len(le2.classes_)
    for i in range(max(num_classes1,num_classes2)):
        if i < num_classes1 and i < num_classes2:
            print(f" {i} ===> Task 1: '{le1.classes_[i]}'\t|\tTask 2: '{le2.classes_[i]}'")
        elif i < num_classes1:
            print(f" {i} ===> Task 1: '{le1.classes_[i]}'")
        elif i < num_classes2:
            print(f" {i} ===> Task 2: '{le2.classes_[i]}'")

    idx = np.arange(len(file_paths))
    # Stratify using the combined labels to balance all 16 classes across splits
    train_idx, val_idx = train_test_split(
        idx, 
        test_size=0.2, 
        random_state=42, 
        stratify=combined_labels
    )
    
    # Calculate exact steps per epoch
    steps_per_epoch = int(np.ceil(len(train_idx) / BATCH_SIZE))
    validation_steps = int(np.ceil(len(val_idx) / BATCH_SIZE))

    train_gen = TIFFDataGenerator([file_paths[i] for i in train_idx], y1_onehot[train_idx], y2_onehot[train_idx], TARGET_SHAPE)
    val_gen = TIFFDataGenerator([file_paths[i] for i in val_idx], y1_onehot[val_idx], y2_onehot[val_idx], TARGET_SHAPE)

    train_gen = TIFFDataGenerator([file_paths[i] for i in train_idx], y1_onehot[train_idx], y2_onehot[train_idx], TARGET_SHAPE)
    val_gen = TIFFDataGenerator([file_paths[i] for i in val_idx], y1_onehot[val_idx], y2_onehot[val_idx], TARGET_SHAPE)

    output_signature = (
        tf.TensorSpec(shape=TARGET_SHAPE + (1,), dtype=tf.float32),
        {
            "task1_output": tf.TensorSpec(shape=(len(le1.classes_),), dtype=tf.float32),
            "task2_output": tf.TensorSpec(shape=(len(le2.classes_),), dtype=tf.float32)
        }
    )

    train_dataset = tf.data.Dataset.from_generator(train_gen, output_signature=output_signature)
    train_dataset = train_dataset.repeat().shuffle(buffer_size=32).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_generator(val_gen, output_signature=output_signature)
    val_dataset = val_dataset.repeat().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    input_shape = TARGET_SHAPE + (1,)
    print("Building model graph...", flush=True)
    
    model = multi_conv_3D(input_shape, len(le1.classes_), len(le2.classes_))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4, gradient_accumulation_steps=8),
        loss={'task1_output': 'categorical_crossentropy', 'task2_output': 'categorical_crossentropy'},
        loss_weights={'task1_output': 0.6, 'task2_output': 1.0},
        metrics={'task1_output': 'accuracy', 'task2_output': 'accuracy'}
    )

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='best_multi_task_model.keras', # Name of the file to save to
        monitor='val_loss',                     # Track validation loss
        mode='min',                             # We want the lowest loss possible
        save_best_only=True,                    # Only overwrite if performance improves
        verbose=1                               # Prints a message when a new best is found
    )

    model.fit(train_dataset, validation_data=val_dataset, epochs=10, 
    steps_per_epoch = int(np.ceil(len(train_idx) / BATCH_SIZE)), 
    validation_steps = int(np.ceil(len(val_idx) / BATCH_SIZE)),
    callbacks=[checkpoint_callback])

if __name__ == "__main__":
    main()