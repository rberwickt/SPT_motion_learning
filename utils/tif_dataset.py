import os
import glob
import numpy as np
import tifffile as tiff
import tensorflow as tf

class TiffGenerator:
    def __init__(self, tiff_folder, label_map, num_classes, target_frames=32, crop_size=(64, 64), shuffle=True):
        """
        Args:
            tiff_folder: Path to directory containing the .tif videos
            label_map: Dict mapping filename to a class integer
            target_frames: Number of temporal frames to slice per video
            crop_size: Tuple of (Height, Width) to standardize input dimensions
        """
        all_paths = glob.glob(os.path.join(tiff_folder, "*.tif"))
        self.file_paths = [p for p in all_paths if os.path.basename(p) in label_map]
        self.label_map = label_map
        self.target_frames = target_frames
        self.crop_size = crop_size
        self.shuffle = shuffle
        self.num_classes = num_classes 

    def __call__(self):
        # Local copy to prevent shuffling original state variables across epochs
        paths_to_iterate = list(self.file_paths)
        if self.shuffle:
            np.random.shuffle(paths_to_iterate)
        
        for file_path in paths_to_iterate:
            filename = os.path.basename(file_path)
            label = self.label_map[filename]
            
            try:
                with tiff.TiffFile(file_path) as tif:
                    video_stack = tif.asarray()
                
                if len(video_stack.shape) == 2:
                    video_stack = np.expand_dims(video_stack, axis=0)
                
                if video_stack.shape[0] < self.target_frames:
                    pad_width = self.target_frames - video_stack.shape[0]
                    video_stack = np.pad(video_stack, ((0, pad_width), (0, 0), (0, 0)), mode='edge')
                else:
                    video_stack = video_stack[:self.target_frames, :, :]
                
                f, h, w = video_stack.shape
                th, tw = self.crop_size
                if h >= th and w >= tw:
                    dy, dx = (h - th) // 2, (w - tw) // 2
                    video_stack = video_stack[:, dy:dy+th, dx:dx+tw]
                else:
                    raise ValueError(f"TIFF dimensions {h}x{w} smaller than target crop {th}x{tw}")
                
                video_stack = video_stack.astype(np.float32)
                mean = np.mean(video_stack)
                std = np.std(video_stack)
                video_stack = (video_stack - mean) / (std if std > 1e-6 else 1.0)
                
                video_stack = np.expand_dims(video_stack, axis=-1)
                
                yield video_stack, tf.one_hot(label, depth=self.num_classes)
                
            except Exception as e:
                print(f"Skipping corrupted file {filename}: {str(e)}")
                continue

def create_label_map(tiff_folder, class_labels):

    # Create a mapping from text to integer -> {'ND': 0, 'AD': 1, 'DM': 2, etc.}
    class_to_int = {name: idx for idx, name in enumerate(class_labels)}

    label_map = {}
    all_files = glob.glob(os.path.join(tiff_folder, "*.tif"))

    for file_path in all_files:
        filename = os.path.basename(file_path)
    
        # Check which class name is inside the filename
        for name in class_labels:
            if name in filename:
                label_map[filename] = class_to_int[name]
                break

    print("Total files mapped:", len(label_map))
    return label_map

def create_tf_dataset(tiff_folder, label_map, num_classes, batch_size=4, target_frames=30, crop_size=(256, 256), shuffle=True):
    # Use the generator to create the processed data
    tiff_gen = TiffGenerator(tiff_folder, label_map, num_classes, target_frames, crop_size, shuffle)
    
    # Wrap inside tf.data
    dataset = tf.data.Dataset.from_generator(
        tiff_gen,
        output_signature=(
            tf.TensorSpec(shape=(target_frames, crop_size[0], crop_size[1], 1), dtype=tf.float32),
            tf.TensorSpec(shape=(num_classes,), dtype=tf.float32)
        )
    )
    
    # don't need to call repeat() or shuffle() since the generator handles it
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset