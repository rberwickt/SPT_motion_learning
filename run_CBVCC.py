import os, json
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers  # Layers
from tensorflow.keras.optimizers import Adam # Optimizer 
from tensorflow.keras.callbacks import ModelCheckpoint
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import csv

# might be needed since i have a newer version
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
#---------------------------------------------------------------------------

def get_actual_predicted_labels(dataset, model, save_errors=False, path_output=None):
    """
      Create a list of actual ground truth values and the predictions from the model.

      Args:
        dataset: An iterable data structure, such as a TensorFlow Dataset, with features and labels.

      Return:
        Ground truth and predicted values for a particular dataset.
    """
    actual = [labels for _, labels in dataset.unbatch()]
    videos = [vid for vid, _ in dataset.unbatch()]
    actual = tf.argmax(actual, axis=1)
    res = model.predict(dataset)

    actual = tf.stack(actual, axis=0)
    proba = tf.concat(res, axis=0)
    predicted = tf.argmax(proba, axis=1)
    
    if save_errors==True and path_output!=None:
        os.makedirs(path_output, exist_ok=True)
        for i, win in enumerate(videos):
            if actual[i]!=predicted[i]:
                outfile = path_output + "C{}_P{}_{}.avi".format(actual[i], predicted[i], i)
                out = cv2.VideoWriter(outfile, cv2.VideoWriter_fourcc(*'MJPG'), 1, (win.shape[1], win.shape[2]), True)
                for k in range(0,win.shape[0]):
                    #plt.imshow(win[k,...])
                    #plt.show()
                    data = np.array(win[k,...]*255., dtype=np.uint8)[..., [2, 1, 0]]
                    out.write(data)
                out.release()

    return actual, predicted, proba

#---------------------------------------------------------------------------

def format_frames(frame, output_size):
    """
    Pad and resize an image from a video.

    Args:
      frame: Image that needs to resized and padded. 
      output_size: Pixel size of the output frame image.

    Return:
      Formatted frame with padding of specified output size.
    """
    frame = tf.image.convert_image_dtype(frame, tf.float32) # convert dtype and normalize [0,1)
    frame = tf.image.resize_with_pad(frame, *output_size)
    return frame

def frames_from_video_file(video_path, n_frames, output_size = (224,224), frame_step = 15):
    """
    Creates frames from each video file present for each category.

    Args:
      video_path: File path to the video.
      n_frames: Number of frames to be created per video file.
      output_size: Pixel size of the output frame image.

    Return:
      An NumPy array of frames in the shape of (n_frames, height, width, channels).
    """
    #leggo il video
    video_reader = cv2.VideoCapture(video_path)
    video = []
    k = 0
    while True:
        ret, frame = video_reader.read()
        if frame is None: 
            if k!=20:
                print('NOO', k)
            break
        else: 
            frame = format_frames(frame, output_size)
            video.append(frame) #RGB
            k+=1
    result = np.array(video)[..., [2, 1, 0]]
    return result
    
#---------------------------------------------------------------------------

def main():
    
    tf.keras.backend.clear_session()
    DATA_PATH = args.testing_path
    WEIGHTS_DIR = args.weights_path
    
    WINDOW_SIZE = (20,50,50,3) 
    
    model = tf.keras.models.load_model(WEIGHTS_DIR+os.path.sep+'classifier.hdf5')
    with open(os.path.join(WEIGHTS_DIR, 'subm_phase2.csv'), mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for fn in glob(os.path.join(DATA_PATH, '*/*.avi')):
            base = os.path.basename(fn)
            video = frames_from_video_file(fn, WINDOW_SIZE[0], output_size=WINDOW_SIZE[1:3], frame_step=1)
            video_array = np.expand_dims(video, axis=0)
            prob = model.predict(video_array, verbose=0)

            print(base, prob[0][1]) 
            writer.writerow([base, prob[0][1]])   
   
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Evaluate CVBCC")
    
    parser.add_argument('--testing_path', required=True, help='Path to the testing video (.avi file)')
    parser.add_argument('--weights_path', required=True, help='Directory to save the model weights')

    args = parser.parse_args()
    main(args)
    
