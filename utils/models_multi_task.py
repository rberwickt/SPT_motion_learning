import tensorflow as tf
from tensorflow.keras import layers 
from tensorflow.keras.models import Model 
from tensorflow.keras.regularizers import l2

def multi_conv_3D(input_shape, num_classes_task1, num_classes_task2):
    inputs = layers.Input(shape=input_shape)
    
    # 1. LOOSEN REGULARIZATION: Relax weight decay so weights can change freely
    weight_decay = 1e-5  
    
    # =========================================================================
    # CBVCC_CNN STYLE SHARED SPATIOTEMPORAL BACKBONE
    # =========================================================================
    # Block 1
    x = layers.Conv3D(32, (3, 3, 3), padding='same', kernel_regularizer=l2(weight_decay))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPool3D((1, 2, 2), strides=(1, 2, 2))(x) 
    
    # Block 2
    x = layers.Conv3D(64, (3, 3, 3), padding='same', kernel_regularizer=l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPool3D((2, 2, 2), strides=(2, 2, 2))(x)

    # Block 3
    x = layers.Conv3D(128, (3, 3, 3), padding='same', kernel_regularizer=l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPool3D((2, 2, 2), strides=(2, 2, 2))(x)

    # =========================================================================
    # BRANCH 1: AUGMENTATION HEAD (Motion Blur, Shot Noise, Gradients)
    # =========================================================================
    s_pooled = layers.GlobalAveragePooling3D()(x)
    
    # Increased representation capacity to 128 and lowered dropout to 0.2
    s_dense = layers.Dense(128, kernel_regularizer=l2(weight_decay))(s_pooled)
    s_dense = layers.BatchNormalization()(s_dense)
    s_dense = layers.Activation('relu')(s_dense)
    s_dropout = layers.Dropout(0.2)(s_dense) 
    out_task1 = layers.Dense(num_classes_task1, activation='softmax', name='task1_output')(s_dropout)

    # =========================================================================
    # BRANCH 2: RECURRENT MOTION TRACKING HEAD (Type of Motion)
    # =========================================================================
    # 1. Keep the 4D spatial-temporal tensor structure intact.
    # We do NOT collapse depth into channels here. We treat depth as time steps.
    
    # ConvLSTM2D reads the video frames sequentially, tracking vectors over time
    t_lstm = layers.ConvLSTM2D(
        filters=32, 
        kernel_size=(3, 3), 
        padding='same', 
        return_sequences=False, # Consolidates the timeline into a final motion summary map
        kernel_regularizer=l2(weight_decay)
    )(x) # Directly processes the 3D block output
    t_lstm = layers.BatchNormalization()(t_lstm)
    t_lstm = layers.Activation('relu')(t_lstm)
    t_lstm = layers.SpatialDropout2D(0.2)(t_lstm)
    
    # 2. Safely flatten the final recurrent motion map 
    t_flattened = layers.Flatten()(t_lstm)
    
    t_dense = layers.Dense(128, kernel_regularizer=l2(weight_decay))(t_flattened)
    t_dense = layers.BatchNormalization()(t_dense)
    t_dense = layers.Activation('relu')(t_dense)
    t_dropout = layers.Dropout(0.4)(t_dense)  
    out_task2 = layers.Dense(num_classes_task2, activation='softmax', name='task2_output')(t_dropout)
    # =========================================================================
    # MODEL ASSEMBLY
    # =========================================================================
    model = Model(inputs=inputs, outputs=[out_task1, out_task2], name="cbvcc_multitask_tracking_network")
    return model