# Import dependencies
import tensorflow as tf
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD


def load_model(num_outputs, inp_shape=(224, 224, 3)):
    # Load DenseNet201
    base_model = DenseNet201(input_shape=inp_shape, include_top=False)

    # Freeze base model, so the underlying pre-trained patterns
    # aren't updated during training
    base_model.trainable = False 

    # Add fine-tuning layers
    new_model = tf.keras.Sequential()
    new_model.add(base_model)
    new_model.add(GlobalAveragePooling2D())
    new_model.add(BatchNormalization(momentum=0.90))
    new_model.add(Dropout(rate=0.5))
    new_model.add(Dense(units=4096, activation='relu'))
    new_model.add(Dropout(rate=0.6))
    new_model.add(BatchNormalization(momentum=0.9))
    new_model.add(Dense(units=num_outputs, activation='softmax'))
    
    # Compile the model
    new_model.compile(loss="categorical_crossentropy",
                      optimizer=SGD(learning_rate=0.001, momentum=0.9, nesterov=True),
                      metrics=["accuracy"])

    return new_model


m = load_model(num_outputs=170, inp_shape=(224, 224, 3))
m.summary()
