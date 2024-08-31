import os
import warnings
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image as image_utils
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Định nghĩa các biến
gestures = {'L_': 'L',
           'fi': 'E',
           'ok': 'F',
           'pe': 'V',
           'pa': 'B'
            }

gestures_map = {'E': 0,
                'L': 1,
                'F': 2,
                'V': 3,
                'B': 4
                }

gesture_names = {0: 'E',
                 1: 'L',
                 2: 'F',
                 3: 'V',
                 4: 'B'}

image_path = 'data'
models_path = 'models/saved_model.keras'  # Đổi định dạng thành .keras
rgb = False
imageSize = 224

# Hàm xử lý ảnh resize về 224x224 và chuyển về numpy array
def process_image(path):
    img = Image.open(path)
    img = img.resize((imageSize, imageSize))
    img = np.array(img)
    return img

# Xử lý dữ liệu đầu vào
def process_data(X_data, y_data):
    X_data = np.array(X_data, dtype='float32')
    if rgb:
        pass
    else:
        X_data = np.stack((X_data,) * 3, axis=-1)
    X_data /= 255
    y_data = np.array(y_data)
    y_data = to_categorical(y_data)
    return X_data, y_data

# Hàm duyệt thư mục ảnh dùng để train
def walk_file_tree(image_path):
    X_data = []
    y_data = []
    for directory, subdirectories, files in os.walk(image_path):
        for file in files:
            if not file.startswith('.'):
                path = os.path.join(directory, file)
                gesture_key = file[0:2]
                
                if gesture_key in gestures:
                    gesture_name = gestures[gesture_key]
                    print(f"Processing gesture: {gesture_name}")
                    y_data.append(gestures_map[gesture_name])
                    X_data.append(process_image(path))
                else:
                    print(f"Warning: '{gesture_key}' not found in gestures dictionary.")
            else:
                continue

    X_data, y_data = process_data(X_data, y_data)
    return X_data, y_data

# Load dữ liệu vào X và Y
X_data, y_data = walk_file_tree(image_path)

# Phân chia dữ liệu train và test theo tỷ lệ 80/20
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=12, stratify=y_data)

# Đặt các checkpoint để lưu lại model tốt nhất
model_checkpoint = ModelCheckpoint(filepath=models_path, save_best_only=True, save_weights_only=False, mode='auto', verbose=1)
early_stopping = EarlyStopping(monitor='val_accuracy',  # Đã sửa thành 'val_accuracy'
                               min_delta=0,
                               patience=10,
                               verbose=1,
                               mode='auto',
                               restore_best_weights=True)

# Khởi tạo model
model1 = VGG16(weights='imagenet', include_top=False, input_shape=(imageSize, imageSize, 3))
optimizer1 = optimizers.Adam()
base_model = model1

# Thêm các lớp bên trên
x = base_model.output
x = Flatten()(x)
x = Dense(128, activation='relu', name='fc1')(x)
x = Dense(128, activation='relu', name='fc2')(x)
x = Dense(128, activation='relu', name='fc2a')(x)
x = Dense(128, activation='relu', name='fc3')(x)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu', name='fc4')(x)

predictions = Dense(5, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Đóng băng các lớp dưới, chỉ train lớp bên trên mình thêm vào
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=optimizer1, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test), verbose=1,
          callbacks=[early_stopping, model_checkpoint])

# Lưu model đã train ra file
model.save('models/mymodel.h5')
