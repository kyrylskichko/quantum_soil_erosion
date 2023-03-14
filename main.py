import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input


IMG_SIZE = (90, 90)
epochs = 100

features_data_path = f'data/train/features/'
labels_data_path = f'data/train/masks/'

batch_size = 32

train_features_datagen = ImageDataGenerator(rescale=1./255)
train_labels_datagen = ImageDataGenerator(rescale=1./255)


train_features_generator = train_features_datagen.flow_from_directory(
    features_data_path,
    target_size=IMG_SIZE,
    batch_size=batch_size,
    class_mode=None,
    color_mode='rgb')

train_labels_generator = train_labels_datagen.flow_from_directory(
    labels_data_path,
    target_size=IMG_SIZE,
    batch_size=batch_size,
    class_mode=None,
    color_mode='grayscale')

train_generator = zip(train_features_generator, train_labels_generator)

input_shape = (90, 90, 3)

model = Sequential()

# Encoder
model.add(Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=input_shape))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))

# Decoder
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(16, (4, 4), activation='relu'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))

model.compile(optimizer='adam', loss='binary_crossentropy')

model.fit(
    train_generator,
    steps_per_epoch=train_features_generator.samples // batch_size,
    epochs=epochs)