from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Flatten, Input, BatchNormalization, Conv2D, MaxPool2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback, ModelCheckpoint
import matplotlib.pyplot as plt
import pandas as pd

# Constants
BATCH_SIZE = 32
IMAGE_SIZE = (64, 64)


# Load dataset
dataframe = pd.read_csv('/Users/piggymaruuu/Desktop/Study 📖/Comvision2/computer_vision/fried_noodles_dataset.csv', delimiter=',', header=0)
dataframe["norm_meat"] = dataframe["meat"] / 300
dataframe["norm_veggie"] = dataframe["veggie"] / 300
dataframe["norm_noodle"] = dataframe["noodle"] / 300

# Image data generator
datagen = ImageDataGenerator(rescale=1./255)

# Data generators
def create_data_generator(subset, shuffle=True):
    return datagen.flow_from_dataframe(
        dataframe=dataframe.loc[subset],
        directory='/Users/piggymaruuu/Desktop/Study 📖/Comvision2/computer_vision/images',
        x_col='filename',
        y_col=['norm_meat', 'norm_veggie', 'norm_noodle'],
        shuffle=shuffle,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='other'
    )

train_generator = create_data_generator(slice(0, 1599))
validation_generator = create_data_generator(slice(1600, 1699), shuffle=False)
test_generator = create_data_generator(slice(1700, None), shuffle=False)

# Model architecture
inputIm = Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3,))
conv1 = Conv2D(64, 3, activation='relu')(inputIm)
conv1 = Conv2D(64, 3, activation='relu')(conv1)
conv1 = BatchNormalization()(conv1)
pool1 = MaxPool2D()(conv1)
conv2 = Conv2D(128, 3, activation='relu')(pool1)
conv2 = Conv2D(128, 3, activation='relu')(conv2)
conv2 = BatchNormalization()(conv2)
pool2 = MaxPool2D()(conv2)
conv3 = Conv2D(256, 3, activation='relu')(pool2)
conv3 = Conv2D(256, 3, activation='relu')(conv3)
conv3 = BatchNormalization()(conv3)
pool3 = MaxPool2D()(conv3)
flat = Flatten()(pool3)
dense1 = Dense(512, activation='relu')(flat)
dense1 = Dropout(0.5)(dense1)
dense1 = Dense(512, activation='relu')(dense1)
dense1 = Dropout(0.5)(dense1)
dense1 = Dense(512, activation='relu')(dense1)
dense1 = Dropout(0.5)(dense1)
predictedW = Dense(3, activation='sigmoid')(dense1)

model = Model(inputs=inputIm, outputs=predictedW)
model.compile(optimizer=Adam(lr=1e-4), loss='mse', metrics=['mean_absolute_error'])

class PlotLosses(Callback):
    def on_train_begin(self, logs={}):
        self.x = []
        self.losses = []
        self.val_losses = []
        self.fig = plt.figure()
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(epoch)
        self.losses.append(logs.get('mean_absolute_error'))
        self.val_losses.append(logs.get('val_mean_absolute_error'))

        plt.clf()
        plt.plot(self.x, self.losses, label='mean_absolute_error')
        plt.plot(self.x, self.val_losses, label='val_mean_absolute_error')
        plt.legend()
        plt.pause(0.01)


# Callbacks
checkpoint = ModelCheckpoint('noodles_fried.h5', verbose=1, monitor='val_mean_absolute_error', save_best_only=True, mode='min')
plot_losses = PlotLosses()

# Train Model
model.fit_generator(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=40,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=[checkpoint, plot_losses]
)

# Test Model
model = load_model('noodles_fried.h5')
score = model.evaluate(
    test_generator,
    steps=len(test_generator)
)
print('score (mse, mae):\n', score)

test_generator.reset()
predict = model.predict(
    test_generator,
    steps=len(test_generator),
    workers=1,
    use_multiprocessing=False
)
print('prediction:\n', predict)
