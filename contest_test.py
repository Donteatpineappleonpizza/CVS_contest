from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
import cv2

BATCH_SIZE = 32
IMAGE_SIZE = (64, 64)

# Load the test dataset
dataframe = pd.read_csv('/Users/piggymaruuu/Desktop/Study 📖/Comvision2/computer_vision/fried_noodles_dataset.csv', delimiter=',', header=0)
dataframe["norm_meat"] = dataframe["meat"] / 300
dataframe["norm_veggie"] = dataframe["veggie"] / 300
dataframe["norm_noodle"] = dataframe["noodle"] / 300

# Image data generator for testing without augmentation
datagen_noaug = ImageDataGenerator(rescale=1./255)

test_generator = datagen_noaug.flow_from_dataframe(
    dataframe=dataframe.loc[1700:],
    directory='/Users/piggymaruuu/Desktop/Study 📖/Comvision2/computer_vision/images',
    x_col='filename',
    y_col=['norm_meat', 'norm_veggie', 'norm_noodle'],
    shuffle=False,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='other'
)

# Load the model
model = load_model('noodles_fried.h5')

# Evaluate the model on the test set
score = model.evaluate_generator(
    test_generator,
    steps=len(test_generator)
)
print('Test Score (MSE, MAE):', score)

# Predict on the test set
test_generator.reset()
predictions = model.predict_generator(
    test_generator,
    steps=len(test_generator),
    workers=1,
    use_multiprocessing=False
)
print('Predictions:\n', predictions)

# Example prediction on a single image
imgfile = '/Users/piggymaruuu/Desktop/Study 📖/Comvision2/computer_vision/images/2019_11_08-fried_noodles-set_1-1.jpg'
test_im = cv2.imread(imgfile, cv2.IMREAD_COLOR)
test_im = cv2.cvtColor(test_im, cv2.COLOR_BGR2RGB)
test_im = cv2.resize(test_im, IMAGE_SIZE)
test_im = test_im / 255.
test_im = np.expand_dims(test_im, axis=0)
w_pred = model.predict(test_im)
print(imgfile, "Prediction:", w_pred[0][0] * 300, "grams")
# Assuming 'predictions' contains the predicted values from the model
result_df = pd.DataFrame(predictions * 300, columns=['meat', 'veggie', 'noodle'])

# Add the filenames to the result_df
result_df['filename'] = test_generator.filenames

# Reorder columns to match 'fried_noodles_dataset.csv'
result_df = result_df[['filename', 'meat', 'veggie', 'noodle']]

# Save the result dataframe to 'result.csv'
result_df.to_csv('result.csv', index=False)

print("Result exported to result.csv:")
print(result_df.head())