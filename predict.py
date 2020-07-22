import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from keras.applications import VGG16
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.applications import imagenet_utils
import numpy as np
import pickle
import argparse
from urllib.request import urlopen
from PIL import Image
from cv2 import resize

parser = argparse.ArgumentParser()
parser.add_argument("-path", required=True, help="Path of image file to predict.")
args = parser.parse_args()

class_name = 'categories.txt'
classes = list()
with open(class_name) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ')[0][3:])
classes = tuple(classes)

image_path = args.path
try: 
  image = load_img(image_path,target_size=(224,224))
  image=  img_to_array(image)
  image = np.expand_dims(image,0)
  image = imagenet_utils.preprocess_input(image)
except:
  image = Image.open(urlopen(image_path))
  image = np.array(image, dtype=np.uint8)
  image = resize(image, (224, 224))
  image = np.expand_dims(image, 0)



filename = 'logistic_model.sav'
logistic_model = pickle.load(open(filename, 'rb'))
base_model = VGG16(weights='imagenet', include_top = False)
features = base_model.predict(image)
features = features.reshape((features.shape[0], 512*7*7))

predictions_to_return = 5
preds = logistic_model.predict_proba(features)[0]
top_preds = np.argsort(preds)[::-1][0:predictions_to_return]
print('--PREDICTED SCENE CATEGORIES:')
for i in range(0, predictions_to_return):
  print(classes[top_preds[i]])