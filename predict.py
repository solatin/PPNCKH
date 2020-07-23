import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from keras.applications import VGG16
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.applications import imagenet_utils
from vgg16_hybrid_places_1365 import VGG16_Hybrid_1365
import numpy as np
import pickle
import argparse
from urllib.request import urlopen
from PIL import Image
from cv2 import resize

parser = argparse.ArgumentParser()
parser.add_argument("-image_path", required=True, help="Path of image file to predict.")
parser.add_argument("-model_path", required=True, help="Path of logistic model file.")
parser.add_argument("-base_model", required=False, help="VGG16 BaseModel: places365 or imagenet .")
args = parser.parse_args()


class_name = 'categories.txt'
classes = list()
with open(class_name) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ')[0][3:])
classes = tuple(classes)

image_path = args.image_path
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

filename = args.model_path
logistic_model = pickle.load(open(filename, 'rb'))
if args.base_model == "places365":
  base_model = VGG16_Hybrid_1365(weights='vgg16-hybrid1365_weights_notop.h5', include_top = False)
if args.base_model == "imagenet":
  base_model = VGG16(weights='imagenet', include_top = False)

features = base_model.predict(image)
features = features.reshape((features.shape[0], 512*7*7))
predictions_to_return = 5
proba_preds = logistic_model.predict_proba(features)[0]
top_preds = np.argsort(proba_preds)[::-1][0:predictions_to_return]
print('--PREDICTED SCENE CATEGORIES:')
for i in range(0, predictions_to_return):
  print(classes[top_preds[i]], proba_preds[top_preds[i]])
