from io import BytesIO

import requests
from PIL import Image
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import decode_predictions
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array

import keract

model = InceptionV3()

url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/1/14/Gatto_europeo4.jpg/250px-Gatto_europeo4.jpg'
response = requests.get(url)
image = Image.open(BytesIO(response.content))
image = image.crop((0, 0, 299, 299))
image = img_to_array(image)
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
image = preprocess_input(image)
yhat = model.predict(image)
label = decode_predictions(yhat)
label = label[0][0]
print('{} ({})'.format(label[1], label[2] * 100))  # a tabby is a cat!

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
activations = keract.get_activations(model, image)
keract.display_activations(activations)
