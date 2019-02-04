from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

model = VGG16()

from PIL import Image
import requests
from io import BytesIO

url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/1/14/Gatto_europeo4.jpg/250px-Gatto_europeo4.jpg'
response = requests.get(url)
image = Image.open(BytesIO(response.content))
image = image.crop((0, 0, 244, 244))
image.save('cat.jpg')

image = load_img('cat.jpg', target_size=(224, 224))
image = img_to_array(image)
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
image = preprocess_input(image)
yhat = model.predict(image)
label = decode_predictions(yhat)
label = label[0][0]
print('{} ({})'.format(label[1], label[2] * 100))

import keract

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
activations = keract.get_activations(model, image)
first = activations.get('block1_conv1/Relu:0')
keract.display_activations(activations)
#
# import matplotlib.pyplot as plt
#
# fig = plt.figure(figsize=(12, 12))
#
# rows = 8
# columns = 8
#
# first = activations.get('block1_conv1/Relu:0')
#
# for i in range(1, columns * rows + 1):
#     img = first[0, :, :, i - 1]
#     fig.add_subplot(rows, columns, i)
#     plt.imshow(img)
#     plt.axis('off')
# plt.show()
