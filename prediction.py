from keras.preprocessing import image
from keras.models import load_model
import numpy as np

test_image = image.load_img('prediction/cat.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

classifier = load_model('model.h5')
result = classifier.predict(test_image)

if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'