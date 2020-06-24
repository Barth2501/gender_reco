import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from PIL import Image


ds = tfds.load('celeb_a', split='train',shuffle_files=True)
ds = ds.batch(1)
count=1
for example in tfds.as_numpy(ds):
    image, label = example["image"], example["attributes"]
    image = Image.fromarray(image[0])
    if label['Male'][0]==True:
        image.save('./{}/male_{}.jpg'.format('male',count))
    else:
        image.save('./{}/female_{}.jpg'.format('female',count))
    count+=1
    print(count)
    if count==5000:
        break