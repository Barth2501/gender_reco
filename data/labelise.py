import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

ds = tfds.load('celeb_a', split='train',shuffle_files=True)
ds = ds.batch(1)
count=1
for example in tfds.as_numpy(ds):
    image, label = example["image"], example["attributes"]
    if label['Male'][0]==True:
        image.save('./{}/male_{}.jpg'.format('male',count))
    else:
        image.save('./{}/female_{}.jpg'.format('female',count))
    count+=1
    if count==2:
        break