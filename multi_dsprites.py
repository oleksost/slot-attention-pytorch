
import os
import random
import json
import numpy as np
from PIL import Image
import torch
import tensorflow_datasets as tfds
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
# from multi_object_datasets import multi_dsprites
# batch_size=64
# tfrecord_path = "/Users/oleksostapenko/Projects/data/multi_dsprites_colored_on_grayscale.tfrecords"
# dataset = multi_dsprites.dataset(tfrecord_path, 'colored_on_grayscale')
# batched_dataset = dataset.batch(batch_size)  # optional batching
# iterator = batched_dataset.make_one_shot_iterator()
# # data = iterator.get_next()
      
# # ds_numpy = tfds.as_numpy(dataset)
# color, image, mask, scale, shape, visibility, x, y = [],[],[],[],[],[],[],[]
# for i,batch in enumerate(iterator):
#     print(i)
#     color.append(batch['color'].numpy())
#     image.append(batch['image'].numpy())
#     mask.append(batch['mask'].numpy())
#     scale.append(batch['scale'].numpy())
#     shape.append(batch['shape'].numpy())
#     visibility.append(batch['visibility'].numpy())
#     x.append(batch['x'].numpy())
#     y.append(batch['y'].numpy())
#     if i ==5000:
#         break
# np.save('Data/multi_dsprites/color.npy',np.stack(color))
# np.save('Data/multi_dsprites/image.npy',np.stack(image))
# np.save('Data/multi_dsprites/mask.npy',np.stack(mask))
# np.save('Data/multi_dsprites/scale.npy',np.stack(scale))
# np.save('Data/multi_dsprites/shape.npy',np.stack(shape))
# np.save('Data/multi_dsprites/visibility.npy',np.stack(visibility))
# np.save('Data/multi_dsprites/x.npy',np.stack(x))
# np.save('Data/multi_dsprites/y.npy',np.stack(y))
# print('')

class Dataset_multidStripes(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.data = np.load('Data/multi_dsprites/image.npy')
        _,_,h,w,c = self.data.shape
        self.data=self.data.reshape((-1,h,w,c))
        self.len = len(self.data)
        self.transform = transforms.ToTensor()

    def __len__(self):
        return self.len
    def __getitem__(self, index):
        return self.transform(self.data[index])

if __name__=="__main__":
    ds = Dataset_tfrecord()

