from glob import glob
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
import cv2
import os

class DataLoader():
    def __init__(self, img_res=(128, 1024)):
        self.img_res = img_res

    def load_data(self, domain, path_to_data, batch_size=1, is_testing=False, is_random = True):
        data_type = "train%s" % domain if not is_testing else "test%s" % domain
        paths_to_imgs = glob(os.path.join(path_to_data, '**'))
        
        paths_to_imgs.sort() #patch for demo

        if is_random: batch_images = np.random.choice(paths_to_imgs, size=batch_size, replace=False)
        else: batch_images = paths_to_imgs
        
        imgs = []
        for img_path in batch_images:
            img = self.load_img(img_path) #, color_mode='grayscale', target_size=(128, 1024))
            #img = image.img_to_array(img).astype('float32')
            #img = img / 255.0
            if not is_testing and is_random:

                if np.random.random() > 0.5:
                    img = np.fliplr(img)

              
            imgs.append(img)

        imgs = np.array(imgs)

        return imgs

    def load_batch(self, batch_size=1, is_testing=False):
        data_typeA = "trainA" if not is_testing else "testA"
        data_typeB = "trainB" if not is_testing else "testB"
        path_A = glob('/home/erik/Riksarkivet/Projects/handwritten-text-recognition/data/1930_census/Gotland/cyclegan/cyclegan_datasets/only_overwritten/%s/*' % (data_typeA))
        path_B = glob('/home/erik/Riksarkivet/Projects/handwritten-text-recognition/data/1930_census/Gotland/cyclegan/cyclegan_datasets/only_overwritten/%s/*' % (data_typeB))
        
        self.n_batches = int(min(len(path_A), len(path_B)) / batch_size)
        total_samples = self.n_batches * batch_size

        path_A = np.random.choice(path_A, total_samples, replace=False)
        path_B = np.random.choice(path_B, total_samples, replace=False)
        
        

        for i in range(self.n_batches):
            batch_A = path_A[i*batch_size:(i+1)*batch_size]
            batch_B = path_B[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B = [], []
            for img_A, img_B in zip(batch_A, batch_B):
                
                img_A = self.load_img(img_A)
                #img_A = image.img_to_array(img_A).astype('float32')
                #img_A = img_A / 255.0
                
                img_B = self.load_img(img_B)
                #img_B = image.img_to_array(img_B).astype('float32')
                #img_B = img_B / 255.0

                if not is_testing and np.random.random() > 0.5:
                        img_A = np.fliplr(img_A)
                        img_B = np.fliplr(img_B)

                imgs_A.append(img_A)
                imgs_B.append(img_B)

            imgs_A = np.array(imgs_A)
            imgs_B = np.array(imgs_B)

            yield imgs_A, imgs_B

    def load_img(self, path):
        img = self.imread(path)
        #img.resize(self.img_res)
        img = img/255.0
        return img[np.newaxis, :, :]

    def imread(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        u, i = np.unique(np.array(img).flatten(), return_inverse=True)
        bg = int(u[np.argmax(np.bincount(i))])

        wt, ht, _ = (1024, 128, 1)
        h, w = np.asarray(img).shape
        f = max((w / wt), (h / ht))

        new_size = (max(min(wt, int(w / f)), 1), max(min(ht, int(h / f)), 1))
        img = cv2.resize(img, new_size)

        target = np.ones([ht, wt], dtype=np.uint8) * bg
        target[0:new_size[1], 0:new_size[0]] = img

        target = target.astype(np.float)

        return target

