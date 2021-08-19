from cyclegan.layers import InstanceNormalization

from cyclegan.dataloader import DataLoader
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from cyclegan.model import CycleGAN

gan = CycleGAN()
history = gan.train(epochs=301,batch_size=16,sample_interval=30)


g_AB = load_model('/home/erik/Riksarkivet/Projects/handwritten-text-recognition/output/cycleGAN_models/overwritten/120_g_AB.h5',custom_objects={'InstanceNormalization': InstanceNormalization()})

"""data_loader = DataLoader()

imgs = data_loader.load_data(domain="A", path_to_data='/home/erik/Riksarkivet/Projects/handwritten-text-recognition/data/1930_census/Gotland/cyclegan/cyclegan_datasets/only_overwritten/trainA', is_testing=True, is_random=False, batch_size=1)

print(len(imgs))

for i, img in enumerate(imgs):
  
  img_p = img.reshape(-1, 128, 1024, 1)
  
  img_p = g_AB.predict(img_p)

  img_show = img_p[0,:,:,0]

  plt.imsave("/home/erik/Riksarkivet/Projects/handwritten-text-recognition/output/cycleGAN_models/overwritten/test_imgs/test1_%s.jpg" % str(i).zfill(2), img_show, cmap='gray')
  
  if i % 50 == 0:
    print(i)"""