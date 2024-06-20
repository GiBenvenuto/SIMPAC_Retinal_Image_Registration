import tensorflow as tf
from scr.DeformUnet import STN as dUnet
from scr.config import get_config
import cv2, numpy as np
import glob
from scr import Segmentation as seg


def main():
  sess = tf.compat.v1.Session()
  config = get_config(is_train=False)


  dunet = dUnet(sess, config, "dUnet", is_train=False)
  dunet.restore(config.ckpt_ddir)


  batch_x = []
  batch_y = []

  img = cv2.imread("data/S10_1.jpg")
  img = seg.segmentation(img=img)
  #img = img[:, :, 1]
  img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
  img = img.reshape(img.shape + (1,))
  img = img / 255
  batch_x.append(img)
  batch_x = np.array(batch_x)

  img = cv2.imread("data/S10_2.jpg")
  img = seg.segmentation(img=img)
  #img = img[:, :, 1]
  img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
  img = img.reshape(img.shape + (1,))
  img = img / 255
  batch_y.append(img)
  batch_y = np.array(batch_y)

  result_dir = "result"


  dunet.deploy(result_dir, batch_x, batch_y, 1)




if __name__ == "__main__":
   main()
