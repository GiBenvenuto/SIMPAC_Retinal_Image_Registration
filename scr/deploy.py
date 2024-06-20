import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt
from FundusData import FundusDataHandler
from DeformUnet import STN as dUnet
from config import get_config
from ops import mkdir
import plots as p




def main():

  sess = tf.compat.v1.Session()
  config = get_config(is_train=False)
  mkdir(config.result_ddir)


  dh = FundusDataHandler(is_train=False, is_lbp=False, im_size=config.im_size, db_size=config.db_size, db_type=0)
  dunet = dUnet(sess, config, "dUnet", is_train=False)
  dunet.restore(config.ckpt_ddir)


  result_i_dir = config.result_ddir+"/{}".format(3)
  mkdir(result_i_dir)
  for j in range(85):
      batch_x, batch_y = dh.sample_pair_bases(j)
      #dunet.varlist(batch_x, batch_y)
      dunet.deploy(result_i_dir, batch_x, batch_y, j)


if __name__ == "__main__":
   main()
