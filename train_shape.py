import tensorflow as tf
import numpy as np
import os
import sys
sys.path.append('./utils')
sys.path.append('./models')

import dataset as dataset
import model_shape as model

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', './train_shape',
                           """Directory where to write summaries and checkpoint.""")
tf.app.flags.DEFINE_string('base_dir', './data/ShapeNetCore_im2avatar', 
                           """The path containing all the samples.""")
tf.app.flags.DEFINE_string('cat_id', '02958343', 
                           """The category id for each category: 02958343, 03001627, 03467517, 04379243""")
tf.app.flags.DEFINE_string('data_list_path', './data_list', 
                          """The path containing data lists.""")

tf.app.flags.DEFINE_integer('train_epochs', 501, """Training epochs.""")
tf.app.flags.DEFINE_integer('batch_size', 60, """Batch size.""")
tf.app.flags.DEFINE_integer('gpu', 0, """""")
tf.app.flags.DEFINE_float('learning_rate', 0.0003, """""")
tf.app.flags.DEFINE_float('wd', 0.00001, """""")
tf.app.flags.DEFINE_integer('epochs_to_save',20, """""")
tf.app.flags.DEFINE_integer('decay_step',20000, """for lr""")
tf.app.flags.DEFINE_float('decay_rate', 0.7, """for lr""")

IM_DIM = 128 
VOL_DIM = 64 

BATCH_SIZE = FLAGS.batch_size
TRAIN_EPOCHS = FLAGS.train_epochs
GPU_INDEX = FLAGS.gpu
BASE_LEARNING_RATE = FLAGS.learning_rate
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_INDEX)

TRAIN_DIR = os.path.join(FLAGS.train_dir, FLAGS.cat_id)
if not os.path.exists(TRAIN_DIR): 
  os.makedirs(TRAIN_DIR)
LOG_FOUT = open(os.path.join(TRAIN_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(tf.flags._global_parser.parse_args())+'\n')

def log_string(out_str):
  LOG_FOUT.write(out_str+'\n')
  LOG_FOUT.flush()
  print(out_str)


def get_learning_rate(batch):
  learning_rate = tf.train.exponential_decay(
                      BASE_LEARNING_RATE,  # Base learning rate.
                      batch * BATCH_SIZE,  # Current index into the dataset.
                      DECAY_STEP,          # Decay step.
                      DECAY_RATE,          # Decay rate.
                      staircase=True)
  learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
  return learning_rate 


def get_bn_decay(batch):
  bn_momentum = tf.train.exponential_decay(
                    BN_INIT_DECAY,
                    batch*BATCH_SIZE,
                    BN_DECAY_DECAY_STEP,
                    BN_DECAY_DECAY_RATE,
                    staircase=True)
  bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
  return bn_decay 


def train(dataset_):
  with tf.Graph().as_default():
    with tf.device('/gpu:'+str(GPU_INDEX)):
      is_train_pl = tf.placeholder(tf.bool)
      img_pl, vol_pl = model.placeholder_inputs(BATCH_SIZE, IM_DIM, VOL_DIM)

      global_step = tf.Variable(0)
      bn_decay = get_bn_decay(global_step)
      tf.summary.scalar('bn_decay', bn_decay)

      pred = model.get_model(img_pl, is_train_pl, weight_decay=FLAGS.wd, bn_decay=bn_decay)
      loss = model.get_MSFE_cross_entropy_loss(pred, vol_pl)
      tf.summary.scalar('loss', loss)

      learning_rate = get_learning_rate(global_step)
      tf.summary.scalar('learning_rate', learning_rate)
      optimizer = tf.train.AdamOptimizer(learning_rate)
      train_op = optimizer.minimize(loss, global_step=global_step)

      summary_op = tf.summary.merge_all()

      saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC'
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    with tf.Session(config=config) as sess:
      model_path = os.path.join(TRAIN_DIR, "trained_models")
      if tf.gfile.Exists(os.path.join(model_path, "checkpoint")):
        ckpt = tf.train.get_checkpoint_state(model_path)
        restorer = tf.train.Saver()
        restorer.restore(sess, ckpt.model_checkpoint_path)
        print ("Load parameters from checkpoint.")
      else:
        sess.run(tf.global_variables_initializer())

      train_summary_writer = tf.summary.FileWriter(model_path, graph=sess.graph)

      train_sample_size = dataset_.getTrainSampleSize()
      train_batches = train_sample_size // BATCH_SIZE # The number of batches per epoch

      val_sample_size = dataset_.getValSampleSize()
      val_batches = val_sample_size // BATCH_SIZE

      for epoch in range(TRAIN_EPOCHS):
        ####################
        # For training
        ####################
        dataset_.shuffleIds()
        for batch_idx in range(train_batches):
          imgs, vols_clr = dataset_.next_batch(batch_idx * BATCH_SIZE, BATCH_SIZE, vol_dim=VOL_DIM)
          vols_occu = np.prod(vols_clr > -0.5, axis=-1, keepdims=True) # (batch, vol_dim, vol_dim, vol_dim, 1)
          vols_occu = vols_occu.astype(np.float32)

          feed_dict = {img_pl: imgs, vol_pl: vols_occu, is_train_pl: True}

          step = sess.run(global_step)
          _, loss_val = sess.run([train_op, loss], feed_dict=feed_dict)

          log_string("<TRAIN> Epoch {} - Batch {}: loss: {}.".format(epoch, batch_idx, loss_val))

    
        #####################
        # For validation
        #####################
        loss_sum = 0.0
        for batch_idx in range(val_batches):
          imgs, vols_clr = dataset_.next_batch(batch_idx * BATCH_SIZE, BATCH_SIZE, vol_dim=VOL_DIM,  process="val")
          vols_occu = np.prod(vols_clr > -0.5, axis=-1, keepdims=True) # (batch, vol_dim, vol_dim, vol_dim, 1)
          vols_occu = vols_occu.astype(np.float32)

          feed_dict = {img_pl: imgs, vol_pl: vols_occu, is_train_pl: False}
          
          loss_val = sess.run(loss, feed_dict=feed_dict)
          loss_sum += loss_val
        log_string("<VAL> Epoch {}: loss: {}.".format(epoch, loss_sum/val_batches))

        #####################
        # Save model parameters.
        #####################
        if epoch % FLAGS.epochs_to_save == 0:
          checkpoint_path = os.path.join(model_path, 'model.ckpt')
          saver.save(sess, checkpoint_path, global_step=epoch)


def main():
  train_dataset = dataset.Dataset(base_path=FLAGS.base_dir, 
                                  cat_id=FLAGS.cat_id, 
                                  data_list_path=FLAGS.data_list_path)

  train(train_dataset)

if __name__ == '__main__':
  main()







