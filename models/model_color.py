import tensorflow as tf
import sys
sys.path.append('../utils')

import tf_utils

def placeholder_inputs(batch_size, im_dim, vol_dim):
  img_pl = tf.placeholder(tf.float32, shape=(batch_size, im_dim, im_dim, 3))
  vol_clr_pl = tf.placeholder(tf.float32, shape=(batch_size, vol_dim, vol_dim, vol_dim, 3))
  vol_flow_pl = tf.placeholder(tf.float32, shape=(batch_size, vol_dim, vol_dim, vol_dim, 2))
  return img_pl, vol_clr_pl, vol_flow_pl


def get_loss(regressed_clr, blended_clr, target_clr, pred_flow, target_flow):
  ''' calculate the loss by using regressed color, regressed flow, soft-blended color (confidence, regressed color, sampled color)

      Args:
        regressed_clr: (batch, vol_dim, vol_dim, vol_dim, 3).
        blended_clr: (batch, vol_dim, vol_dim, vol_dim, 3).
        target_clr: (batch, vol_dim, vol_dim, vol_dim, 3). -1 for empty voxels, [0, 1] for occupied voxels
        pred_flow: (batch, vol_dim, vol_dim, vol_dim, 2).
        target_flow: (batch, vol_dim, vol_dim, vol_dim, 2). -1 for empty voxels, [0, 1] for occupied voxels
      Rrturns:
        The total loss including l2 loss and regularization terms.
  '''
  num_pos_clr = tf.shape(tf.where(target_clr > -0.5))[0] # int
  num_pos_clr = tf.to_float(num_pos_clr)

  num_pos_flow = tf.shape(tf.where(target_flow > -0.5))[0] # int
  num_pos_flow = tf.to_float(num_pos_flow)

  ###################################
  # calculate positive loss for regrssed color
  regressed_clr_vol_loss = tf.abs(regressed_clr - target_clr)
  regressed_clr_pos_loss = tf.where(target_clr > -0.5, regressed_clr_vol_loss, tf.zeros_like(regressed_clr_vol_loss)) 
  regressed_clr_pos_loss = tf.reduce_sum(regressed_clr_pos_loss) / num_pos_clr

  # calculate positive loss for soft_blended color
  blended_clr_vol_loss = tf.abs(blended_clr - target_clr)
  blended_clr_pos_loss = tf.where(target_clr > -0.5, blended_clr_vol_loss, tf.zeros_like(blended_clr_vol_loss)) 
  blended_clr_pos_loss = tf.reduce_sum(blended_clr_pos_loss) / num_pos_clr

  ##################################
  # Calculate positive loss for flow
  flow_vol_loss = tf.abs(pred_flow - target_flow)
  flow_pos_loss = tf.where(target_flow > -0.5, flow_vol_loss, tf.zeros_like(flow_vol_loss))
  flow_pos_loss = tf.reduce_sum(flow_pos_loss) / num_pos_flow

  tf.add_to_collection('losses', regressed_clr_pos_loss + blended_clr_pos_loss + flow_pos_loss)
  
  return tf.add_n(tf.get_collection('losses'), name='total_loss')


def get_model(imgs, is_training, weight_decay=0.0, bn_decay=None):
  batch_size = imgs.get_shape()[0].value
  im_dim = imgs.get_shape()[1].value

  ########
  with tf.variable_scope('Encoding'):
    # (batch_size, 64, 64, 64)
    net = tf_utils.conv2d(imgs, 64, [7,7],
                         padding='SAME', stride=[2,2],
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay, 
                         weight_decay=weight_decay, activation_fn=tf.nn.elu)
    # (batch_size, 32, 32, 64)
    net = tf_utils.conv2d(net, 64, [5,5],
                         padding='SAME', stride=[2,2],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay, 
                         weight_decay=weight_decay, activation_fn=tf.nn.elu)
    # (batch_size, 16, 16, 128)
    net = tf_utils.conv2d(net, 128, [5,5],
                         padding='SAME', stride=[2,2],
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay, 
                         weight_decay=weight_decay, activation_fn=tf.nn.elu)
    # (batch_size, 8, 8, 128)
    net = tf_utils.conv2d(net, 128, [3,3],
                         padding='SAME', stride=[2,2],
                         bn=True, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay, 
                         weight_decay=weight_decay, activation_fn=tf.nn.elu)
    # (batch_size, 4, 4, 256)
    net = tf_utils.conv2d(net, 256, [3,3],
                         padding='SAME', stride=[2,2],
                         bn=True, is_training=is_training,
                         scope='conv5', bn_decay=bn_decay, 
                         weight_decay=weight_decay, activation_fn=tf.nn.elu)
    # (batch_size, 1, 1, 512)
    net = tf_utils.conv2d(net, 512, [4,4],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv6', bn_decay=bn_decay, 
                         weight_decay=weight_decay, activation_fn=tf.nn.elu)

  ########
  with tf.variable_scope('Latent_variable'):
    net = tf.reshape(net, [batch_size, 512])
    net = tf_utils.fully_connected(net, 512, scope="fc1", 
                    weight_decay=weight_decay, activation_fn=tf.nn.elu, 
                    bn=True, bn_decay=bn_decay, is_training=is_training)
    net = tf_utils.fully_connected(net, 128*4*4*4, scope="fc2", 
                    weight_decay=weight_decay, activation_fn=tf.nn.elu, 
                    bn=True, bn_decay=bn_decay, is_training=is_training)
    net = tf.reshape(net, [batch_size, 4, 4, 4, 128])

  ########
  with tf.variable_scope('Decoding'):
    # (batch_size, 8, 8, 8, 64)
    net = tf_utils.conv3d_transpose(net, 64, [3, 3, 3], scope="deconv1",
                     stride=[2, 2, 2], padding='SAME',
                     weight_decay=weight_decay, activation_fn=tf.nn.elu,
                     bn=True, bn_decay=bn_decay, is_training=is_training)
    
    # (batch_size, 16, 16, 16, 32)
    net = tf_utils.conv3d_transpose(net, 32, [3, 3, 3], scope="deconv2",
                     stride=[2, 2, 2], padding='SAME',
                     weight_decay=weight_decay, activation_fn=tf.nn.elu,
                     bn=True, bn_decay=bn_decay, is_training=is_training)
    
    # (batch_size, 32, 32, 32, 32)
    net = tf_utils.conv3d_transpose(net, 32, [3, 3, 3], scope="deconv3",
                     stride=[2, 2, 2], padding='SAME',
                     weight_decay=weight_decay, activation_fn=tf.nn.elu,
                     bn=True, bn_decay=bn_decay, is_training=is_training)

    ##################
    ## regressed color
    ##################
    # (batch_size, 64, 64, 64, 24)
    net_reg_clr = tf_utils.conv3d_transpose(net, 16, [3, 3, 3], scope="deconv_reg_clr1",
                     stride=[2, 2, 2], padding='SAME',
                     weight_decay=weight_decay, activation_fn=tf.nn.elu,
                     bn=True, bn_decay=bn_decay, is_training=is_training)
    # (batch_size, 64, 64, 64, 3)
    net_reg_clr = tf_utils.conv3d(net_reg_clr, 3, [3, 3, 3], scope="deconv_reg_clr2",
                     stride=[1, 1, 1], padding='SAME',
                     weight_decay=weight_decay, activation_fn=tf.sigmoid,
                     bn=True, bn_decay=bn_decay, is_training=is_training)

    ##############
    ### confidence
    ############## 
    net_conf = tf_utils.conv3d_transpose(net, 16, [3, 3, 3], scope="deconv_conf1",
                     stride=[2, 2, 2], padding='SAME',
                     weight_decay=weight_decay, activation_fn=tf.nn.elu,
                     bn=True, bn_decay=bn_decay, is_training=is_training)
     # (batch_size, 64, 64, 64, 1)
    net_conf = tf_utils.conv3d(net_conf, 1, [3, 3, 3], scope="conv_conf2",
                     stride=[1, 1, 1], padding='SAME',
                     weight_decay=weight_decay, activation_fn=tf.sigmoid,
                     bn=True, bn_decay=bn_decay, is_training=is_training)

    ##############
    ### flow color
    ##############
    net_flow = tf_utils.conv3d_transpose(net, 2, [3, 3, 3], scope="deconv_flow",
                     stride=[2, 2, 2], padding='SAME',
                     weight_decay=weight_decay, activation_fn=tf.sigmoid,
                     bn=True, bn_decay=bn_decay, is_training=is_training)
    # (batch_size, 64, 64, 64, 3)
    net_flow_clr = tf_utils.Sampler(net_flow, imgs)

    #################
    ### blended color
    #################
    net_blended_clr = net_reg_clr * net_conf + net_flow_clr * (1.0 - net_conf)


  return net_reg_clr, net_conf, net_flow, net_blended_clr


if __name__=='__main__':
  with tf.Graph().as_default():
    batch_size = 3
    im_dim = 128
    vol_dim = 64

    img_pl, vol_clr_pl, vol_flow_pl = placeholder_inputs(batch_size, im_dim, vol_dim)
    pred_reg_clr, pred_conf, pred_flow, pred_blended_clr = get_model(img_pl, tf.constant(True))
    print pred_reg_clr
    print pred_conf
    print pred_flow
    print pred_blended_clr

    loss = get_loss(pred_reg_clr, pred_blended_clr, vol_clr_pl, pred_flow, vol_flow_pl)
    print loss

