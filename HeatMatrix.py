######DOCKER VERSION#######
# Defines and builds our network
#    Computes input images and labels using inputs() or distorted inputs ()
#    Computes inference on the models (forward pass) using inference()
#    Computes the total loss using loss()
#    Performs the backprop using train()

from __future__ import absolute_import  # import multi line and Absolute/Relative
from __future__ import division  # change the division operator to output float if dividing two integers
from __future__ import print_function  # use the print function from python 3

_author_ = 'simi'

import tensorflow as tf
import Input
import SODKit.SODNetwork as SDN

# Define the FLAGS class to hold our immutable global variables
FLAGS = tf.app.flags.FLAGS

# Retreive helper function object
sdn = SDN.SODMatrix()
sdloss = SDN.SODLoss(2)


def forward_pass_unet(images, phase_train):

    """
    This function builds the network architecture and performs the forward pass
    Two main architectures depending on where to insert the inception or residual layer
    :param images: Images to analyze
    :param phase_train1: bool, whether this is the training phase or testing phase
    :return: logits: the predicted age from the network
    :return: l2: the value of the l2 loss
    """

    K = 4
    images = tf.expand_dims(images, -1)

    # Network blocks
    conv1 = sdn.convolution('Conv1', images, 3, K, 1, phase_train=phase_train)
    down = sdn.convolution('Down128', conv1, 2, K*2, 2, phase_train=phase_train)

    conv2 = sdn.convolution('Conv2', down, 3, K*2, 1, phase_train=phase_train)
    conv2 = sdn.residual_layer('Conv2b', conv2, 3, K*2, 1, phase_train=phase_train)
    down = sdn.convolution('Down64', conv2, 2, K*4, 2, phase_train=phase_train)

    conv3 = sdn.residual_layer('Conv3', down, 3, K*4, 1, phase_train=phase_train)
    conv3 = sdn.residual_layer('Conv3b', conv3, 3, K*4, 1, phase_train=phase_train)
    down = sdn.convolution('Down32', conv3, 2, K*8, 2, phase_train=phase_train) # Now 32x32

    conv4 = sdn.residual_layer('Conv4', down, 3, K*8, 1, phase_train=phase_train)
    conv4 = sdn.residual_layer('Conv4b', conv4, 3, K*8, 1, phase_train=phase_train)
    down = sdn.convolution('Down16', conv4, 2, K*16, 2, phase_train=phase_train)

    conv5 = sdn.inception_layer('Conv5', down, K*16, 1, phase_train=phase_train)
    conv5 = sdn.inception_layer('Conv5b', conv5, K*16, 1, phase_train=phase_train)
    down = sdn.convolution('Down8', conv5, 2, K*32, 2, phase_train=phase_train)

    conv6 = sdn.inception_layer('Conv6', down, K*32, phase_train=phase_train)
    conv6 = sdn.inception_layer('Conv6b', conv6, K*32, phase_train=phase_train)
    down = sdn.convolution('Down4', conv6, 2, K*64, 2, phase_train=phase_train)

    # Bottom of the decoder: 4x4
    conv7 = sdn.inception_layer('Bottom1', down, K*64, phase_train=phase_train)
    conv7 = sdn.residual_layer('Bottom2', conv7, 3, K*64, 1, dropout=FLAGS.dropout_factor, phase_train=phase_train)
    conv7 = sdn.inception_layer('Bottom2', conv7, K*64, phase_train=phase_train)

    # Upsample 1
    dconv = sdn.deconvolution('Dconv1', conv7, 2, K*32, S=2, phase_train=phase_train, concat=False, concat_var=conv6, out_shape=[FLAGS.batch_size, 8, 8, K*32])
    dconv = sdn.inception_layer('Dconv1b', dconv, K*32, phase_train=phase_train)

    dconv = sdn.deconvolution('Dconv2', dconv, 2, K*16, S=2, phase_train=phase_train, concat=False, concat_var=conv5, out_shape=[FLAGS.batch_size, 16, 16, K*16])
    dconv = sdn.inception_layer('Dconv2b', dconv, K*16, phase_train=phase_train)

    dconv = sdn.deconvolution('Dconv3', dconv, 2, K*8, S=2, phase_train=phase_train, concat=False, concat_var=conv4, out_shape=[FLAGS.batch_size, 32, 32, K*8])
    dconv = sdn.inception_layer('Dconv3b', dconv, K*8, phase_train=phase_train)

    dconv = sdn.deconvolution('Dconv4', dconv, 2, K*4, S=2, phase_train=phase_train, concat=False, concat_var=conv3, out_shape=[FLAGS.batch_size, 64, 64, K*4])
    dconv = sdn.residual_layer('Dconv4b', dconv, 3, K*4, S=1, phase_train=phase_train)

    dconv = sdn.deconvolution('Dconv5', dconv, 2, K*2, S=2, phase_train=phase_train, concat=False, concat_var=conv2, out_shape=[FLAGS.batch_size, 128, 128, K*2])
    dconv = sdn.residual_layer('Dconv5b', dconv, 3, K*2, S=1, phase_train=phase_train)

    dconv = sdn.deconvolution('Dconv6', dconv, 2, K, S=2, phase_train=phase_train, concat=False, concat_var=conv1, out_shape=[FLAGS.batch_size, 256, 256, K])
    dconv = sdn.convolution('Dconv6b', dconv, 3, K, S=1, phase_train=phase_train, dropout=FLAGS.dropout_factor)

    # Output is a 1x1 box with 3 labels
    Logits = sdn.convolution('Logits', dconv, 1, FLAGS.num_classes, S=1, phase_train=phase_train, BN=False, relu=False, bias=False)

    return Logits, sdn.calc_L2_Loss(FLAGS.l2_gamma)


def total_loss(logits_tmp, labels_tmp, loss_type='COMBINED'):

    """
    Loss function.
    :param logits: Raw logits - batchx32x24x40x2
    :param labels: The labels - batchx
    :param type: Type of loss
    :return:
    """

    # reduce dimensionality
    labels, logits = tf.squeeze(labels_tmp), tf.squeeze(logits_tmp)

    # Generate the mask
    mask = tf.expand_dims(tf.cast(labels > 0, tf.float32), -1)

    # Summary images
    im_num = int(FLAGS.batch_size / 2)
    tf.summary.image('Labels', tf.reshape(tf.cast(labels[im_num], tf.float32), shape=[1, FLAGS.network_dims, FLAGS.network_dims, 1]), 2)
    tf.summary.image('Logits', tf.reshape(logits[im_num, :, :, 1], shape=[1, FLAGS.network_dims, FLAGS.network_dims, 1]), 2)
    tf.summary.image('Mask', tf.reshape(tf.cast(mask[im_num], tf.float32), shape=[1, FLAGS.network_dims, FLAGS.network_dims, 1]), 2)

    # Remove background label
    labels = tf.cast(labels > 1, tf.uint8)

    # Make labels one hot
    labels = tf.cast(tf.one_hot(labels, depth=FLAGS.num_classes, dtype=tf.uint8), tf.float32)

    if loss_type == 'DICE':

        # Get the generalized DICE loss
        loss = sdloss.dice(logits, labels)

        # Apply mask
        loss = tf.multiply(loss, mask)

        # Display loss landscape
        tf.summary.image('DICE_Landscape',
                         tf.reshape(loss[im_num], shape=[1, FLAGS.network_dims, FLAGS.network_dims, 1]), 2)

        loss = tf.reduce_mean(loss)
        tf.summary.scalar('Dice_Loss', loss)

    elif loss_type == 'WASS_DICE':

        # Get the generalized DICE loss
        loss = sdloss.generalised_wasserstein_dice_loss(labels, logits)

        # Apply mask
        loss = tf.multiply(loss, mask)

        # Display loss landscape
        tf.summary.image('Loss_Landscape1', tf.reshape(loss[im_num, :, :, 1], shape=[1, FLAGS.network_dims, FLAGS.network_dims, 1]), 2)
        tf.summary.image('Loss_Landscape0', tf.reshape(loss[im_num, :, :, 0], shape=[1, FLAGS.network_dims, FLAGS.network_dims, 1]), 2)

        loss = tf.reduce_mean(loss)
        tf.summary.scalar('WassersteinDice Loss', loss)

    elif loss_type == 'WCE':

        # Weighted CE, beta: > 1 decreases false negatives, <1 decreases false positives
        loss = sdloss.weighted_cross_entropy(logits, labels, beta=1)

        # Apply mask
        loss = tf.multiply(loss, mask)

        # Display loss landscape
        tf.summary.image('Loss_Landscape1', tf.reshape(loss[im_num, :, :, 1], shape=[1, FLAGS.network_dims, FLAGS.network_dims, 1]), 2)
        tf.summary.image('Loss_Landscape0', tf.reshape(loss[im_num, :, :, 0], shape=[1, FLAGS.network_dims, FLAGS.network_dims, 1]), 2)

        loss = tf.reduce_mean(loss)
        tf.summary.scalar('Cross_Entropy_Loss', loss)

    else:

        # Combine weighted cross entropy and DICe
        wce = sdloss.weighted_cross_entropy(logits, labels, 1)
        wce *= mask
        wce = tf.reduce_mean(wce)

        # DICE Part
        dice = sdloss.dice(logits, labels)
        dice *= mask
        dice = tf.reduce_mean(dice)

        # Add the losses with a weighting for each
        loss = wce*1 + dice*10

        # Output the summary of the MSE and MAE
        tf.summary.scalar('Cross_Entropy_Loss', wce)
        tf.summary.scalar('Dice_Loss', dice)

    # Total loss
    tf.summary.scalar('Total_loss', loss)

    # Add these losses to the collection
    tf.add_to_collection('losses', loss)

    return loss


def backward_pass(total_loss):
    """ This function performs our backward pass and updates our gradients
    Args:
        total_loss is the summed loss caculated above
        global_step1 is the number of training steps we've done to this point, useful to implement learning rate decay
    Returns:
        train_op: operation for training"""

    # Get the tensor that keeps track of step in this graph or create one if not there
    global_step = tf.train.get_or_create_global_step()

    # Print summary of total loss
    tf.summary.scalar('Total_Loss', total_loss)

    # Decay the learning rate
    dk_steps = int((FLAGS.epoch_size / FLAGS.batch_size) * 75)
    lr_decayed = tf.train.cosine_decay_restarts(FLAGS.learning_rate, global_step, dk_steps)

    # Compute the gradients. NAdam optimizer came in tensorflow 1.2
    opt = tf.contrib.opt.NadamOptimizer(learning_rate=lr_decayed, beta1=FLAGS.beta1,
                                        beta2=FLAGS.beta2, epsilon=0.1)

    # Compute the gradients
    gradients = opt.compute_gradients(total_loss)

    # Apply the gradients
    train_op = opt.apply_gradients(gradients, global_step, name='train')

    # Add histograms for the trainable variables. i.e. the collection of variables created with Trainable=True
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # Maintain average weights to smooth out training
    variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_avg_decay, global_step)

    # Applies the average to the variables in the trainable ops collection
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([train_op, variable_averages_op]):  # Wait until we apply the gradients
        dummy_op = tf.no_op(name='train')  # Does nothing. placeholder to control the execution of the graph

    return dummy_op


def inputs(training=True, skip=True):

    """
    Loads the inputs
    :param filenames: Filenames placeholder
    :param training: if training phase
    :param skip: Skip generating tfrecords if already done
    :return:
    """

    # To Do: Skip part 1 and 2 if the protobuff already exists
    if not skip:
        Input.pre_process_BRCA(FLAGS.box_dims)
        Input.pre_process_RISK(FLAGS.box_dims)
        Input.pre_process_CALCS(FLAGS.box_dims)
        Input.pre_process_PREV(FLAGS.box_dims)

    else:
        print('-------------------------Previously saved records found! Loading...')

    return Input.load_protobuf(training)
