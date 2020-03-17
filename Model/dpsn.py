import tensorflow as tf
import numpy as np
import os
import timeit
from __future__ import division
from numpy import ogrid, repeat, newaxis
from skimage import io
import skimage.transform

# Pad with 0 values, similar to how Tensorflow does it. Order=1 is bilinear upsampling
def upsample_skimage(factor, input_img):
    return skimage.transform.rescale(input_img, factor, mode='constant', cval=0, order=1)

# Find the kernel size given the desired factor of upsampling
def get_kernel_size(factor):
    return 2 * factor - factor % 2

# Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size
def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)

# Create weights matrix for transposed convolution with bilinear filter initialization
def bilinear_upsample_weights(factor, number_of_classes):
    filter_size = get_kernel_size(factor)

    weights = np.zeros((filter_size,
                        filter_size,
                        number_of_classes,
                        number_of_classes), dtype=np.float32)

    upsample_kernel = upsample_filt(filter_size)

    for i in range(number_of_classes):
        weights[:, :, i, i] = upsample_kernel

    return weights


def upsample_tf(factor, input_img):
    number_of_classes = input_img.shape[2]

    new_height = input_img.shape[0] * factor
    new_width = input_img.shape[1] * factor

    expanded_img = np.expand_dims(input_img, axis=0)

    # upsample_filter_np = bilinear_upsample_weights(factor, number_of_classes)

    with tf.Graph().as_default():
        with tf.Session() as sess:
            with tf.device("/cpu:0"):
                upsample_filt_pl = tf.placeholder(tf.float32)
                logits_pl = tf.placeholder(tf.float32)

                upsample_filter_np = bilinear_upsample_weights(factor, number_of_classes)

                res = tf.nn.conv2d_transpose(logits_pl, upsample_filt_pl,
                                             output_shape=[1, new_height, new_width, number_of_classes],
                                             strides=[1, factor, factor, 1])
                final_result = sess.run(res, feed_dict={upsample_filt_pl: upsample_filter_np,
                                                        logits_pl: expanded_img})
    return final_result.squeeze()


def upsample(input, factor, channel=1):
    # upsample_weight
    upsample_filter_np = bilinear_upsample_weights(factor, channel)
    # Convert to a Tensor type
    upsample_filter_tensor = tf.constant(upsample_filter_np)
    down_shape = tf.shape(input)
    # Calculate the output size of the upsampled tensor here only has a shape
    up_shape = tf.stack([down_shape[0],
                         down_shape[1] * factor,
                         down_shape[2] * factor,
                         down_shape[3]])

    upsampled_input = tf.nn.conv2d_transpose(input, upsample_filter_tensor,
                                             output_shape=up_shape,
                                             strides=[1, factor, factor, 1])
    upsampled_input = tf.reshape(upsampled_input,
                                 [-1, up_shape[1], up_shape[2], channel])
    return upsampled_input


def data_norm(images, masks):
    images, masks = np.array(images, dtype=np.float32), np.array(masks, dtype=np.float32)
    for i in range(images.shape[0]):
        images[i, :, :] -= np.mean(images[i, :, :])
        images[i, :, :] /= (np.std(images[i, :, :]) + 1e-12)
    masks = masks / 255.0
    return images, masks


def conv2d(tensor, in_ch, out_ch, k_size, bias=False):
    initial = tf.contrib.layers.xavier_initializer_conv2d()
    weights = tf.Variable(initial([k_size, k_size, in_ch, out_ch]))
    tensor = tf.nn.conv2d(tensor, weights, strides=[1, 1, 1, 1], padding='SAME')
    if bias is True:
        bias_tmp = tf.Variable(tf.constant(0.01, shape=[out_ch]))
        tensor = tensor + bias_tmp
    return tensor


def dilated_conv2d(tensor, in_ch, out_ch, k_size, d_rate, bias=False):
    initial = tf.contrib.layers.xavier_initializer_conv2d()
    weights = tf.Variable(initial([k_size, k_size, in_ch, out_ch]))
    tensor = tf.nn.atrous_conv2d(tensor, weights, d_rate, padding='SAME')
    if bias is True:
        bias_tmp = tf.Variable(tf.constant(0.01, shape=[out_ch]))
        tensor = tensor + bias_tmp
    return tensor


def batch_norm(tensor, is_train, center=True, scale=True, epsilon=0.001, decay=0.95):
    shape = tensor.get_shape().as_list()
    l_axes = list(range(len(shape) - 1))
    mean, var = tf.nn.moments(tensor, l_axes)
    ema = tf.train.ExponentialMovingAverage(decay=decay)

    def mean_var_with_update():
        ema_apply_op = ema.apply([mean, var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(mean), tf.identity(var)

    mean, var = tf.cond(is_train, mean_var_with_update,
                        lambda: (ema.average(mean), ema.average(var)))
    if center is True:
        shift_v = tf.Variable(tf.zeros(shape[-1]))
    else:
        shift_v = None
    if scale is True:
        scale_v = tf.Variable(tf.ones(shape[-1]))
    else:
        scale_v = None
    tensor = tf.nn.batch_normalization(tensor, mean, var, shift_v, scale_v, epsilon)
    return tensor


def bn_act_dconv(tensor, in_ch, out_ch, k_size, d_rate, is_train, keep):
    tensor = batch_norm(tensor, is_train=is_train)
    tensor = tf.nn.relu(tensor)
    tensor = dilated_conv2d(tensor, in_ch, out_ch, k_size, d_rate)
    tensor = tf.nn.dropout(tensor, keep_prob=keep)
    return tensor


def bn_act_conv(tensor, in_ch, out_ch, k_size, is_train, keep):
    tensor = batch_norm(tensor, is_train=is_train)
    tensor = tf.nn.relu(tensor)
    tensor = conv2d(tensor, in_ch, out_ch, k_size)
    tensor = tf.nn.dropout(tensor, keep_prob=keep)
    return tensor


def block(tensor, layers, in_ch, growth_ch, d_rate, is_train, keep):
    for idx in range(layers):
        tmp = bn_act_dconv(tensor, in_ch, growth_ch, k_size=3, d_rate=d_rate, is_train=is_train, keep=keep)
        tensor = tf.concat((tensor, tmp), axis=3)
        in_ch += growth_ch
    return tensor, in_ch


def ppm64(tensor, in_ch, out_ch=1):
    tensor1 = tf.nn.avg_pool(tensor, ksize=[1, 64, 64, 1], strides=[1, 64, 64, 1], padding='VALID')
    tensor1 = conv2d(tensor1, in_ch, out_ch, k_size=1)
    tensor1 = upsample(tensor1, 64, out_ch)

    tensor2 = tf.nn.avg_pool(tensor, ksize=[1, 32, 32, 1], strides=[1, 32, 32, 1], padding='VALID')
    tensor2 = conv2d(tensor2, in_ch, out_ch, k_size=1)
    tensor2 = upsample(tensor2, 32, out_ch)

    tensor3 = tf.nn.avg_pool(tensor, ksize=[1, 21, 21, 1], strides=[1, 21, 21, 1], padding='VALID')
    tensor3 = conv2d(tensor3, in_ch, out_ch, k_size=1)
    tensor3 = upsample(tensor3, 21, out_ch)
    tensor3 = tf.pad(tensor3, [[0, 0], [0, 1], [0, 1], [0, 0]], mode="SYMMETRIC")

    tensor4 = tf.nn.avg_pool(tensor, ksize=[1, 10, 10, 1], strides=[1, 10, 10, 1], padding='VALID')
    tensor4 = conv2d(tensor4, in_ch, out_ch, k_size=1)
    tensor4 = upsample(tensor4, 10, out_ch)
    tensor4 = tf.pad(tensor4, [[0, 0], [2, 2], [2, 2], [0, 0]], mode="SYMMETRIC")

    tensor = tf.concat((tensor1, tensor2, tensor3, tensor4), axis=3)
    tensor = tf.reshape(tensor, [-1, 64, 64, 4 * out_ch])
    return tensor, out_ch * 4


def coefficients(gt, pred, smooth=1e-12):
    pred = tf.cast(tf.greater(pred, 0.5), dtype=tf.float32)
    intersection = tf.reduce_sum(gt * pred)
    gt, pred = tf.reduce_sum(gt), tf.reduce_sum(pred)
    union = gt + pred - intersection

    precision = intersection / (pred + smooth)
    recall = intersection / (gt + smooth)

    beta_square = 0.3
    f_beta_coeff = (1 + beta_square) * precision * recall / (beta_square * precision + recall + smooth)
    dice_coeff = (2. * intersection) / (union + intersection + smooth)
    jaccard_coeff = intersection / (union + smooth)
    return dice_coeff, jaccard_coeff, f_beta_coeff


def fuse_loss(gt, pred, smooth=1e-12):
    gt_back, pred_back = (1 - gt), (1 - pred)

    mae_loss = tf.reduce_mean(tf.log(1 + tf.exp(tf.abs(gt - pred))))

    alpha, beta = 1 / (tf.pow(tf.reduce_sum(gt), 2) + smooth), 1 / (tf.pow(tf.reduce_sum(gt_back), 2) + smooth)
    numerator = alpha * tf.reduce_sum(gt * pred) + beta * tf.reduce_sum(gt_back * pred_back)
    denominator = alpha * tf.reduce_sum(gt + pred) + beta * tf.reduce_sum(gt_back + pred_back)
    dice_loss = 1 - 2 * numerator / (denominator + smooth)

    w = (image_size * image_size * batch_size - tf.reduce_sum(pred)) / (tf.reduce_sum(pred) + smooth)
    cross_entropy_loss = -tf.reduce_mean(key_w * w * gt * tf.log(pred + smooth) + gt_back * tf.log(pred_back + smooth))
    return mae_loss + dice_loss + cross_entropy_loss


def run_in_batch_avg(session, tensors, placeholders, feed_dict):
    res = [0] * (len(tensors) - 1)
    batch_tensors = [(placeholder, feed_dict[placeholder]) for placeholder in placeholders]
    total_size = len(batch_tensors[0][1])
    batch_count_ = int((total_size + batch_size - 1) / batch_size)

    for batch_idx_ in range(batch_count_):
        current_batch_size = None

        for (placeholder, tensor) in batch_tensors:
            batch_tensor = tensor[batch_idx_ * batch_size: (batch_idx_ + 1) * batch_size]
            current_batch_size = len(batch_tensor)
            feed_dict[placeholder] = tensor[batch_idx_ * batch_size: (batch_idx_ + 1) * batch_size]

        feed_dict[placeholders[0]], feed_dict[placeholders[1]] = data_norm(feed_dict[placeholders[0]],
                                                                           feed_dict[placeholders[1]])
        tmp = session.run(tensors, feed_dict=feed_dict)
        res = [r + t * current_batch_size for (r, t) in zip(res, tmp[1:])]
    return [r / float(total_size) for r in res]


def dpsn():
    graph = tf.Graph()
    cv_d, cv_j, cv_f = 0, 0, 0
    with graph.as_default():
        xs = tf.placeholder(tf.float32, shape=[None, image_size, image_size], name='xs')
        ys = tf.placeholder(tf.float32, shape=[None, image_size, image_size], name='ys')
        xsc = tf.reshape(xs, [-1, image_size, image_size, 1], name='xsc')
        ysc = tf.reshape(ys, [-1, image_size, image_size, 1], name='ysc')

        lr = tf.placeholder(tf.float32, shape=[], name='lr')
        keep = tf.placeholder(tf.float32, shape=[], name='keep')
        is_train = tf.placeholder(tf.bool, shape=[], name='is_train')

        # Block1 (256, 256) dilated_rate=1
        tensor = conv2d(xsc, 1, 16, k_size=3)
        tensor, ch = block(tensor, block_layers, in_ch=16, growth_ch=growth, d_rate=1, is_train=is_train, keep=keep)
        fm1_256 = conv2d(tensor, ch, fuse_ch, k_size=3, bias=True)
        tensor = bn_act_dconv(tensor, ch, np.int32(ch * theta), k_size=1, d_rate=1, is_train=is_train, keep=keep)
        tensor = tf.nn.avg_pool(tensor, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        print('block 1 out =======', np.int32(ch * theta))

        # Block2 (128, 128) dilated_rate=1
        tensor, ch = block(tensor, block_layers, np.int32(ch * theta), growth, d_rate=1, is_train=is_train, keep=keep)
        fm2_128 = conv2d(tensor, ch, fuse_ch, k_size=3, bias=True)
        tensor = bn_act_dconv(tensor, ch, np.int32(ch * theta), k_size=1, d_rate=1, is_train=is_train, keep=keep)
        tensor = tf.nn.avg_pool(tensor, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        print('block 2 out =======', np.int32(ch * theta))

        # Block3 (64, 64) dilated_rate=2
        tensor, ch = block(tensor, block_layers, np.int32(ch * theta), growth, d_rate=2, is_train=is_train, keep=keep)
        fm3_64 = conv2d(tensor, ch, fuse_ch, k_size=3, bias=True)
        tensor = bn_act_dconv(tensor, ch, np.int32(ch * theta), k_size=1, d_rate=1, is_train=is_train, keep=keep)
        print('block 3 out =======', np.int32(ch * theta))

        # Block4 (64, 64) dilated_rate=4
        tensor, ch = block(tensor, block_layers, np.int32(ch * theta), growth, d_rate=4, is_train=is_train, keep=keep)
        fm4_64 = conv2d(tensor, ch, fuse_ch, k_size=3, bias=True)
        tensor = bn_act_dconv(tensor, ch, np.int32(ch * theta), k_size=1, d_rate=1, is_train=is_train, keep=keep)
        print('block 4 out =======', np.int32(ch * theta))

        # Block5 (64, 64) dilted_rate=8
        tensor, ch = block(tensor, block_layers, np.int32(ch * theta), growth, d_rate=8, is_train=is_train, keep=keep)
        fm5_64 = conv2d(tensor, ch, fuse_ch, k_size=3, bias=True)
        print('block 5 out =======', fuse_ch)

        # PPM
        tensor, ch = ppm64(fm5_64, fuse_ch, out_ch=1)

        # UP5 (64, 64)
        tensor = tf.concat([fm5_64, tensor], axis=3)
        tensor_p5 = batch_norm(tensor, is_train=is_train)
        tensor_p5 = tf.nn.relu(tensor_p5)
        ch = ch + fuse_ch
        tensor_p5 = conv2d(tensor_p5, ch, 1, 3)

        # UP4 (64, 64)
        tensor = tf.concat([fm4_64, tensor], axis=3)
        tensor_p4 = batch_norm(tensor, is_train=is_train)
        tensor_p4 = tf.nn.relu(tensor_p4)
        ch = ch + fuse_ch
        tensor_p4 = conv2d(tensor_p4, ch, 1, 3)

        # UP3 (64, 64)
        tensor = tf.concat([fm3_64, tensor], axis=3)
        tensor_p3 = batch_norm(tensor, is_train=is_train)
        tensor_p3 = tf.nn.relu(tensor_p3)
        ch = ch + fuse_ch
        tensor_p3 = conv2d(tensor_p3, ch, 1, 3)

        # UP2 (128, 128)
        tensor = upsample(tensor, 2, ch)
        tensor = tf.concat([fm2_128, tensor], axis=3)
        tensor_p2 = batch_norm(tensor, is_train=is_train)
        tensor_p2 = tf.nn.relu(tensor_p2)
        ch = ch + fuse_ch
        tensor_p2 = conv2d(tensor_p2, ch, 1, 3, True)

        # UP1 (256, 256)
        tensor = upsample(tensor, 2, ch)
        tensor = tf.concat([fm1_256, tensor], axis=3)
        tensor_p1 = batch_norm(tensor, is_train=is_train)
        tensor_p1 = tf.nn.relu(tensor_p1)
        ch = ch + fuse_ch
        tensor_p1 = conv2d(tensor_p1, ch, 1, 3, True)

        # FUSE
        tensor_p5 = upsample(tensor_p5, 4, 1)
        tensor_p4 = upsample(tensor_p4, 4, 1)
        tensor_p3 = upsample(tensor_p3, 4, 1)
        tensor_p2 = upsample(tensor_p2, 2, 1)
        p5 = tf.nn.sigmoid(tensor_p5)
        p4 = tf.nn.sigmoid(tensor_p4)
        p3 = tf.nn.sigmoid(tensor_p3)
        p2 = tf.nn.sigmoid(tensor_p2)
        p1 = tf.nn.sigmoid(tensor_p1)
        tensor_p0 = tf.concat([tensor_p5, tensor_p4, tensor_p3, tensor_p2, tensor_p1], axis=3)
        tensor_p0 = conv2d(tensor_p0, 5, 1, 3, True)
        yp = tf.nn.sigmoid(tensor_p0, name='yp')

        # fuse loss
        loss_p5 = fuse_loss(ysc, p5)
        loss_p4 = fuse_loss(ysc, p4)
        loss_p3 = fuse_loss(ysc, p3)
        loss_p2 = fuse_loss(ysc, p2)
        loss_p1 = fuse_loss(ysc, p1)
        loss_yp = fuse_loss(ysc, yp)
        final_loss = loss_p1 + loss_p2 + loss_p3 + loss_p4 + loss_p5 + loss_yp

        # evaluation
        dice, jaccard, f_beta = coefficients(ysc, yp)

        # l2
        l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        train_step = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True).minimize(final_loss + weight_decay * l2)

    with tf.Session(graph=graph) as session:
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=999)
        print(' >> Training ...')

        if tf.gfile.Exists(log_dir):
            tf.gfile.DeleteRecursively(log_dir)
        tf.gfile.MakeDirs(log_dir)

        dice_epoch = 0
        early_stop_epoch = 0
        for epoch in range(1, 1 + AllEpoch):
            start = timeit.default_timer()

            l_r = learning_rate * ((1 - (epoch - 1) / 30) ** 0.5)

            train_data, train_labels = data['train_data'], data['train_labels']
            pi = np.random.permutation(len(train_data))
            train_data, train_labels = train_data[pi], train_labels[pi]

            batch_count = int(len(train_data) / batch_size)
            batches_data = np.split(train_data[:int(batch_count * batch_size)], batch_count)
            batches_labels = np.split(train_labels[:int(batch_count * batch_size)], batch_count)
            print('Epoch ', epoch, ' contains ', batch_count, ' batches')

            for batch_idx in range(batch_count):
                x, y = data_norm(batches_data[batch_idx],
                                 batches_labels[batch_idx])
                batch_res = session.run([train_step, final_loss, dice, jaccard, f_beta],
                                        feed_dict={xs: x, ys: y,
                                                   lr: l_r, is_train: True, keep: 0.8})
                # if batch_idx % 100 == 0:
                #     print(epoch, batch_idx, batch_res[1:])
            print(' >> Testing ...')
            test_results = run_in_batch_avg(session, [final_loss, dice, jaccard, f_beta], [xs, ys],
                                            feed_dict={xs: data['test_data'], ys: data['test_labels'],
                                                       is_train: False, keep: 1.})
            print('Epoch ', epoch, test_results)
            elapsed = (timeit.default_timer() - start)
            print("Time used: %f" % elapsed)
            if test_results[0] > dice_epoch:
                dice_epoch = test_results[0]
                saver.save(session, save_path=log_dir + '/' + 'dense.ckpt')
                print('Save: dice increased to %f and save ckpt!\n' % dice_epoch)
                cv_d, cv_j, cv_f = test_results[0], test_results[1], test_results[2]
                early_stop_epoch = 0
            else:
                early_stop_epoch += 1
                print('No Change: dice does not change from %f !\n' % dice_epoch)
                if early_stop_epoch == EarlyStopEpoch:
                    print('Early Stop: dice does not increase in %d epochs and stop training!\n' % EarlyStopEpoch)
                    break
                    
    return cv_d, cv_j, cv_f
