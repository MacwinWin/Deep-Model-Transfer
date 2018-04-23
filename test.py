import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
import inception_preprocessing
import vgg_preprocessing
#from nets.nasnet.nasnet import build_nasnet_large, nasnet_large_arg_scope
from nets.inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
#from nets.inception_v4 import inception_v4, inception_v4_arg_scope
from nets.inception_v3 import inception_v3, inception_v3_arg_scope
from nets.resnet_v1 import resnet_v1_50, resnet_arg_scope
from nets.vgg import vgg_16, vgg_arg_scope
from nets.vgg import vgg_19, vgg_arg_scope
import time
import os
from datasets import dataset_factory
import numpy as np
import math

slim = tf.contrib.slim

#State your log directory where you can retrieve your model

log_dir = '/../102_inception_resnet_v2_150'

#Create a new evaluation log directory to visualize the validation process

log_eval = '/../validation'

#State the dataset directory where the validation set is found

dataset_dir = '/../Flower_102_test'


#State the batch_size to evaluate each time, which can be a lot more than the training batch
batch_size = 1

#State the number of epochs to evaluate
num_epochs = 1

#State the image size you're resizing your images to. We will use the default inception size of 299.
image_size = 299

#State the number of classes to predict:
num_classes = 102

#Get the latest checkpoint file
checkpoint_file = tf.train.latest_checkpoint(log_dir)

#State the labels file and read it

labels_file = '/../Flower_102_test/labels.txt'

labels = open(labels_file, 'r')

#Create a dictionary to refer each label to their string name
labels_to_name = {}
for line in labels:
    label, string_name = line.split(':')
    string_name = string_name[:-1] #Remove newline
    labels_to_name[int(label)] = string_name

#Create the file pattern of your TFRecord files so that it could be recognized later on
file_pattern = 'origin_%s_*.tfrecord'

#Create a dictionary that will help people understand your dataset better. This is required by the Dataset class later.
items_to_descriptions = {
    'image': 'A 3-channel RGB coloured flower image that is either tulips, sunflowers, roses, dandelion, or daisy.',
    'label': 'A label that is as such -- 0:daisy, 1:dandelion, 2:roses, 3:sunflowers, 4:tulips'
}

#============== DATASET LOADING ======================
#We now create a function that creates a Dataset class which will give us many TFRecord files to feed in the examples into a queue in parallel.
def get_split(split_name, dataset_dir, file_pattern=file_pattern, file_pattern_for_counting='flowers'):
    '''
    Obtains the split - training or validation - to create a Dataset class for feeding the examples into a queue later on. This function will
    set up the decoder and dataset information all into one Dataset class so that you can avoid the brute work later on.
    Your file_pattern is very important in locating the files later. 

    INPUTS:
    - split_name(str): 'train' or 'validation'. Used to get the correct data split of tfrecord files
    - dataset_dir(str): the dataset directory where the tfrecord files are located
    - file_pattern(str): the file name structure of the tfrecord files in order to get the correct data
    - file_pattern_for_counting(str): the string name to identify your tfrecord files for counting

    OUTPUTS:
    - dataset (Dataset): A Dataset class object where we can read its various components for easier batch creation later.
    '''

    #First check whether the split_name is train or validation
    if split_name not in ['train', 'validation']:
        raise ValueError('The split_name %s is not recognized. Please input either train or validation as the split_name' % (split_name))

    #Create the full path for a general file_pattern to locate the tfrecord_files
    file_pattern_path = os.path.join(dataset_dir, file_pattern % (split_name))

    #Count the total number of examples in all of these shard
    num_samples = 0
    file_pattern_for_counting = file_pattern_for_counting + '_' + split_name
    tfrecords_to_count = [os.path.join(dataset_dir, file) for file in os.listdir(dataset_dir) if file.startswith(file_pattern_for_counting)]
    for tfrecord_file in tfrecords_to_count:
        for record in tf.python_io.tf_record_iterator(tfrecord_file):
            num_samples += 1

    #Create a reader, which must be a TFRecord reader in this case
    reader = tf.TFRecordReader

    #Create the keys_to_features dictionary for the decoder
    keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
      'image/class/label': tf.FixedLenFeature(
          [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    }

    #Create the items_to_handlers dictionary for the decoder.
    items_to_handlers = {
    'image': slim.tfexample_decoder.Image(),
    'label': slim.tfexample_decoder.Tensor('image/class/label'),
    }

    #Start to create the decoder
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

    #Create the labels_to_name file
    labels_to_name_dict = labels_to_name

    #Actually create the dataset
    dataset = dataset_factory.get_dataset(
        'origin', 'validation', dataset_dir)

    return dataset


def load_batch(dataset, batch_size, height=image_size, width=image_size, is_training=False):
    '''
    Loads a batch for training.

    INPUTS:
    - dataset(Dataset): a Dataset class object that is created from the get_split function
    - batch_size(int): determines how big of a batch to train
    - height(int): the height of the image to resize to during preprocessing
    - width(int): the width of the image to resize to during preprocessing
    - is_training(bool): to determine whether to perform a training or evaluation preprocessing

    OUTPUTS:
    - images(Tensor): a Tensor of the shape (batch_size, height, width, channels) that contain one batch of images
    - labels(Tensor): the batch's labels with the shape (batch_size,) (requires one_hot_encoding).

    '''
    #First create the data_provider object
    data_provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        common_queue_capacity = 24 + 3 * batch_size,
        common_queue_min = 24)

    #Obtain the raw image using the get method
    raw_image, label = data_provider.get(['image', 'label'])

    #Perform the correct preprocessing for this image depending if it is training or evaluating
    image = inception_preprocessing.preprocess_image(raw_image, height, width, is_training)
    #image = vgg_preprocessing.preprocess_image(raw_image, height, width)

    #As for the raw images, we just do a simple reshape to batch it up
    raw_image = tf.expand_dims(raw_image, 0)
    raw_image = tf.image.resize_nearest_neighbor(raw_image, [height, width])
    raw_image = tf.squeeze(raw_image)

    #Batch up the image by enqueing the tensors internally in a FIFO queue and dequeueing many elements with tf.train.batch.
    images, raw_images, labels = tf.train.batch(
        [image, raw_image, label],
        batch_size = batch_size,
        num_threads = 4,
        capacity = 4 * batch_size,
        allow_smaller_final_batch = True)

    return images, raw_images, labels

def run():
    #Create log_dir for evaluation information
    if not os.path.exists(log_eval):
        os.makedirs(log_eval)

    #Just construct the graph from scratch again
    with tf.Graph().as_default() as graph:
        tf.logging.set_verbosity(tf.logging.INFO)
        #Get the dataset first and load one batch of validation images and labels tensors. Set is_training as False so as to use the evaluation preprocessing
        dataset = get_split('validation', dataset_dir)
        images, raw_images, labels = load_batch(dataset, batch_size = batch_size, is_training = False)

        #Create some information about the training steps
        #num_batches_per_epoch = int(dataset.num_samples / batch_size)
        # This ensures that we make a single pass over all of the data.
        num_batches_per_epoch = math.ceil(dataset.num_samples / float(batch_size))
        num_steps_per_epoch = num_batches_per_epoch

        #Now create the inference model but set is_training=False
        #with slim.arg_scope(nasnet_large_arg_scope()):
        #    logits, end_points = build_nasnet_large(images, num_classes = dataset.num_classes, is_training = False)        
        #with slim.arg_scope(resnet_arg_scope()):
        #    logits, end_points = resnet_v1_50(images, num_classes = dataset.num_classes, is_training = False)
        #with slim.arg_scope(inception_v3_arg_scope()):
        #    logits, end_points = inception_v3(images, num_classes = dataset.num_classes, is_training = False)
        with slim.arg_scope(inception_resnet_v2_arg_scope()):
            logits, end_points = inception_resnet_v2(images, num_classes = dataset.num_classes, is_training = False)
        #with slim.arg_scope(inception_v4_arg_scope()):
        #    logits, end_points = inception_v4(images, num_classes = dataset.num_classes, is_training = False)
        #with slim.arg_scope(vgg_arg_scope()):
        #    logits, end_points = vgg_16(images, num_classes = dataset.num_classes, is_training = False)
        #with slim.arg_scope(vgg_arg_scope()):
        #    logits, end_points = vgg_19(images, num_classes = dataset.num_classes, is_training = False)

        # #get all the variables to restore from the checkpoint file and create the saver function to restore
        variables_to_restore = slim.get_variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        def restore_fn(sess):
            return saver.restore(sess, checkpoint_file)

        #Perform one-hot-encoding of the labels (Try one-hot-encoding within the load_batch function!)
        one_hot_labels = slim.one_hot_encoding(labels, dataset.num_classes)
        
        #Performs the equivalent to tf.nn.sparse_softmax_cross_entropy_with_logits but enhanced with checks
        loss = tf.losses.softmax_cross_entropy(onehot_labels = one_hot_labels, logits = logits)
        total_loss = tf.losses.get_total_loss()    #obtain the regularization losses as well

        #Just define the metrics to track without the loss or whatsoever
        predictions = tf.argmax(logits, 1)
        #predictions = tf.argmax(end_points['Predictions'], 1)
        labels = tf.squeeze(labels)
        accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(predictions, labels)
        metrics_op = tf.group(accuracy_update)

        #Create the global step and an increment op for monitoring
        global_step = get_or_create_global_step()
        global_step_op = tf.assign(global_step, global_step + 1) #no apply_gradient method so manually increasing the global_step
        
        labels_all = []
        predictions_all = []
        #Create a evaluation step function
        def eval_step(sess, metrics_op, global_step):
            '''
            Simply takes in a session, runs the metrics op and some logging information.
            '''
            start_time = time.time()
            _, global_step_count, labels_, predictions_, accuracy_value, = sess.run([metrics_op, global_step_op, labels, predictions, accuracy])
            time_elapsed = time.time() - start_time
            #Log some information
            logging.info('Global Step %s: Streaming Accuracy: %.4f (%.2f sec/step)', global_step_count, accuracy_value, time_elapsed)
            #labels_all = np.append(labels_all, labels_)
            #predictions_all = np.append(predictions_all, labels_)
            #labels_all = labels_all.astype(int)
            #predictions_all = predictions_all.astype(int)
            #print(labels_)
            #print(predictions_)
            return accuracy_value, labels_, predictions_


        #Define some scalar quantities to monitor
        tf.summary.scalar('Validation_Accuracy', accuracy)
        tf.summary.scalar('Loss_validation', total_loss)
        my_summary_op = tf.summary.merge_all()

        #Get your supervisor
        sv = tf.train.Supervisor(logdir = log_eval, summary_op = None, saver = None, init_fn = restore_fn)
 
        all_predictions = np.zeros(
            (dataset.num_samples, num_classes), dtype=np.float32)
        all_labels = np.zeros(
            (dataset.num_samples, num_classes), dtype=np.float32)

        #Now we are ready to run in one session
        #config = tf.ConfigProto(device_count={'GPU':0}) # mask GPUs visible to the session so it falls back on CPU
        with sv.managed_session() as sess:
            for step in range(num_steps_per_epoch * num_epochs):
                sess.run(sv.global_step)
                #print vital information every start of the epoch as always
                if step % num_batches_per_epoch == 0:
                    logging.info('Epoch: %s/%s', step / num_batches_per_epoch + 1, num_epochs)
                    logging.info('Current Streaming Accuracy: %.4f', sess.run(accuracy))
                    logging.info('Current Loss: %.4f', sess.run(total_loss))
                #Compute summaries every 10 steps and continue evaluating
                #if step % 10 == 0:
                #    eval_step(sess, metrics_op = metrics_op, global_step = sv.global_step)
                #    summaries = sess.run(my_summary_op)
                #    sv.summary_computed(sess, summaries)
                    

                #Otherwise just run as per normal
                else:
                    _, labels_, predictions_ = eval_step(sess, metrics_op = metrics_op, global_step = sv.global_step)
                    labels_all = np.append(labels_all, labels_)
                    predictions_all = np.append(predictions_all, predictions_)
                    labels_all = labels_all.astype(int)
                    predictions_all = predictions_all.astype(int)
                    print(labels_)
                    print(predictions_)
            #At the end of all the evaluation, show the final accuracy
            logging.info('Final Streaming Accuracy: %.4f', sess.run(accuracy))
            logging.info('Final Loss: %.4f', sess.run(total_loss))
            np.savetxt("/../labels.txt", labels_all)
            np.savetxt("/../predictions.txt", predictions_all)
            #print(labels_all)
            #print(predictions_all)
            #Now we want to visualize the last batch's images just to see what our model has predicted
            #raw_images, labels, predictions = sess.run([raw_images, labels, predictions])

            logging.info('Model evaluation has completed! Visit TensorBoard for more information regarding your evaluation.')

if __name__ == '__main__':
    run()
