# Deep-Model-Transfer
[![Documentation](https://img.shields.io/badge/Python-3.5.0%2B-green.svg)]()
[![Documentation](https://img.shields.io/badge/TensorFlow-1.5.0%2B-orange.svg)]()

> A method for Fine-Grained image classification implemented by TensorFlow. The best accuracy we have got are 73.33%(Bird-200), 91.03%(Car-196), 72.23%(Dog-120), 96.27%(Flower-102), 86.07%(Pet-37).

------------------

**Note**: For Fine-Grained image classification problem, our solution is combining deep model and transfer learning. Firstly, the deep model, e.g., [vgg-16](https://arxiv.org/abs/1409.1556), [vgg-19](https://arxiv.org/abs/1409.1556), [inception-v1](https://arxiv.org/abs/1409.4842), [inception-v2](https://arxiv.org/abs/1502.03167), [inception-v3](https://arxiv.org/abs/1512.00567), [inception-v4](https://arxiv.org/abs/1602.07261), [inception-resnet-v2](https://arxiv.org/abs/1602.07261), [resnet50](https://arxiv.org/abs/1512.03385), is pretrained in [ImageNet](http://image-net.org/challenges/LSVRC/2014/browse-synsets) dataset to gain the feature extraction abbility. Secondly, transfer the pretrained model to Fine-Grained image dataset, e.g., 🕊️[Bird-200](http://www.vision.caltech.edu/visipedia/CUB-200.html), 🚗[Car-196](https://ai.stanford.edu/~jkrause/cars/car_dataset.html), 🐶[Dog-120](http://vision.stanford.edu/aditya86/ImageNetDogs/), 🌸[Flower-102](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/), 🐶🐱[Pet-37](http://www.robots.ox.ac.uk/~vgg/data/pets/).

## Installation
1. Install the requirments:
  - Ubuntu 16.04+
  - TensorFlow 1.5.0+
  - Python 3.5+
  - NumPy
  - Nvidia GPU(optional)

2. Clone the repository
  ```Shell
  $ git clone https://github.com/MacwinWin/Deep-Model-Transfer.git
  ```

## Pretrain
Slim provide a log of pretrained [models](https://github.com/tensorflow/models/tree/master/research/slim). What we need is just downloading the .ckpt files and then using them. Make a new folder, download and extract the .ckpt file to the folder.
```
pretrained
├── inception_v1.ckpt
├── inception_v2.ckpt
├── inception_v3.ckpt
├── inception_v4.ckpt
├── inception_resnet_v2.ckpt
|   └── ...
```

## Transfer
1. set environment variables
  - Edit the [set_train_env.sh](https://github.com/MacwinWin/Deep-Model-Transfer/blob/master/set_train_env.sh) and [set_eval_env.sh](https://github.com/MacwinWin/Deep-Model-Transfer/blob/master/set_eval_env.sh) files to specify the "DATASET_NAME", "DATASET_DIR", "CHECKPOINT_PATH", "TRAIN_DIR", "MODEL_NAME". 

  - "DATASET_NAME" and "DATASET_DIR" define the dataset name and location. For example, the dataset structure is shown below. "DATASET_NAME" is "origin", "DATASET_DIR" is "/../Flower_102"
```
Flower_102
├── _origin
|   ├── _class1
|            ├── image1.jpg
|            ├── image2.jpg
|            └── ...
|   └── _class2
|   └── ...
```
  - "CHECKPOINT_PATH" is the path to pretrained model. For example, '/../pretrained/inception_v1.ckpt'.
  - "TRAIN_DIR" stores files generated during training. 
  - "MODEL_NAME" is the name of pretrained model, such as resnet_v1_50, vgg_19, vgg_16, inception_resnet_v2, inception_v1, inception_v2, inception_v3, inception_v4.
  - Source the [set_train_env.sh](https://github.com/MacwinWin/Deep-Model-Transfer/blob/master/set_train_env.sh) and [set_eval_env.sh](https://github.com/MacwinWin/Deep-Model-Transfer/blob/master/set_eval_env.sh)
  ```Shell
  $ source set_train_env.sh
  ```
2. prepare data

We use the tfrecord format to feed the model, so we should convert the .jpg file to tfrecord file.
  - After downloading the dataset, arrange the iamges like the structure below. 
```
Flower_102
├── _origin
|   ├── _class1
|            ├── image1.jpg
|            ├── image2.jpg
|            └── ...
|   └── _class2
|   └── ...

Flower_102_eval
├── _origin
|   ├── _class1
|            ├── image1.jpg
|            ├── image2.jpg
|            └── ...
|   └── _class2
|   └── ...

Flower_102_test
├── _origin
|   ├── _class1
|            ├── image1.jpg
|            ├── image2.jpg
|            └── ...
|   └── _class2
|   └── ...
```
If the dataset doesn't have validation set, we can extract some images from test set. The percentage or quantity is defined at [./datasets/convert_tfrecord.py](https://github.com/MacwinWin/Deep-Model-Transfer/blob/master/datasets/convert_tfrecord.py) line [170](https://github.com/MacwinWin/Deep-Model-Transfer/blob/f399fd6011bc35e42e8b6559ea3846ed0d6a57c0/datasets/convert_tfrecord.py#L170) [171](https://github.com/MacwinWin/Deep-Model-Transfer/blob/f399fd6011bc35e42e8b6559ea3846ed0d6a57c0/datasets/convert_tfrecord.py#L171).

  - Run [./convert_data.py](https://github.com/MacwinWin/Deep-Model-Transfer/blob/master/convert_data.py) 
  ```Shell
  $ python convert_data.py \
       --dataset_name=$DATASET_NAME \
       --dataset_dir=$DATASET_DIR
  ```
3. train and evaluate  

  - Edit [./train.py](https://github.com/MacwinWin/Deep-Model-Transfer/blob/master/train.py) to specify "image_size", "num_classes".
  - Edit [./train.py](https://github.com/MacwinWin/Deep-Model-Transfer/blob/master/train.py) line [162](https://github.com/MacwinWin/Deep-Model-Transfer/blob/f399fd6011bc35e42e8b6559ea3846ed0d6a57c0/train.py#L162) to selecet image preprocessing method.
  - Edit [./train.py](https://github.com/MacwinWin/Deep-Model-Transfer/blob/master/train.py) line [203](https://github.com/MacwinWin/Deep-Model-Transfer/blob/f399fd6011bc35e42e8b6559ea3846ed0d6a57c0/train.py#L203) to create the model inference.
  - Edit [./train.py](https://github.com/MacwinWin/Deep-Model-Transfer/blob/master/train.py) line [219](https://github.com/MacwinWin/Deep-Model-Transfer/blob/f399fd6011bc35e42e8b6559ea3846ed0d6a57c0/train.py#L219) to define the scopes that you want to exclude for restoration
  - Edit [./set_train_env.sh]
  - Run script[./run_train.sh](https://github.com/MacwinWin/Deep-Model-Transfer/blob/master/run_train.sh) to start training.
  - Create a new terminal window and set the [./set_eval_env.sh](https://github.com/MacwinWin/Deep-Model-Transfer/blob/master/set_eval_env.sh) to satisfy validation set.
  - Create a new terminal, edit [./set_eval_env.sh](https://github.com/MacwinWin/Deep-Model-Transfer/blob/master/set_eval_env.sh), and run script[./run_eval.sh](https://github.com/MacwinWin/Deep-Model-Transfer/blob/master/run_eval.sh) as the following command.
  
  **Note**: If you have 2 GPU, you can evaluate with GPU by changing [[./eval.py](https://github.com/MacwinWin/Deep-Model-Transfer/blob/master/eval.py)] line [175](https://github.com/MacwinWin/Deep-Model-Transfer/blob/f399fd6011bc35e42e8b6559ea3846ed0d6a57c0/eval.py#L175)-line[196](https://github.com/MacwinWin/Deep-Model-Transfer/blob/f399fd6011bc35e42e8b6559ea3846ed0d6a57c0/eval.py#L196) as shown below
   ```python
    #config = tf.ConfigProto(device_count={'GPU':0})
    if not FLAGS.eval_interval_secs:
      slim.evaluation.evaluate_once(
          master=FLAGS.master,
          checkpoint_path=checkpoint_path,
          logdir=FLAGS.eval_dir,
          num_evals=num_batches,
          eval_op=list(names_to_updates.values()),
          variables_to_restore=variables_to_restore
          #session_config=config)
          )
    else:
      slim.evaluation.evaluation_loop(
          master=FLAGS.master,
          checkpoint_dir=checkpoint_path,
          logdir=FLAGS.eval_dir,
          num_evals=num_batches,
          eval_op=list(names_to_updates.values()),
          eval_interval_secs=60,
          variables_to_restore=variables_to_restore
          #session_config=config)
          )
  ```
4. test

The [./test.py](https://github.com/MacwinWin/Deep-Model-Transfer/blob/master/test.py) looks like [./train.py](https://github.com/MacwinWin/Deep-Model-Transfer/blob/master/train.py), so edit [./set_test_env.sh] as shown before to satisfy your environment. Then run [./run_test.py](https://github.com/MacwinWin/Deep-Model-Transfer/blob/master/run_test.sh).
  **Note**: After test, you can get 2 .txt file. One is ground truth lable, another is predicted lable. Edit line[303](https://github.com/MacwinWin/Deep-Model-Transfer/blob/1045afed03b0dbbe317b91b416fb9b937da40649/test.py#L303) and line[304](https://github.com/MacwinWin/Deep-Model-Transfer/blob/1045afed03b0dbbe317b91b416fb9b937da40649/test.py#L304) to change the store path.
  
## Visualization

Through tensorboard, you can visualization the training and testing process.
   ```Shell
  $ tensorboard --logdir $TRAIN_DIR
  ```
:point_down:Screenshot:
<p align="center">
  <img src="https://s31.postimg.cc/wx7ama097/Screenshot_from_2018-04-17_18-20-12.png" width="400px" alt=""><img src="https://s31.postimg.cc/npf25mb7f/Screenshot_from_2018-04-17_18-21-23.png" width="400px" alt="">
</p>
<p align="center">
  <img src="https://s31.postimg.cc/xmq2yob3f/Screenshot_from_2018-04-17_18-20-36.png" width="400px" alt=""><img src="https://s31.postimg.cc/k5t4ftg7f/Screenshot_from_2018-04-17_18-20-56.png" width="400px" alt="">
</p>

## Deploy

Deploy methods support html and api. Through html method, you can upload image file and get result in web browser. If you want to get result in json format, you can use api method.
Because I'm not good at front-end and back-end skill, so the code not looks professional.

The deployment repository: https://github.com/MacwinWin/Deep-Model-Transfer-Deployment

<p align="center"> 
<img src="https://github.com/MacwinWin/Deep-Model-Transfer-Deployment/blob/master/Peek%202020-04-05%2016-02.gif" width = 100% height = 100%>
</p> 
<p align="center"> 
<img src="https://github.com/MacwinWin/Deep-Model-Transfer-Deployment/blob/master/Peek%202020-04-05%2016-08.gif" width = 100% height = 100%>
</p> 
