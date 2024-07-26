# -*- coding: utf-8 -*-
"""Video Classification High-RAM GPU Runtime

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1RiXeToGPB6ZldAnCByUzt12ajGfsVxTK

# Making the Most of your Colab Subscription

## Faster GPUs

Users who have purchased one of Colab's paid plans have access to premium GPUs. You can upgrade your notebook's GPU settings in `Runtime > Change runtime type` in the menu to enable Premium accelerator. Subject to availability, selecting a premium GPU may grant you access to a V100 or A100 Nvidia GPU.

The free of charge version of Colab grants access to Nvidia's T4 GPUs subject to quota restrictions and availability.

You can see what GPU you've been assigned at any time by executing the following cell. If the execution result of running the code cell below is "Not connected to a GPU", you can change the runtime by going to `Runtime > Change runtime type` in the menu to enable a GPU accelerator, and then re-execute the code cell.
"""

gpu_info = !nvidia-smi
gpu_info = '\n'.join(gpu_info)
if gpu_info.find('failed') >= 0:
  print('Not connected to a GPU')
else:
  print(gpu_info)

"""In order to use a GPU with your notebook, select the `Runtime > Change runtime type` menu, and then set the hardware accelerator dropdown to GPU.

## More memory

Users who have purchased one of Colab's paid plans have access to high-memory VMs when they are available.



You can see how much memory you have available at any time by running the following code cell. If the execution result of running the code cell below is "Not using a high-RAM runtime", then you can enable a high-RAM runtime via `Runtime > Change runtime type` in the menu. Then select High-RAM in the Runtime shape dropdown. After, re-execute the code cell.
"""

from psutil import virtual_memory
ram_gb = virtual_memory().total / 1e9
print('Your runtime has {:.1f} gigabytes of available RAM\n'.format(ram_gb))

if ram_gb < 20:
  print('Not using a high-RAM runtime')
else:
  print('You are using a high-RAM runtime!')

"""## Longer runtimes

All Colab runtimes are reset after some period of time (which is faster if the runtime isn't executing code). Colab Pro and Pro+ users have access to longer runtimes than those who use Colab free of charge.

## Background execution

Colab Pro+ users have access to background execution, where notebooks will continue executing even after you've closed a browser tab. This is always enabled in Pro+ runtimes as long as you have compute units available.

## Relaxing resource limits in Colab Pro

Your resources are not unlimited in Colab. To make the most of Colab, avoid using resources when you don't need them. For example, only use a GPU when required and close Colab tabs when finished.



If you encounter limitations, you can relax those limitations by purchasing more compute units via Pay As You Go. Anyone can purchase compute units via [Pay As You Go](https://colab.research.google.com/signup); no subscription is required.

## Send us feedback!

If you have any feedback for us, please let us know. The best way to send feedback is by using the Help > 'Send feedback...' menu. If you encounter usage limits in Colab Pro consider subscribing to Pro+.

If you encounter errors or other issues with billing (payments) for Colab Pro, Pro+, or Pay As You Go, please email [colab-billing@google.com](mailto:colab-billing@google.com).

## More Resources

### Working with Notebooks in Colab
- [Overview of Colaboratory](/notebooks/basic_features_overview.ipynb)
- [Guide to Markdown](/notebooks/markdown_guide.ipynb)
- [Importing libraries and installing dependencies](/notebooks/snippets/importing_libraries.ipynb)
- [Saving and loading notebooks in GitHub](https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb)
- [Interactive forms](/notebooks/forms.ipynb)
- [Interactive widgets](/notebooks/widgets.ipynb)

<a name="working-with-data"></a>
### Working with Data
- [Loading data: Drive, Sheets, and Google Cloud Storage](/notebooks/io.ipynb)
- [Charts: visualizing data](/notebooks/charts.ipynb)
- [Getting started with BigQuery](/notebooks/bigquery.ipynb)

### Machine Learning Crash Course
These are a few of the notebooks from Google's online Machine Learning course. See the [full course website](https://developers.google.com/machine-learning/crash-course/) for more.
- [Intro to Pandas DataFrame](https://colab.research.google.com/github/google/eng-edu/blob/main/ml/cc/exercises/pandas_dataframe_ultraquick_tutorial.ipynb)
- [Linear regression with tf.keras using synthetic data](https://colab.research.google.com/github/google/eng-edu/blob/main/ml/cc/exercises/linear_regression_with_synthetic_data.ipynb)


<a name="using-accelerated-hardware"></a>
### Using Accelerated Hardware
- [TensorFlow with GPUs](/notebooks/gpu.ipynb)
- [TensorFlow with TPUs](/notebooks/tpu.ipynb)

<a name="machine-learning-examples"></a>

## Machine Learning Examples

To see end-to-end examples of the interactive machine learning analyses that Colaboratory makes possible, check out these  tutorials using models from [TensorFlow Hub](https://tfhub.dev).

A few featured examples:

- [Retraining an Image Classifier](https://tensorflow.org/hub/tutorials/tf2_image_retraining): Build a Keras model on top of a pre-trained image classifier to distinguish flowers.
- [Text Classification](https://tensorflow.org/hub/tutorials/tf2_text_classification): Classify IMDB movie reviews as either *positive* or *negative*.
- [Style Transfer](https://tensorflow.org/hub/tutorials/tf2_arbitrary_image_stylization): Use deep learning to transfer style between images.
- [Multilingual Universal Sentence Encoder Q&A](https://tensorflow.org/hub/tutorials/retrieval_with_tf_hub_universal_encoder_qa): Use a machine learning model to answer questions from the SQuAD dataset.
- [Video Interpolation](https://tensorflow.org/hub/tutorials/tweening_conv3d): Predict what happened in a video between the first and the last frame.
"""

from mmengine import registry

registry



### Video Classification

### Using drive

# import os, sys
# from google.colab import drive

# drive.mount('/content/drive')
# my_path = "/content/drive/MyDrive/Colab\ Notebooks"

# # sys.path.insert(0, my_path)
# # sys.path = ['/content/drive/MyDrive/Colab\\ Notebooks',
# #             '/env/python',
# #             '/usr/lib/python39.zip',
# #             '/usr/lib/python3.9',
# #             '/usr/lib/python3.9/lib-dynload',
# #             '/usr/local/lib/python3.9/dist-packages',
# #             '/usr/lib/python3/dist-packages',
# #             '/usr/local/lib/python3.9/dist-packages/IPython/extensions',
# #             '/root/.ipython'
# #             ]
# sys.path = [my_path, '/content', '/env/python', '/usr/local/lib/python3.9/dist-packages/IPython/extensions', '/root/.ipython']
# print(sys.path)

# %cd $my_path
# !pip list

# os.symlink('/content/gdrive/My Drive/Colab Notebooks', my_path)

# Check pytorch installation
# %cd $my_path

# import torch, torchvision
# print(torch.__version__, torch.cuda.is_available())

# Prerequisites
# !pip3 install jedi numpy==1.23 pandas==1.5.3
# !pip3 install --target=$my_path jedi numpy==1.23 pandas==1.5.3

# # Step 1:
# !pip3 install -U --target=$my_path openmim
# !mim install --target=$my_path mmengine
# !mim install --target=$my_path mmcv

# # Step 2:
# # !git clone https://github.com/open-mmlab/mmaction2.git
# mmaction2_path = '/content/drive/My Drive/Colab Notebooks/mmaction2'
# %cd $mmaction2_path
# sys.path.append(mmaction2_path)
# !pip3 install --target=$my_path -v -e .

# # Step 3: Verify installation
# # !mim download mmaction2 --config tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb --dest .
# # !python demo/demo.py tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.py \
# #     tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth \
# #     demo/demo.mp4 tools/data/kinetics/label_map_k400.txt

# !ls





# Commented out IPython magic to ensure Python compatibility.
### Use default local
import numpy as np
import os

from google.colab import drive
drive.mount('/content/drive')

!pip3 install torch torchvision
!pip3 install -U openmim
!mim install mmcv
!mim install mmengine
!mim install mmcv
!mim install mmdet

!git clone https://github.com/open-mmlab/mmaction2.git
# %cd mmaction2
!pip3 install -v -e .

# drive.mount('/content/drive', force_remount=True)
!ls '/content/drive/MyDrive'

# !pip list

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/mmaction2

# Check pytorch installation
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())

# Check mmaction2 installation
import mmaction
print(mmaction.__version__)

# Check mmcv installation
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
print(get_compiling_cuda_version())
print(get_compiler_version())

from mmaction.engine.hooks import OutputHook

# import mmengine
# config_path = 'configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py' # num_classes=3
# config = mmengine.Config.fromfile(config_path)

# Commented out IPython magic to ensure Python compatibility.

# from mmaction.apis.inference import *
import sys
my_path = '/content/drive/My\ Drive/'
# sys.path.append(my_path)

# %cd $my_path
from mmaction2_custom.inference import init_recognizer, inference_recognizer

# %cd /content/mmaction2

# # Test demo
# config_path = 'configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.py'
# checkpoint_path = 'https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth' # can be a local path
# img_path = '/content/drive/MyDrive/Ding_human_training/train/EL_G_ball_C_left_no_00163_528x320.avi'   # you can specify your own picture path

# # build the model from a config file and a checkpoint file
# model = init_recognizer(config_path, checkpoint_path, device="cuda:0")  # device can be 'cpu'
# # test a single image
# result = inference_recognizer(model, img_path)
# print(result)

# # Copy custom config
# !cp /content/drive/MyDrive/mmaction2_custom_functions/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.py configs/recognition/tsn/tsn_custom.py
# !ls configs/recognition/tsn

# !cp /content/drive/MyDrive/mmaction2_custom_functions/tsn_flow_custom.py configs/recognition/tsn/tsn_flow_custom.py
# !ls configs/recognition/tsn

# Print model config
# config_path = 'configs/recognition/tsn/tsn_flow_custom.py'
# checkpoint_path = 'https://download.pytorch.org/models/resnet50-11ad3fa6.pth' # can be a local path or url
# model = init_recognizer(config_path, checkpoint_path, device="cuda:0")  # device can be 'cpu'
# model

!pip3 install pytorchvideo

drive.mount('/content/drive', force_remount=True)

# Train with customed config

# !python3 tools/train.py /content/drive/MyDrive/mmaction2_custom/tsn_r50_8xb32-1x1x3-100e_rgb_kinetics400_tuned_transfer_batchsize8_configs.py

# !python3 tools/train.py /content/drive/MyDrive/mmaction2_custom/tsn_r50_8xb32-1x1x8-100e_rgb_kinetics400_tuned_transfer_batchsize8_configs.py

# !python3 tools/train.py /content/drive/MyDrive/mmaction2_custom/tsn_r50_8xb32-1x1x8-50e_rgb_sthv2_tuned_transfer_batchsize8_configs.py

# !python3 tools/train.py /content/drive/MyDrive/mmaction2_custom/tsn_r50_8xb32-1x1x8-100e_flow_kinetics400_tuned_transfer_batchsize8_configs.py

# !python3 tools/train.py /content/drive/MyDrive/mmaction2_custom/tsn_r50_8xb32-1x1x8-50e_flow_sthv2_tuned_transfer_batchsize8_configs.py

# !python3 tools/train.py /content/drive/MyDrive/mmaction2_custom/timesformer_divST_8xb8-8x32x1-15e_kinetics400_tuned_transfer_batchsize4_configs.py

# !python3 tools/train.py /content/drive/MyDrive/mmaction2_custom/timesformer_divST_8xb8-8x32x1-15e_untuned_transfer_batchsize4_configs.py

!python3 tools/train.py /content/drive/MyDrive/mmaction2_custom/uniformerv2-base-p16-res224_clip_u8_kinetics400_tuned_transfer_batchsize4_configs.py

# Model inference and extract intermediate layer outputs
config_path = '/content/drive/MyDrive/mmaction2_custom/timesformer_divST_8xb8-8x32x1-15e_untuned_transfer_batchsize4_configs.py' # num_classes=3
# layer_list format: 
# temporal attention output candidates: 
#     'backbone.transformer_layers.layers.(0-11).attentions.0.dropout_layer',
#     'backbone.transformer_layers.layers.(0-11).attentions.0.temporal_fc',
# spatial attention output candidates: 
#     'backbone.transformer_layers.layers.(0-11).attentions.1.dropout_layer',
#     'backbone.transformer_layers.layers.(0-11).ffns.0.layers.0.0',
#     'backbone.transformer_layers.layers.(0-11).ffns.0.layers.0.2',
# MLP output candidates: 
#     'backbone.transformer_layers.layers.(0-11).ffns.norm',
# final loss: 'cls_head.loss_cls', 'cls_head.fc_cls'
layer_list = [
    'backbone.transformer_layers.layers.11.attentions.0.dropout_layer',
    'backbone.transformer_layers.layers.11.attentions.0.temporal_fc',
    'backbone.transformer_layers.layers.11.attentions.1.dropout_layer',
    'backbone.transformer_layers.layers.11.ffns.0.layers.0.0',
    'backbone.transformer_layers.layers.11.ffns.0.layers.0.2',
    'backbone.transformer_layers.layers.11.ffns.0.norm',
    'cls_head.loss_cls', 'cls_head.fc_cls',
]
video_path_list = ['/content/drive/MyDrive/mmaction2_custom/datasets/Ding_human_training/train', 
                   '/content/drive/MyDrive/mmaction2_custom/datasets/Ding_human_generalization', 
                   '/content/drive/MyDrive/mmaction2_custom/datasets/Ding_monkey_scanning']
video_group_names = ['training', 'generalization', 'monkey']
# Class labels:
labels = ['grasp', 'touch', 'reach']
# Loop through all models and all videos
output_rootpath = '/content/drive/MyDrive/mmaction2_custom/timesformer_divST_8xb8-8x32x1-15e_pretrain_inferences'
os.makedirs(output_rootpath, exist_ok=True)
# Milestones:
# epoch_name_list = ['epoch_60', 'epoch_180']
# top1 acc = [0.50, 0.67]
# for epoch_name in ['epoch_60', 'epoch_180']:
#     print(epoch_name)
checkpoint_path_list = [
    'https://download.openmmlab.com/mmaction/recognition/timesformer/vit_base_patch16_224.pth', 
    '/content/drive/MyDrive/mmaction2_custom/datasets/pretrain_checkpoints/timesformer_divST_8xb8-8x32x1-15e_kinetics400-rgb_20220815-a4d0d01f.pth',
]
model_name_list = [
    'pretrain_vit_base_patch16_224',
    'pretrain_timesformer_divST_8xb8-8x32x1-15e_kinetics400-rgb_20220815-a4d0d01f',
]
for modelI, checkpoint_path in enumerate(checkpoint_path_list):
    # checkpoint_path = '/content/drive/MyDrive/mmaction2_custom/work_dirs/timesformer_divST_8xb8-8x32x1-15e_untuned_transfer_batchsize4_output/{}.pth'.format(epoch_name)
    model_name = model_name_list[modelI]
    model = init_recognizer(config_path, checkpoint_path, device="cuda:0")
    for groupI, video_folder in enumerate(video_path_list):
        group_name = video_group_names[groupI]
        output_path = os.path.join(output_rootpath, group_name)
        os.makedirs(output_path, exist_ok=True)
        for video in os.listdir(video_folder):
            video_path = os.path.join(video_folder, video)
            print(video_path)
            # video_dict = dict(filename=video_path, label=-1, start_index=0, modality='RGB')
            video_name = video.split('.avi')[0]
            results, layer_features = inference_recognizer(model, video_path, layer_list)
            # results.pred_scores to list
            pred_scores = results.pred_scores.item.tolist()
            pred_labels = [labels[i] for i in results.pred_labels.item.tolist()]
            print(pred_scores, pred_labels)
            # layer_features is already a dict
            cur_model_video_output = {'checkpoint': model_name, 'video': video_name, 'layer_list': layer_list, 'pred_labels': pred_labels, 'pred_scores': pred_scores, 'layer_outputs': layer_features}
            np.savez(os.path.join(output_path, 'timesformer_{}_{}.npz'.format(model_name, video_name)), Results=cur_model_video_output)
            # model_output_dict[epoch_name][video] = {'pred_labels': pred_labels, 'pred_scores': pred_scores, 'layer_outputs': layer_features}
    # print(model_output_dict[epoch_name].keys())

config_path = '/content/drive/MyDrive/mmaction2_custom/uniformerv2-base-p16-res224_clip_u8_kinetics400_tuned_transfer_batchsize4_configs.py' # num_classes=3
checkpoint_path = '/content/drive/MyDrive/mmaction2_custom/work_dirs/uniformerv2-base-p16-res224_clip_u8_kinetics400_tuned_transfer_batchsize4_output/epoch_100.pth'
video_path_list = ['/content/drive/MyDrive/mmaction2_custom/datasets/Ding_human_training/train', 
                   '/content/drive/MyDrive/mmaction2_custom/datasets/Ding_human_generalization', 
                   '/content/drive/MyDrive/mmaction2_custom/datasets/Ding_monkey_scanning']
labels = ['grasp', 'touch', 'reach']
model = init_recognizer(config_path, checkpoint_path, device="cuda:0")
for video_folder in video_path_list:
        for video in os.listdir(video_folder):
            video_path = os.path.join(video_folder, video)
            print(video)
            video_name = video.split('.avi')[0]
            results = inference_recognizer(model, video_path)
            # results.pred_scores to list
            pred_scores = results.pred_scores.item.tolist()
            pred_labels = [labels[i] for i in results.pred_labels.item.tolist()]
            print('------', pred_scores, pred_labels)

# # Model inference and extract intermediate layer outputs
# config_path = '/content/drive/MyDrive/mmaction2_custom/tsn_r50_8xb32-1x1x8-50e_flow_sthv2_tuned_transfer_batchsize8_configs.py' # num_classes=3
# # layer_list format: 
# layer_list = ['cls_head.consensus', 'cls_head.avg_pool', 'cls_head.dropout', 'cls_head.fc_cls']
# # layer_list = ['backbone.conv1.conv']
# video_path_list = ['/content/drive/MyDrive/mmaction2_custom/datasets/Ding_human_training_flow/train', 
#                    '/content/drive/MyDrive/mmaction2_custom/datasets/Ding_human_generalization_flow', 
#                    '/content/drive/MyDrive/mmaction2_custom/datasets/Ding_monkey_scanning_flow']
# video_group_names = ['training', 'generalization', 'monkey']
# # Class labels:
# labels = ['grasp', 'touch', 'reach']
# # Loop through all models and all videos
# output_rootpath = '/content/drive/MyDrive/mmaction2_custom/tsn_r50_8xb32-1x1x8-50e_flow_sthv2_tuned_transfer_batchsize8_inferences'
# os.makedirs(output_rootpath, exist_ok=True)
# # Milestones:
# # epoch_name_list = ['epoch_10', 'epoch_90', 'epoch_150', 'epoch_300']
# # loss_cls_list = [1.0, 0.9, 0.83, 0.74]
# for epoch_name in ['epoch_10', 'epoch_90', 'epoch_150', 'epoch_300']:
#     print(epoch_name)
#     checkpoint_path = '/content/drive/MyDrive/mmaction2_custom/work_dirs/tsn_r50_8xb32-1x1x8-50e_flow_sthv2_tuned_transfer_batchsize8_output/{}.pth'.format(epoch_name)
#     model = init_recognizer(config_path, checkpoint_path, device="cuda:0")
#     for groupI, video_folder in enumerate(video_path_list):
#         group_name = video_group_names[groupI]
#         output_path = os.path.join(output_rootpath, group_name)
#         os.makedirs(output_path, exist_ok=True)
#         for video in os.listdir(video_folder):
#             video_path = os.path.join(video_folder, video)
#             print(video_path)
#             # video_dict = dict(filename=video_path, label=-1, start_index=0, modality='RGB')
#             video_name = video.split('.avi')[0]
#             results, layer_features = inference_recognizer(model, video_path, layer_list)
#             # results.pred_scores to list
#             pred_scores = results.pred_scores.item.tolist()
#             pred_labels = [labels[i] for i in results.pred_labels.item.tolist()]
#             print(pred_scores, pred_labels)
#             # layer_features is already a dict
#             cur_model_video_output = {'checkpoint': epoch_name, 'video': video_name, 'layer_list': layer_list, 'pred_labels': pred_labels, 'pred_scores': pred_scores, 'layer_outputs': layer_features}
#             np.savez(os.path.join(output_path, 'tsn_{}_{}.npz'.format(epoch_name, video_name)), Results=cur_model_video_output)
#             # model_output_dict[epoch_name][video] = {'pred_labels': pred_labels, 'pred_scores': pred_scores, 'layer_outputs': layer_features}
#     # print(model_output_dict[epoch_name].keys())

model

# !pwd
# !cp -r /content/mmaction2/work_dirs/timesformer_divST_8xb8-8x32x1-15e_kinetics400-rgb /content/drive/MyDrive/mmaction2_custom_functions/timesformer_custom_output
# !cp -r /content/mmaction2/configs/_base_ /content/drive/MyDrive/mmaction2_custom/configs/_base_
# !cp -r /content/mmaction2/work_dirs/tsn_rgb_kinetics400_tuned_transfer_output /content/drive/MyDrive/mmaction2_custom/tsn_rgb_kinetics400_tuned_transfer_output

# # Make inference
# # config_path = 'configs/recognition/tsn/tsn_custom.py' # num_classes=3
# config_path = '/content/drive/MyDrive/mmaction2_sthv2_tuned/tsn_r50_1x1x8_rgb_kinetics400_tuned_transfer.py' # num_classes=3
# # test_pipeline = Compose(config.test_pipeline)
# pretrain_checkpoint_path = '/content/drive/MyDrive/mmaction2_sthv2_tuned/tsn_imagenet-pretrained-r50_8xb32-1x1x8-50e_sthv2-rgb_20230313-06ad7d03.pth'
# best_epoch_name = 'best_acc_top1_epoch_85'
# # best_checkpoint_path = 'work_dirs/tsn_custom/{}.pth'.format(best_epoch_name) # can be a local path
# best_checkpoint_path = '/content/drive/MyDrive/mmaction2_sthv2_tuned/tsn_rgb_sthv2_custom/{}.pth'.format(best_epoch_name)
# # epoch to save: range(10,310,10)
# checkpoint_path_list = [pretrain_checkpoint_path]
# epoch_name_list = ['pretrain']
# # model_output_dict = {}
# # model_output_dict['pretrain'] = {}
# for i in range(10,310,10):
#     # checkpoint_path_list.append('work_dirs/tsn_custom/epoch_{}.pth'.format(i))
#     checkpoint_path_list.append('/content/drive/MyDrive/mmaction2_sthv2_tuned/tsn_rgb_sthv2_custom/epoch_{}.pth'.format(i))
#     epoch_name = 'epoch_{}'.format(i)
#     epoch_name_list.append(epoch_name)
#     # model_output_dict[epoch_name] = {}
# checkpoint_path_list.append(best_checkpoint_path)
# epoch_name_list.append(best_epoch_name)
# print(checkpoint_path_list)
# print(epoch_name_list)
# model_output_dict[best_epoch_name] = {}
# # build the model from a config file and a checkpoint file
# model = init_recognizer(config_path, pretrain_checkpoint_path, device="cuda:0")  # device can be 'cpu'
# model

# # Model inference and extract intermediate layer outputs
# # layer_list format: 
# layer_list = ['backbone.conv1.conv', 'backbone.conv1.bn', 'backbone.conv1.activate', 'backbone.maxpool', 
#               'backbone.layer1.2.relu', 'backbone.layer2.3.relu', 
#               'backbone.layer3.5.relu', 'backbone.layer4.2.relu', 
#               'cls_head.loss_cls', 'cls_head.consensus', 'cls_head.avg_pool', 'cls_head.dropout', 'cls_head.fc_cls']
# # layer_list = ['backbone.conv1.conv']
# # input path:
# training_video_path = '/content/drive/MyDrive/Ding_human_training/train'
# # labels:
# labels = ['grasp', 'touch', 'reach']
# # test one video
# video = 'EL_G_ball_C_left_no_00163_528x320.avi'
# video_path = os.path.join(training_video_path, video)
# results, layer_features = inference_recognizer(model, video_path, layer_list)
# # transform results
# pred_scores = results.pred_scores.item.tolist()
# pred_labels = [labels[i] for i in results.pred_labels.item.tolist()]
# print(pred_scores, pred_labels)
# layer_features.keys()

