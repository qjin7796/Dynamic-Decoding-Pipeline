# Script to visualize model confusion matrix and layer feature distance maps across training stages on generalization videos (9 videos)
# 1. Get the confusion matrix of each model for each video class (one video class = 3 videos of grasp/touch/reach)
# 2. Correlate the confusion matrix with the behavior confusion matrix
# 3. Get the internal layer output from each model for each video
# 4. Compute the distance map of the internal layer output across action types for each video class
# 5. Correlate model internal layer output dist maps with behavior Youden index dist maps
# @Qiuhan Jin, 15/05/2023


import numpy as np
import os
import json
from json import JSONEncoder
import matplotlib.pyplot as plt


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


model_configs = {
    'i3d_rgb_kinetics400-tuned': {
        'model_name': 'i3d',
        'model_type': 'rgb',
        'path2checkpoints': '',
        'path2inferences': '/data/local/u0151613/Qiuhan_Projects/DL_VideoClassification_Qiuhan/mmaction2_custom/model_tuning/i3d/i3d_r50_8xb8-32x2x1-100e_rgb_kinetics400-tuned_batchsize-8_inferences.npz',
    },
    'i3d_flow_kinetics400-tuned': {
        'model_name': 'i3d',
        'model_type': 'flow',
        'path2checkpoints': '',
        'path2inferences': '/data/local/u0151613/Qiuhan_Projects/DL_VideoClassification_Qiuhan/mmaction2_custom/model_tuning/i3d/i3d_r50_8xb8-32x2x1-100e_flow_kinetics400-tuned_batchsize-8_inferences.npz',
    },
    'tsn_rgb_kinetics400-tuned': {
        'model_name': 'tsn',
        'model_type': 'rgb',
        'path2checkpoints': '',
        'path2inferences': '/data/local/u0151613/Qiuhan_Projects/DL_VideoClassification_Qiuhan/mmaction2_custom/model_tuning/tsn/tsn_r50_8xb32-1x1x8-100e_rgb_kinetics400-tuned_batchsize-8_inferences.npz',
    },
    'tsn_flow_kinetics400-tuned': {
        'model_name': 'tsn',
        'model_type': 'flow',
        'path2checkpoints': '',
        'path2inferences': '/data/local/u0151613/Qiuhan_Projects/DL_VideoClassification_Qiuhan/mmaction2_custom/model_tuning/tsn/tsn_r50_8xb32-1x1x8-100e_flow_kinetics400-tuned_batchsize-8_inferences.npz',
    },
    'tsn_rgb_ssv2-tuned': {
        'model_name': 'tsn',
        'model_type': 'rgb',
        'path2checkpoints': '',
        'path2inferences': '/data/local/u0151613/Qiuhan_Projects/DL_VideoClassification_Qiuhan/mmaction2_custom/model_tuning/tsn/tsn_r50_8xb32-1x1x8-50e_rgb_sthv2-tuned_batchsize-8_inferences.npz',
    },
    'tsn_flow_ssv2-tuned': {
        'model_name': 'tsn',
        'model_type': 'flow',
        'path2checkpoints': '',
        'path2inferences': '/data/local/u0151613/Qiuhan_Projects/DL_VideoClassification_Qiuhan/mmaction2_custom/model_tuning/tsn/tsn_r50_8xb32-1x1x8-50e_flow_sthv2-tuned_batchsize-8_inferences.npz',
    },
    'videomae_kinetics400-tuned': {
        'model_name': 'videomae',
        'model_type': 'rgb',
        'path2checkpoints': '',
        'path2inferences': '/data/local/u0151613/Qiuhan_Projects/DL_VideoClassification_Qiuhan/mmaction2_custom/model_tuning/videomae/videomae-base_kinetics400-tuned_batchsize-4_inferences.npz',
    },
    'videomae_ssv2-tuned': {
        'model_name': 'videomae',
        'model_type': 'rgb',
        'path2checkpoints': '',
        'path2inferences': '/data/local/u0151613/Qiuhan_Projects/DL_VideoClassification_Qiuhan/mmaction2_custom/model_tuning/videomae/videomae-base_sthv2-tuned_batchsize-4_inferences.npz',
    },
    'timesformer_kinetics400-tuned': {
        'model_name': 'timesformer',
        'model_type': 'rgb',
        'path2inferences': '/data/local/u0151613/Qiuhan_Projects/DL_VideoClassification_Qiuhan/mmaction2_custom/model_tuning/timesformer/timesformer_divST_8xb8-8x32x1-15e_kinetics400-tuned_batchsize-4_inferences.npz',
    },
    'mvit_ssv2-tuned': {
        'model_name': 'mvit_small',
        'model_type': 'rgb',
        'layer_list': [
            'backbone.transformer_layers.layers.11.attentions.0.temporal_fc',
            'backbone.transformer_layers.layers.11.attentions.1.dropout_layer',
            'backbone.transformer_layers.layers.11.ffns.0.layers.0.2',
            'backbone.transformer_layers.layers.11.ffns.0.norm',
        ],
        'path2checkpoints': '',
        'path2inferences': '/data/local/u0151613/Qiuhan_Projects/DL_VideoClassification_Qiuhan/mmaction2_custom/model_tuning/transformer/mvit-small-p244_k400-pre_16xb16-u16-100e_sthv2_tuned_transfer_batchsize8_inferences',
        'epoch_list': ['epoch_10', 'epoch_60', 'epoch90', 'epoch_150'],
    },
}
model_to_check_list = [
    'i3d_rgb_kinetics400-tuned', 'i3d_flow_kinetics400-tuned',
    'tsn_rgb_kinetics400-tuned', 'tsn_flow_kinetics400-tuned',
    'tsn_rgb_ssv2-tuned', 'tsn_flow_ssv2-tuned', 
    'videomae_kinetics400-tuned', 'videomae_ssv2-tuned',
]
video_process_order = ['training', 'green', 'small', 'ring', 'flip', 'right_hand', 'actorL', 'actorT', 'monkey_hand', 'monkey_tail']
output_folder = '/data/local/u0151613/Qiuhan_Projects/DL_VideoClassification_Qiuhan/mmaction2_custom/model_features/results'
os.makedirs(output_folder, exist_ok=True)
# Read monkey behavior index dist vector
with open("/data/local/u0151613/Qiuhan_Projects/DL_VideoClassification_Qiuhan/mmaction2_custom/monkey_behavior/results/Elmo_behavior_index_distmap.json", "r") as read_file:
    Elmo_behavior_index_dist_vector = json.load(read_file)['index_vector']
with open("/data/local/u0151613/Qiuhan_Projects/DL_VideoClassification_Qiuhan/mmaction2_custom/monkey_behavior/results/Louis_behavior_index_distmap.json", "r") as read_file:
    Louis_behavior_index_dist_vector = json.load(read_file)['index_vector']
with open("/data/local/u0151613/Qiuhan_Projects/DL_VideoClassification_Qiuhan/mmaction2_custom/monkey_behavior/results/Group_behavior_index_distmap.json", "r") as read_file:
    Group_behavior_index_dist_vector = json.load(read_file)['index_vector']
# Read model inferences and calculate:
# (1) confusion matrix
# (2) layer feature dist vector and correlate with monkey's
# for model_to_check in model_to_check_list:
for model_to_check in ['tsn_rgb_kinetics400-tuned', 'tsn_flow_kinetics400-tuned']:
    print('Model to check now:', model_to_check)
    model_inference_path = model_configs[model_to_check]['path2inferences']
    model_results = np.load(model_inference_path, allow_pickle=True)['Results'].item()
    model_checkpoint_list = model_results['checkpoint_list']
    model_layer_list = model_results['layer_list']
    # model_results['results']['checkpoint-3']['prediction_score']['green'] = [('grasp', xxx), ('touch', xxx), ('reach', xxx)]
    # model_results['results']['checkpoint-3']['confusion_matrix']['green'] = prediction score matrix of grasp/touch/reach
    # model_results['results']['checkpoint-3']['layer_distance_matrix']['green'][layer_name] = dist matrix of grasp/touch/reach
    
    # (1) Plot model confusion matrices among action types across checkpoints and videos
    fig, axes = plt.subplots(len(model_checkpoint_list), len(video_process_order), figsize=(6*len(video_process_order), 4*len(model_checkpoint_list)))
    fig.subplots_adjust(bottom=0.0, top=1.0, left=0.0, right=0.8, wspace=0.0, hspace=0.3)
    for checkpointI, checkpoint_name in enumerate(model_checkpoint_list):
        if checkpoint_name not in model_results['results']:
            checkpoint_name = checkpoint_name.replace('checkpoint-', 'epoch_')
        for videoI, video_name in enumerate(video_process_order):
            checkpoint_confusion_matrix = model_results['results'][checkpoint_name]['confusion_matrix'][video_name]
            cur_ax = axes[checkpointI, videoI]
            cur_cax = cur_ax.imshow(checkpoint_confusion_matrix, cmap='Blues', vmin=0, vmax=1)
            cur_ax.set_title('{} {}'.format(video_name, checkpoint_name), size=20)
            cur_ax.set_xticks([0, 1, 2], ['grasp', 'touch', 'reach'], fontsize='x-large')
            cur_ax.set_yticks([0, 1, 2], ['grasp', 'touch', 'reach'], fontsize='x-large')
    # Add color bar in the last column (last video)
    colorbar_ticks = [0, 0.25, 0.5, 0.75, 1]
    color_labels = ['0       ', '0.25     ', '0.5      ', '0.75     ', '1       ']
    colorBarPosition = [0.8, 0.0, 0.01, 0.8/len(model_checkpoint_list)]
    cb_ax = fig.add_axes(colorBarPosition)
    cbar = fig.colorbar(cur_cax, cax=cb_ax, ticks=colorbar_ticks, orientation='vertical')
    cbar.set_ticklabels(color_labels, fontsize='large')
    cbar.ax.set_ylabel('Model prediction score [0,1]', loc='center', rotation=270, fontsize='x-large')
    fig.suptitle('', fontsize=30, x=0.4, y=1.3)
    plt.savefig(os.path.join(output_folder, 'vis_{}_confusion_matrix.png'.format(model_to_check)),  bbox_inches='tight', dpi=100)
    plt.close()
    print('------ Confusion matrices plotted')
    
    # (2) Plot model layer feature dist maps across checkpoints, videos and layers
    # TODO
    # fig, axes = plt.subplots(len(model_layer_list), len(video_process_order), figsize=(6*len(video_process_order), 4*len(model_layer_list)))
    # fig.subplots_adjust(bottom=0.0, top=1.0, left=0.0, right=0.8, wspace=0.0, hspace=0.3)
    # for checkpointI, checkpoint_name in enumerate(model_checkpoint_list):
    #     for videoI, video_name in enumerate(video_process_order):
    #         for layerI, layer_name in enumerate(model_layer_list):
    # print('------ Confusion matrices plotted')
    
    # (3) Save model results in json files
    with open(os.path.join(output_folder, "{}_inferences.json".format(model_to_check)), "w") as outfile:
        json.dump(model_results, outfile, cls=NumpyArrayEncoder, indent=2, separators=(", ", ": "))


