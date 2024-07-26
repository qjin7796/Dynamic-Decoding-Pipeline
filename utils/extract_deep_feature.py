import numpy as np
import os, copy, json
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.stats import binomtest
from itertools import product
from json import JSONEncoder


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


# Model performance
video_name_list = [
    'green', 'small', 'ring', 'flip', 'right_hand', 'actorL', 'actorT', 'monkey_hand', 'monkey_tail', 'monkey_mean'
]  # video classes
model_results_path = '/data/local/u0151613/Qiuhan_Projects/Encoding_Qiuhan/extract_deep_features_042023/model_features/results'
output_path = '/data/local/u0151613/Qiuhan_Projects/Encoding_Qiuhan/compare_features_052023/deep_features'
model_to_check_list = [
    'i3d_rgb_kinetics400', 'i3d_flow_kinetics400',
    'tsn_rgb_kinetics400', 'tsn_flow_kinetics400',
    'tsn_rgb_ssv2', 'tsn_flow_ssv2', 
    'videomae_kinetics400', 'videomae_ssv2',
]
model_selected = {
    'i3d_flow_kinetics400': 'epoch_60',
    'tsn_flow_kinetics400': 'epoch_240',
    'tsn_flow_ssv2': 'epoch_280',
    'i3d_rgb_kinetics400': 'epoch_50',
    'tsn_rgb_kinetics400': 'epoch_260',
    'tsn_rgb_ssv2': 'epoch_300',
    'videomae_kinetics400': 'epoch_78',
    'videomae_ssv2': 'epoch_42',
}
model_feature_distmaps_file = os.path.join(model_results_path, 'selected_model_layer_feature_dist_heatmaps.json')
with open(model_feature_distmaps_file, "r") as infile:
    model_feature_distmaps = json.load(infile)
# print(model_feature_distmaps.keys())  # model, video, layer_name, dist_matrix

# Get confusion matrix, i.e. prediction score matrix, and layer output feature distance matrix
model_confusion_matrix_dict = {}  # model, video, pred_score_matrix
model_feature_distmaps_dict = {}  # model, video, layer_name, dist_matrix
for model_name in model_to_check_list:
    epoch_name = model_selected[model_name]
    # print(model_name)
    model_results_file = os.path.join(model_results_path, f'{model_name}-tuned_inferences.json')
    with open(model_results_file, "r") as infile:
        model_results = json.load(infile)
    
    # Get confusion matrix
    model_confusion_matrix_dict[model_name] = {}
    for video_name, video_score_matrix in model_results['results'][epoch_name]['confusion_matrix'].items():
        score_matrix = np.array(video_score_matrix)
        if 'videomae' in model_name:
            ## Normalize to [0,1] range and each row sum to 1
            score_matrix = (score_matrix - np.min(score_matrix)) / (np.max(score_matrix) - np.min(score_matrix))
            score_matrix = score_matrix / np.sum(score_matrix, axis=1, keepdims=True)
        model_confusion_matrix_dict[model_name][video_name] = score_matrix
    # print(model_selected[model_name])
    # Add monkey_mean = monkey_hand + monkey_tail / 2
    model_confusion_matrix_dict[model_name]['monkey_mean'] = (model_confusion_matrix_dict[model_name]['monkey_hand'] + model_confusion_matrix_dict[model_name]['monkey_tail']) / 2

    # Get ANN layer output feature distance matrix
    model_feature_distmaps_dict[model_name] = {}
    for video_name, layer_output in model_results['results'][epoch_name]['layer_distance_matrix'].items():
        model_feature_distmaps_dict[model_name][video_name] = {}
        for layer_name, dist_matrix in layer_output.items():
            model_feature_distmaps_dict[model_name][video_name][layer_name] = np.array(dist_matrix)
    # print(model_feature_distmaps_dict.keys())
# Save
with open(os.path.join(output_path, 'ANN_prediction_scores.json'), 'w') as outfile:
    json.dump(model_confusion_matrix_dict, outfile, cls=NumpyArrayEncoder)
with open(os.path.join(output_path, 'ANN_layer_feature_distmaps.json'), 'w') as outfile:
    json.dump(model_feature_distmaps_dict, outfile, cls=NumpyArrayEncoder)

# Plot confusion matrices
fig, axes = plt.subplots(len(model_to_check_list), len(video_name_list), figsize=(64, 8*len(model_to_check_list)))
for model_name, model_confusion_matrices in model_confusion_matrix_dict.items():
    for vi, video_name in enumerate(video_name_list):
        conf_matrix = model_confusion_matrices[video_name]
        cur_ax = axes[model_to_check_list.index(model_name)][vi]
        cur_cax = cur_ax.imshow(conf_matrix, cmap='Blues', vmin=0, vmax=1)
        cur_ax.set_title(f'{model_name}\n{model_selected[model_name]}\n{video_name}', size=28)
        cur_ax.set_xticks([0, 1, 2], ['grasp', 'touch', 'reach'], fontsize='xx-large')
        cur_ax.set_xlabel('Choice', fontsize='xx-large')
        cur_ax.set_yticks([0, 1, 2], ['grasp', 'touch', 'reach'], fontsize='xx-large')
        cur_ax.set_ylabel('Video', fontsize='xx-large')
        fig.colorbar(cur_cax, ax=cur_ax, shrink=0.5)
plt.tight_layout()
plt.savefig(os.path.join(output_path, 'ANN_prediction_scores.png'))
plt.close()

