# Analyze classification behavior data:
# 1. Compute confusion matrix grasp v.s. touch v.s. reach
# 2. Adjust behavioral bias using binomial test (if visible bias in behavior)
# 3. Compute the Youden index (from confusion matrix)
# 4. Compute the distance map of the Youden index across videos, distance = sum of two Youden index
#    such that higher index = both categories can be well classified, and lower index = confusion between the two
# @Qiuhan Jin, 15/05/2023


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


def read_sample_test(behav_data, test_category, test_video_index):
    """ Vargin: sample_name (video set), behavior_test_list (field name in behav_data), category (action type)
        Function: extract the behavior data for the specific action type test
    """
    # behav_data (raw mat file) contains graspTestall, touchTestall, reachTestall
    # graspTestall contains 10 fields (last one unused for this analysis), each corresponding to a video set
    # behavior_test_data = n_trials * 3 array, column 0 = label, column 1 = choice, column 2 = assert
    behavior_test_data = behav_data[test_category][test_video_index, 0]
    return behavior_test_data


def get_behav_stats(behav_dict, sample_bias_dict, choice_index_list):
    grasp_choice_index = choice_index_list.index('grasp') + 1  # range [1,2,3]
    touch_choice_index = choice_index_list.index('touch') + 1
    reach_choice_index = choice_index_list.index('reach') + 1
    behav_bias_corrected = {}
    for sample_name, sample_bias in sample_bias_dict.items():
        grasp_test = behav_dict[sample_name]['grasp_data']
        touch_test = behav_dict[sample_name]['touch_data']
        reach_test = behav_dict[sample_name]['reach_data']
        if sample_bias == 'touch_bias_in_touch_reach':
            # i.e. for touch and reach tests, touch was always chosen
            # grasp vs rest
            grasp_vs_rest_TP = np.count_nonzero(grasp_test[:, 1] == grasp_choice_index)
            grasp_vs_rest_FN = np.count_nonzero(grasp_test[:, 1] != grasp_choice_index)
            grasp_vs_rest_FP = np.count_nonzero(touch_test[:, 1] == grasp_choice_index) + np.count_nonzero(reach_test[:, 1] == grasp_choice_index)
            grasp_vs_rest_TN = np.count_nonzero(touch_test[:, 1] != grasp_choice_index) + np.count_nonzero(reach_test[:, 1] != grasp_choice_index)
            grasp_num_trial = len(grasp_test)
            nongrasp_num_trial = len(touch_test) + len(reach_test)
            # touch vs rest
            num_touch_reach_choice = np.count_nonzero(touch_test[:, 1] != grasp_choice_index) + np.count_nonzero(reach_test[:, 1] != grasp_choice_index)
            touch_vs_rest_TP = int(num_touch_reach_choice / 4)
            touch_vs_rest_FN = np.count_nonzero(touch_test[:, 1] == grasp_choice_index) + touch_vs_rest_TP
            touch_vs_rest_FP = np.count_nonzero(grasp_test[:, 1] == touch_choice_index) + touch_vs_rest_TP
            touch_vs_rest_TN = np.count_nonzero(grasp_test[:, 1] != touch_choice_index) + len(reach_test) - touch_vs_rest_TP
            touch_num_trial = len(touch_test)
            nontouch_num_trial = len(grasp_test) + len(reach_test)
            # reach vs rest
            reach_vs_rest_TP = touch_vs_rest_TP
            reach_vs_rest_FN = np.count_nonzero(reach_test[:, 1] == grasp_choice_index) + touch_vs_rest_TP
            reach_vs_rest_FP = np.count_nonzero(grasp_test[:, 1] == reach_choice_index) + touch_vs_rest_TP
            reach_vs_rest_TN = np.count_nonzero(grasp_test[:, 1] != reach_choice_index) + len(touch_test) - touch_vs_rest_TP
            reach_num_trial = len(reach_test)
            nonreach_num_trial = len(grasp_test) + len(touch_test)
        elif sample_bias == 'touch_bias_in_all':
            # i.e. for all tests, touch was always chosen
            grasp_num_trial = len(grasp_test)
            nongrasp_num_trial = len(touch_test) + len(reach_test)
            touch_num_trial = len(touch_test)
            nontouch_num_trial = len(grasp_test) + len(reach_test)
            reach_num_trial = len(reach_test)
            nonreach_num_trial = len(grasp_test) + len(touch_test)
            # grasp vs rest
            grasp_vs_rest_TP = int(grasp_num_trial/3)
            grasp_vs_rest_FN = int(2*grasp_num_trial/3)
            grasp_vs_rest_FP = int(nongrasp_num_trial/3)
            grasp_vs_rest_TN = int(2*nongrasp_num_trial/3)
            # touch vs rest
            touch_vs_rest_TP = int(touch_num_trial/3)
            touch_vs_rest_FN = int(2*touch_num_trial/3)
            touch_vs_rest_FP = int(nontouch_num_trial/3)
            touch_vs_rest_TN = int(2*nontouch_num_trial/3)
            # reach vs rest
            reach_vs_rest_TP = int(reach_num_trial/3)
            reach_vs_rest_FN = int(2*reach_num_trial/3)
            reach_vs_rest_FP = int(nonreach_num_trial/3)
            reach_vs_rest_TN = int(2*nonreach_num_trial/3)
        elif sample_bias == 'no_bias':
            # i.e. no bias towards any choice
            # grasp vs rest
            grasp_vs_rest_TP = np.count_nonzero(grasp_test[:, 1] == grasp_choice_index)
            grasp_vs_rest_FN = np.count_nonzero(grasp_test[:, 1] != grasp_choice_index)
            grasp_vs_rest_FP = np.count_nonzero(touch_test[:, 1] == grasp_choice_index) + np.count_nonzero(reach_test[:, 1] == grasp_choice_index)
            grasp_vs_rest_TN = np.count_nonzero(touch_test[:, 1] != grasp_choice_index) + np.count_nonzero(reach_test[:, 1] != grasp_choice_index)
            grasp_num_trial = len(grasp_test)
            nongrasp_num_trial = len(touch_test) + len(reach_test)
            # touch vs rest
            touch_vs_rest_TP = np.count_nonzero(touch_test[:, 1] == touch_choice_index)
            touch_vs_rest_FN = np.count_nonzero(touch_test[:, 1] != touch_choice_index)
            touch_vs_rest_FP = np.count_nonzero(grasp_test[:, 1] == touch_choice_index) + np.count_nonzero(reach_test[:, 1] == touch_choice_index)
            touch_vs_rest_TN = np.count_nonzero(grasp_test[:, 1] != touch_choice_index) + np.count_nonzero(reach_test[:, 1] != touch_choice_index)
            touch_num_trial = len(touch_test)
            nontouch_num_trial = len(grasp_test) + len(reach_test)
            # reach vs rest
            reach_vs_rest_TP = np.count_nonzero(reach_test[:, 1] == reach_choice_index)
            reach_vs_rest_FN = np.count_nonzero(reach_test[:, 1] != reach_choice_index)
            reach_vs_rest_FP = np.count_nonzero(grasp_test[:, 1] == reach_choice_index) + np.count_nonzero(touch_test[:, 1] == reach_choice_index)
            reach_vs_rest_TN = np.count_nonzero(grasp_test[:, 1] != reach_choice_index) + np.count_nonzero(touch_test[:, 1] != reach_choice_index)
            reach_num_trial = len(reach_test)
            nonreach_num_trial = len(grasp_test) + len(touch_test)
        grasp_vs_rest_TPR = grasp_vs_rest_TP / grasp_num_trial
        grasp_vs_rest_FPR = grasp_vs_rest_FP / nongrasp_num_trial
        touch_vs_rest_TPR = touch_vs_rest_TP / touch_num_trial
        touch_vs_rest_FPR = touch_vs_rest_FP / nontouch_num_trial
        reach_vs_rest_TPR = reach_vs_rest_TP / reach_num_trial
        reach_vs_rest_FPR = reach_vs_rest_FP / nonreach_num_trial
        print(sample_name, sample_bias, '\n', 'grasp vs rest\n',
              ' --- TP', grasp_vs_rest_TP, 'FN', grasp_vs_rest_FN,
              'FP', grasp_vs_rest_FP, 'TN', grasp_vs_rest_TN,
              'TPR', grasp_vs_rest_TPR, 'FPR', grasp_vs_rest_FPR)
        print(' touch vs rest\n',
              ' --- TP', touch_vs_rest_TP, 'FN', touch_vs_rest_FN,
              'FP', touch_vs_rest_FP, 'TN', touch_vs_rest_TN,
              'TPR', touch_vs_rest_TPR, 'FPR', touch_vs_rest_FPR)
        print(' reach vs rest\n',
              ' --- TP', reach_vs_rest_TP, 'FN', reach_vs_rest_FN,
              'FP', reach_vs_rest_FP, 'TN', reach_vs_rest_TN,
              'TPR', reach_vs_rest_TPR, 'FPR', reach_vs_rest_FPR)
        behav_bias_corrected[sample_name] = {
            'bias_type': sample_bias,
            'grasp_vs_rest': {
                'TP': grasp_vs_rest_TP, 'FP': grasp_vs_rest_FP,
                'TN': grasp_vs_rest_TN, 'FN': grasp_vs_rest_FN,
                'TPR': grasp_vs_rest_TPR, 'FPR': grasp_vs_rest_FPR,
                'grasp_trial_count': grasp_num_trial,
                'nongrasp_trial_count': nongrasp_num_trial,
            },
            'touch_vs_rest': {
                'TP': touch_vs_rest_TP, 'FP': touch_vs_rest_FP,
                'TN': touch_vs_rest_TN, 'FN': touch_vs_rest_FN,
                'TPR': touch_vs_rest_TPR, 'FPR': touch_vs_rest_FPR,
                'touch_trial_count': touch_num_trial,
                'nontouch_trial_count': nontouch_num_trial,
            },
            'reach_vs_rest': {
                'TP': reach_vs_rest_TP, 'FP': reach_vs_rest_FP,
                'TN': reach_vs_rest_TN, 'FN': reach_vs_rest_FN,
                'TPR': reach_vs_rest_TPR, 'FPR': reach_vs_rest_FPR,
                'reach_trial_count': reach_num_trial,
                'nonreach_trial_count': nonreach_num_trial,
            },
        }
    return behav_bias_corrected


def correct_behav_conf(behav_conf, sample_bias_dict):
    """ Correct behavior bias in confusion matrix. Same idea as in get_behav_stats().
    """
    conf_bias_corrected = copy.deepcopy(behav_conf)
    for sample_name, sample_bias in sample_bias_dict.items():
        sample_conf = behav_conf[sample_name]
        if sample_bias == 'touch_bias_in_touch_reach':
            # new touch-reach stats = mean of [touch_vs_touch, touch_vs_reach, reach_vs_touch, reach_vs_reach]
            touch_reach_stat = np.mean([sample_conf[1,1], sample_conf[1,2], sample_conf[2,1], sample_conf[2,2]])
            conf_bias_corrected[sample_name][1,1] = touch_reach_stat
            conf_bias_corrected[sample_name][1,2] = touch_reach_stat
            conf_bias_corrected[sample_name][2,1] = touch_reach_stat
            conf_bias_corrected[sample_name][2,2] = touch_reach_stat
        elif sample_bias == 'touch_bias_in_all':
            # the entire confusion matrix -> mean of all
            corrected_stat = sample_conf.mean()
            conf_bias_corrected[sample_name] = np.tile(corrected_stat, (3,3))
    return conf_bias_corrected


def read_accuracy_from_conf_bias_corrected(conf_bias_corrected_dict):
    """ Read and save the diagonal in all bias controled confusion matrices in the input dict.
    """
    accuracy_dict = {}
    for video_name, conf_matrix in conf_bias_corrected_dict.items():
        accuracy_dict[video_name] = np.array([conf_matrix[0,0], conf_matrix[1,1], conf_matrix[2,2]])
    return accuracy_dict


# Behavior data structure
test_video_order = {
    'flip': 0, 'right_hand': 1, 'green': 2, 'ring': 3, 'small': 4, 'actorL': 5, 'actorT': 6, 
    'monkey_hand': 7, 'monkey_tail': 8, 'monkey_mean': 9, 
}
test_video_name_list = [
    'green', 'small', 'ring', 'flip', 'right_hand', 'actorL', 'actorT', 'monkey_hand', 'monkey_tail', 'monkey_mean'
]  # video classes
test_video_index_list = [2, 4, 3, 0, 1, 5, 6, 7, 8]  # video class index in the behavior data
test_category_list = ['graspTestall', 'touchTestall', 'reachTestall'] 
Louis_choice_index_list = ['grasp', 'reach', 'touch']  # choices available
Elmo_choice_index_list = ['grasp', 'touch', 'reach']  # choices available
category_order = ['grasp', 'touch', 'reach']  # analysis done in this order
# input_path = '/data/local/u0151613/Qiuhan_Projects/DL_VideoClassification_Qiuhan/mmaction2_custom/monkey_behavior/data'
# output_path = '/data/local/u0151613/Qiuhan_Projects/DL_VideoClassification_Qiuhan/mmaction2_custom/monkey_behavior/results'
input_path = '/data/local/u0151613/Qiuhan_Projects/Encoding_Qiuhan/compare_features_052023/data'
output_path = '/data/local/u0151613/Qiuhan_Projects/Encoding_Qiuhan/compare_features_052023/behavior_features'
Louis_test_bias_dict = {
    'green': 'no_bias',
    'small': 'no_bias',
    'ring': 'touch_bias_in_all',
    'flip': 'touch_bias_in_touch_reach',
    'right_hand': 'no_bias',
    'actorL': 'touch_bias_in_touch_reach',
    'actorT': 'touch_bias_in_touch_reach',
    'monkey_hand': 'touch_bias_in_touch_reach',
    'monkey_tail': 'touch_bias_in_touch_reach',
    'monkey_mean': 'touch_bias_in_touch_reach',
}
Elmo_test_bias_dict = {
    'green': 'no_bias',
    'small': 'no_bias',
    'ring': 'no_bias',
    'flip': 'touch_bias_in_all',
    'right_hand': 'touch_bias_in_touch_reach',
    'actorL': 'touch_bias_in_touch_reach',
    'actorT': 'touch_bias_in_touch_reach',
    'monkey_hand': 'touch_bias_in_touch_reach',
    'monkey_tail': 'touch_bias_in_touch_reach',
    'monkey_mean': 'touch_bias_in_touch_reach',
}
# Write params and save in json
behavior_data_analysis_params = {
    'test_video_index_in_raw_data': test_video_order,
    'test_video_category_list': test_category_list,
    'analysis_video_order_list': test_video_name_list,
    'analysis_video_index_list': test_video_index_list,
    'analysis_category_orer_list': category_order,
    'Louis_choice_index_list': Louis_choice_index_list,
    'Elmo_choice_index_list': Elmo_choice_index_list,
    'Louis_behavior_bias_predefined': Louis_test_bias_dict,
    'Elmo_behavior_bias_predefined': Elmo_test_bias_dict,
    'data_input_path': input_path,
    'data_output_path': output_path,
    'behavior_distance_measure': 'Youden index = TPR-FPR for each [category v.s. rest all]; Youden index distance between category1 and category2 = sum of their Youden index.',
}
with open(os.path.join(output_path, "behavior_data_analysis_params.json"), "w") as outfile:
    json.dump(behavior_data_analysis_params, outfile, indent=4)

#######################################################
######################## Louis ########################
#######################################################
# Read all behavior data
Louis_behav = sio.loadmat(os.path.join(input_path, 'Louis_behavInfo.mat'))
Louis_behav_dict = {}
for sampleI, sample_name in enumerate(test_video_name_list):
    if sample_name == 'monkey_mean':
        continue
    sample_index = test_video_index_list[sampleI]
    sample_grasp_test = read_sample_test(Louis_behav, test_category_list[0], sample_index)
    sample_touch_test = read_sample_test(Louis_behav, test_category_list[1], sample_index)
    sample_reach_test = read_sample_test(Louis_behav, test_category_list[2], sample_index)
    print(sample_name, 'sample_index', sample_index, '\n ',
          'sample_grasp_test', sample_grasp_test.shape,
          'sample_touch_test', sample_touch_test.shape,
          'sample_reach_test', sample_reach_test.shape)
    sample_test = np.concatenate((sample_grasp_test, sample_touch_test, sample_reach_test))
    # print('sample_test', sample_test.shape)
    sample_y_true = sample_test[:, 0]  # test label
    sample_y_pred = sample_test[:, 1]  # choice
    # Write to dict: cat_data = concat of all categories
    Louis_behav_dict[sample_name] = {
        'grasp_data': sample_grasp_test,
        'touch_data': sample_touch_test,
        'reach_data': sample_reach_test,
        # 'cat_data': sample_test,
        # 'cat_y_true': sample_y_true,
        # 'cat_y_pred': sample_y_pred,
    }
# Concatenate monkey_hand and monkey_tail to monkey_mean
Louis_behav_dict['monkey_mean'] = {
    'grasp_data': np.concatenate((Louis_behav_dict['monkey_hand']['grasp_data'], Louis_behav_dict['monkey_tail']['grasp_data'])),
    'touch_data': np.concatenate((Louis_behav_dict['monkey_hand']['touch_data'], Louis_behav_dict['monkey_tail']['touch_data'])),
    'reach_data': np.concatenate((Louis_behav_dict['monkey_hand']['reach_data'], Louis_behav_dict['monkey_tail']['reach_data'])),
}
print('monkey_mean', 'sample_index', 9, '\n ',
      'sample_grasp_test', Louis_behav_dict['monkey_mean']['grasp_data'].shape,
      'sample_touch_test', Louis_behav_dict['monkey_mean']['touch_data'].shape,
      'sample_reach_test', Louis_behav_dict['monkey_mean']['reach_data'].shape)
# Save behavior test data in json
with open(os.path.join(output_path, "Louis_behavior_test.json"), "w") as outfile:
    json.dump(Louis_behav_dict, outfile, cls=NumpyArrayEncoder, indent=4)

# Louis confusion matrix
# row: grasp/touch/reach video
# col: choice = grasp/touch/reach
Louis_behav_conf = {
    'training': np.array([
        [0.9571, 0.0195, 0.0234],
        [0.0238, 0.8774, 0.0988],
        [0.0404, 0.1009, 0.8587],
    ]),
}
for gen_test_name, gen_test_behav in Louis_behav_dict.items():
    # numTrials x 3, second col is choice, choice labels are different for two monkeys
    grasp_choice_index = Louis_choice_index_list.index('grasp') + 1  # range [1,2,3]
    touch_choice_index = Louis_choice_index_list.index('touch') + 1
    reach_choice_index = Louis_choice_index_list.index('reach') + 1
    # Get choice percentage for each label
    grasp_test_behav = gen_test_behav['grasp_data']
    grasp_test_grasp_freq = np.sum(grasp_test_behav[:, 1] == grasp_choice_index) / grasp_test_behav.shape[0]
    grasp_test_touch_freq = np.sum(grasp_test_behav[:, 1] == touch_choice_index) / grasp_test_behav.shape[0]
    grasp_test_reach_freq = np.sum(grasp_test_behav[:, 1] == reach_choice_index) / grasp_test_behav.shape[0]
    touch_test_behav = gen_test_behav['touch_data']
    touch_test_grasp_freq = np.sum(touch_test_behav[:, 1] == grasp_choice_index) / touch_test_behav.shape[0]
    touch_test_touch_freq = np.sum(touch_test_behav[:, 1] == touch_choice_index) / touch_test_behav.shape[0]
    touch_test_reach_freq = np.sum(touch_test_behav[:, 1] == reach_choice_index) / touch_test_behav.shape[0]
    reach_test_behav = gen_test_behav['reach_data']
    reach_test_grasp_freq = np.sum(reach_test_behav[:, 1] == grasp_choice_index) / reach_test_behav.shape[0]
    reach_test_touch_freq = np.sum(reach_test_behav[:, 1] == touch_choice_index) / reach_test_behav.shape[0]
    reach_test_reach_freq = np.sum(reach_test_behav[:, 1] == reach_choice_index) / reach_test_behav.shape[0]
    Louis_behav_conf[gen_test_name] = np.array([
        [grasp_test_grasp_freq, grasp_test_touch_freq, grasp_test_reach_freq],
        [touch_test_grasp_freq, touch_test_touch_freq, touch_test_reach_freq],
        [reach_test_grasp_freq, reach_test_touch_freq, reach_test_reach_freq],
    ])
    print(gen_test_name, 'confusion matrix', Louis_behav_conf[gen_test_name])
# print(Louis_behav_conf.keys())
# Save behavior confusion matrix in json
with open(os.path.join(output_path, "Louis_behavior_confusion_matrix.json"), "w") as outfile:
    json.dump(Louis_behav_conf, outfile, cls=NumpyArrayEncoder, indent=4)

# Plot confusion matrices
fig, axes = plt.subplots(1, len(test_video_name_list), figsize=(64, 4))
for sample_name, sample_dict in Louis_behav_dict.items():
    conf_matrix = Louis_behav_conf[sample_name]  # skip 'training'
    cur_ax = axes[test_video_name_list.index(sample_name)]
    cur_cax = cur_ax.imshow(conf_matrix, cmap='Blues', vmin=0, vmax=1)
    cur_ax.set_title(f'{sample_name}\n', size=28)
    cur_ax.set_xticks([0, 1, 2], ['grasp', 'touch', 'reach'], fontsize='xx-large')
    cur_ax.set_xlabel('Choice', fontsize='xx-large')
    cur_ax.set_yticks([0, 1, 2], ['grasp', 'touch', 'reach'], fontsize='xx-large')
    cur_ax.set_ylabel('Video', fontsize='xx-large')
# Add color bar
colorbar_ticks = [0, 0.25, 0.5, 0.75, 1]
color_labels = ['0       ', '0.25     ', '0.5      ', '0.75     ', '1       ']
colorBarPosition = [0.8, 0.0, 0.01, 0.85]
fig.subplots_adjust(bottom=0.0, top=0.85, left=0.0, right=0.8,
                    wspace=0.0, hspace=0.3)
cb_ax = fig.add_axes(colorBarPosition)
cbar = fig.colorbar(cur_cax, cax=cb_ax, ticks=colorbar_ticks, orientation='vertical')
cbar.set_ticklabels(color_labels, fontsize='large')
cbar.ax.set_ylabel('Choice frequency', loc='center', rotation=270, fontsize='xx-large')
fig.suptitle('Louis Confusion Matrix', fontsize=40, x=0.4, y=1.3)
plt.savefig(os.path.join(output_path, 'vis_Louis_behavior_confusion_matrix.png'), bbox_inches='tight', dpi=100)
plt.close()

# Check behavior bias (binomial test)
# binomtest(x, n, p, alternative='greater')
# x: number of success
# n: number of trials
# p: chance level accuracy
# alternative = 'greater' (above chance), 'less' (below chance), 'two-sided' (two tail)
p_chance = 1 / len(category_order)
category_pairs = list(product(category_order, repeat=2))
Louis_behav_binomtest_dict = {}
fig, axes = plt.subplots(1, len(test_video_name_list), figsize=(64, 4))
for sample_name, sample_dict in Louis_behav_dict.items():
    Louis_behav_binomtest_dict[sample_name] = {}
    binomtest_3x3 = np.zeros((len(category_pairs),))
    binomtest_sig_3x3 = np.zeros((len(category_pairs),))
    for pairI, pairTuple in enumerate(category_pairs):
        pairName = 'given_{}_choose_{}'.format(pairTuple[0], pairTuple[1])
        # print(pairName)
        category = pairTuple[0]
        choice = pairTuple[1]
        # category_index = category_order.index(category)
        choice_index = Louis_choice_index_list.index(choice) + 1  # range [1,2,3]
        sample_category_test = Louis_behav_dict[sample_name]['{}_data'.format(category)]
        sample_category_num_trial = len(sample_category_test)
        sample_category_num_choice = np.count_nonzero(sample_category_test[:, 1] == choice_index)
        # binomial test
        sample_category_binomtest = binomtest(
            sample_category_num_choice, sample_category_num_trial, p_chance, alternative='greater'
        ).pvalue
        # Louis_behav_binomtest_dict[sample_name][pairName] = sample_category_binomtest
        # print(sample_name, category, 'choose', choice,
        #       'num_trial', sample_category_num_trial,
        #       'num_success', sample_category_num_choice,
        #       'if>p_chance', sample_category_binomtest, sample_category_binomtest < 0.05)
        binomtest_3x3[pairI] = sample_category_binomtest
        binomtest_sig_3x3[pairI] = int(sample_category_binomtest < 0.05)
    binomtest_reshape = binomtest_3x3.reshape(3, 3)
    binomtest_sig_reshape = binomtest_sig_3x3.reshape(3, 3)
    # print(sample_name, 'grasp/touch/reach binom test', binomtest_reshape)
    # print(sample_name, 'sig', binomtest_sig_reshape)
    Louis_behav_binomtest_dict[sample_name]['binomtest_3x3'] = binomtest_reshape
    cur_ax = axes[test_video_name_list.index(sample_name)]
    cur_cax = cur_ax.imshow(binomtest_reshape, cmap='Blues', vmin=0, vmax=1)
    for rowI in range(binomtest_sig_reshape.shape[0]):
        for colI in range(binomtest_sig_reshape.shape[1]):
            if binomtest_sig_reshape[rowI, colI] == 1:
                cur_ax.text(colI, rowI, '*', horizontalalignment='center', verticalalignment='center',
                            color='red', fontsize=50)
    cur_ax.set_title(f'{sample_name}\n', size=28)
    cur_ax.set_xticks([0, 1, 2], ['grasp', 'touch', 'reach'], fontsize='xx-large')
    cur_ax.set_xlabel('Choice', fontsize='xx-large')
    cur_ax.set_yticks([0, 1, 2], ['grasp', 'touch', 'reach'], fontsize='xx-large')
    cur_ax.set_ylabel('Video', fontsize='xx-large')
# Add color bar
colorbar_ticks = [0, 0.25, 0.5, 0.75, 1]
color_labels = ['0       ', '0.25     ', '0.5      ', '0.75     ', '1       ']
colorBarPosition = [0.8, 0.0, 0.01, 0.85]
fig.subplots_adjust(bottom=0.0, top=0.85, left=0.0, right=0.8,
                    wspace=0.0, hspace=0.3)
cb_ax = fig.add_axes(colorBarPosition)
cbar = fig.colorbar(cur_cax, cax=cb_ax, ticks=colorbar_ticks, orientation='vertical')
cbar.set_ticklabels(color_labels, fontsize='large')
cbar.ax.set_ylabel('Binomial test p-value', loc='center', rotation=270, fontsize='xx-large')
fig.suptitle('Louis Binomial test: accuracy v.s. chance', fontsize=40, x=0.4, y=1.3)
plt.savefig(os.path.join(output_path, 'vis_Louis_behavior_bias.png'), bbox_inches='tight', dpi=100)
plt.close()
# Save bias significance in json
with open(os.path.join(output_path, "Louis_behavior_bias_significance.json"), "w") as outfile:
    json.dump(Louis_behav_binomtest_dict, outfile, cls=NumpyArrayEncoder, indent=4)

# Correction for behavior bias
# Bias toward touch in touch and reach tests
# Adjustment: label half of touch choices as reach in both tests
# Prerequisites for concave ROC curve:
#   (1) number of touch choices in reach test < number of total reach trials
#   (2) number of touch choices in touch test > 1/2 * number of total touch trials
# Result of adjustment:
#   AUC formula:
#     TPR = TP/(TP+FN)
#     FPR = FP/(FP+TN)
#     AUC = area below all sample ROC curve (TPR-FPR curve)
#   For individual samples:
#     x = number of touch choices in reach test adjusted to reach = the same number adjusted in touch test
#     Assume equal number of trials (n) in grasp/touch/reach, TP + FN = n, TN + FP = 2n
#     tp_touch = number of true positive choices in touch test
#     tp_reach = number of true positive choices in reach test
#     Prerequisite: tp_reach < x < n/2 < tp_touch
#     Result of adjustment in tp:
#       tp_touch_adjusted = tp_touch - x
#       tp_reach_adjusted = tp_reach + x
#  Result of adjustment in AUC (touch v.s. rest):
#    TPR_touch_adjusted = TPR_touch - x/n < TPR_touch - 0.5
#    FPR_touch_adjusted = FPR_touch - x/2n < FPR_touch - 0.25
#    More adjustment = more reduction in TPR_touch and FPR_touch = smaller AUC_touch
#  Result of adjustment in AUC (reach v.s. rest):
#     TPR_reach_adjusted = TPR_reach + x/n < TPR_reach + 0.5
#     FPR_reach_adjusted = FPR_reach + x/2n < FPR_reach + 0.25
#     More adjustment = more increase in TPR_reach and FPR_reach = bigger AUC_reach
Louis_behav_bias_corrected_conf = correct_behav_conf(Louis_behav_conf, Louis_test_bias_dict)
Louis_behav_bias_corrected_stats = get_behav_stats(Louis_behav_dict, Louis_test_bias_dict, Louis_choice_index_list)
# # Save behavior conf bias corrected in json
# with open(os.path.join(output_path, "Louis_behavior_stats_bias_corrected.json"), "w") as outfile:
#     json.dump(Louis_behav_bias_corrected_stats, outfile, cls=NumpyArrayEncoder, indent=4)
# with open(os.path.join(output_path, "Louis_confustion_matrix_bias_corrected.json"), "w") as outfile:
#     json.dump(Louis_behav_bias_corrected_conf, outfile, cls=NumpyArrayEncoder, indent=4)

# Compute Youden index = TPR - FPR
Louis_grasp_vs_rest_FPR = []
Louis_touch_vs_rest_FPR = []
Louis_reach_vs_rest_FPR = []
Louis_grasp_vs_rest_TPR = []
Louis_touch_vs_rest_TPR = []
Louis_reach_vs_rest_TPR = []
for sample_name, sample_data in Louis_behav_bias_corrected_stats.items():
    Louis_grasp_vs_rest_FPR.append(sample_data['grasp_vs_rest']['FPR'])
    Louis_touch_vs_rest_FPR.append(sample_data['touch_vs_rest']['FPR'])
    Louis_reach_vs_rest_FPR.append(sample_data['reach_vs_rest']['FPR'])
    Louis_grasp_vs_rest_TPR.append(sample_data['grasp_vs_rest']['TPR'])
    Louis_touch_vs_rest_TPR.append(sample_data['touch_vs_rest']['TPR'])
    Louis_reach_vs_rest_TPR.append(sample_data['reach_vs_rest']['TPR'])
Louis_grasp_index = [Louis_grasp_vs_rest_TPR[x] - Louis_grasp_vs_rest_FPR[x] for x in range(len(test_video_name_list))]
Louis_touch_index = [Louis_touch_vs_rest_TPR[x] - Louis_touch_vs_rest_FPR[x] for x in range(len(test_video_name_list))]
Louis_reach_index = [Louis_reach_vs_rest_TPR[x] - Louis_reach_vs_rest_FPR[x] for x in range(len(test_video_name_list))]
# # Save behavior index in json
# with open(os.path.join(output_path, "Louis_behavior_index.json"), "w") as outfile:
#     json.dump({
#         'grasp_Youden_index': Louis_grasp_index, 
#         'touch_Youden_index': Louis_touch_index,
#         'reach_Youden_index': Louis_reach_index,
#     }, outfile, indent=4)

#######################################################
# Compute pairwise Youden index difference (dist map) #
# Note that the difference here is sum of two Youden index such that higher distance means both categories 
# can be classified well by the subject, and lower distance means there is confusion among the two categories
Louis_behavior_index_dist_matrix = {}
Louis_behavior_index_dist_vector = []
for videoI, video_name in enumerate(list(Louis_behav_bias_corrected_stats.keys())):
    tmp = np.zeros((3, 3))
    tmp[0, 1] = abs(Louis_grasp_index[videoI] + Louis_touch_index[videoI])
    tmp[0, 2] = abs(Louis_grasp_index[videoI] + Louis_reach_index[videoI])
    tmp[1, 0] = abs(Louis_touch_index[videoI] + Louis_grasp_index[videoI])
    tmp[1, 2] = abs(Louis_touch_index[videoI] + Louis_reach_index[videoI])
    tmp[2, 0] = abs(Louis_reach_index[videoI] + Louis_grasp_index[videoI])
    tmp[2, 1] = abs(Louis_reach_index[videoI] + Louis_touch_index[videoI])
    Louis_behavior_index_dist_matrix[video_name] = tmp
    Louis_behavior_index_dist_vector.append(tmp[0,1])
    Louis_behavior_index_dist_vector.append(tmp[0,2])
    Louis_behavior_index_dist_vector.append(tmp[1,2])
    print('Louis Youden index dist map:', video_name, tmp)
# Louis_behav_dist_map_output_filename = os.path.join(output_path, 'Louis_behavior_distance_map.npz')
# print('------ Louis behavior distance map saved to:', Louis_behav_dist_map_output_filename)
# np.savez(Louis_behav_dist_map_output_filename, Results=Louis_behavior_index_dist_matrix)
# Save behavior dist map in json
# Louis_behav_dist_map_output_filename = json.dumps(Louis_behavior_index_dist_matrix, indent = 4)
Louis_distmap_tosave = copy.deepcopy(Louis_behavior_index_dist_matrix)
Louis_distmap_tosave['index_vector_video_order'] = test_video_name_list
Louis_distmap_tosave['index_vector'] = Louis_behavior_index_dist_vector
with open(os.path.join(output_path, "Louis_behavior_index_distmap.json"), "w") as outfile:
    json.dump(Louis_distmap_tosave, outfile, cls=NumpyArrayEncoder, indent=4)

# Plot behavior Youden index dist map
fig, axes = plt.subplots(1, len(Louis_behav_bias_corrected_stats.keys()), figsize=(64, 4))
for sample_name, diff_data in Louis_behavior_index_dist_matrix.items():
    cur_ax = axes[list(Louis_behav_bias_corrected_stats.keys()).index(sample_name)]
    cur_cax = cur_ax.imshow(diff_data, cmap='Blues', vmin=0, vmax=2)
    cur_ax.set_title(f'{sample_name}\n', size=28)
    cur_ax.set_xticks([0, 1, 2], ['grasp', 'touch', 'reach'], fontsize='xx-large')
    cur_ax.set_yticks([0, 1, 2], ['grasp', 'touch', 'reach'], fontsize='xx-large')
# Add color bar
colorbar_ticks = [0, 0.5, 1, 1.5, 2]
color_labels = ['0       ', '0.5     ', '1      ', '1.5     ', '2       ']
colorBarPosition = [0.8, 0.0, 0.01, 0.85]
fig.subplots_adjust(bottom=0.0, top=0.85, left=0.0, right=0.8,
                    wspace=0.0, hspace=0.3)
cb_ax = fig.add_axes(colorBarPosition)
cbar = fig.colorbar(cur_cax, cax=cb_ax, ticks=colorbar_ticks, orientation='vertical')
cbar.set_ticklabels(color_labels, fontsize='large')
cbar.ax.set_ylabel('Sum of Youden index [0,2]', loc='center', rotation=270, fontsize='xx-large')
fig.suptitle('Louis Behavior Distance Map (sum of Youden index [TPR-FPR])', fontsize=40, x=0.4, y=1.3)
plt.savefig(os.path.join(output_path, 'vis_Louis_behavior_index_distmap.png'),  bbox_inches='tight', dpi=100)
plt.close()



######################################################
######################## Elmo ########################
######################################################
# Read all behavior data
Elmo_behav = sio.loadmat(os.path.join(input_path, 'Elmo_behavInfo.mat'))
Elmo_behav_dict = {}
for sampleI, sample_name in enumerate(test_video_name_list):
    if sample_name == 'monkey_mean':
        continue
    sample_index = test_video_index_list[sampleI]
    sample_grasp_test = read_sample_test(Elmo_behav, test_category_list[0], sample_index)
    sample_touch_test = read_sample_test(Elmo_behav, test_category_list[1], sample_index)
    sample_reach_test = read_sample_test(Elmo_behav, test_category_list[2], sample_index)
    print(sample_name, 'sample_index', sample_index, '\n ',
          'sample_grasp_test', sample_grasp_test.shape,
          'sample_touch_test', sample_touch_test.shape,
          'sample_reach_test', sample_reach_test.shape)
    sample_test = np.concatenate((sample_grasp_test, sample_touch_test, sample_reach_test))
    # print('sample_test', sample_test.shape)
    sample_y_true = sample_test[:, 0]  # test label
    sample_y_pred = sample_test[:, 1]  # choice
    # Write to dict: cat_data = concat of all categories
    Elmo_behav_dict[sample_name] = {
        'grasp_data': sample_grasp_test,
        'touch_data': sample_touch_test,
        'reach_data': sample_reach_test,
        # 'cat_data': sample_test,
        # 'cat_y_true': sample_y_true,
        # 'cat_y_pred': sample_y_pred,
    }
# Concatenate monkey_hand and monkey_tail to monkey_mean
Elmo_behav_dict['monkey_mean'] = {
    'grasp_data': np.concatenate((Elmo_behav_dict['monkey_hand']['grasp_data'], Elmo_behav_dict['monkey_tail']['grasp_data'])),
    'touch_data': np.concatenate((Elmo_behav_dict['monkey_hand']['touch_data'], Elmo_behav_dict['monkey_tail']['touch_data'])),
    'reach_data': np.concatenate((Elmo_behav_dict['monkey_hand']['reach_data'], Elmo_behav_dict['monkey_tail']['reach_data'])),
}
print('monkey_mean', 'sample_index', 9, '\n ',
        'sample_grasp_test', Elmo_behav_dict['monkey_mean']['grasp_data'].shape,
        'sample_touch_test', Elmo_behav_dict['monkey_mean']['touch_data'].shape,
        'sample_reach_test', Elmo_behav_dict['monkey_mean']['reach_data'].shape)
# Save behavior test data in json
with open(os.path.join(output_path, "Elmo_behavior_test.json"), "w") as outfile:
    json.dump(Elmo_behav_dict, outfile, cls=NumpyArrayEncoder, indent=4)

# Elmo confusion matrix
# row: grasp/touch/reach video
# col: choice = grasp/touch/reach
Elmo_behav_conf = {
    'training': np.array([
        [0.97006652, 0.02993348, 0.0],
        [0.00336417, 0.80656013, 0.19007569],
        [0.0, 0.12903226, 0.87096774],
    ]),
}
for gen_test_name, gen_test_behav in Elmo_behav_dict.items():
    # numTrials x 3, second col is choice, choice labels are different for two monkeys
    grasp_choice_index = Elmo_choice_index_list.index('grasp') + 1  # range [1,2,3]
    touch_choice_index = Elmo_choice_index_list.index('touch') + 1
    reach_choice_index = Elmo_choice_index_list.index('reach') + 1
    # Get choice percentage for each label
    grasp_test_behav = gen_test_behav['grasp_data']
    grasp_test_grasp_freq = np.sum(grasp_test_behav[:, 1] == grasp_choice_index) / grasp_test_behav.shape[0]
    grasp_test_touch_freq = np.sum(grasp_test_behav[:, 1] == touch_choice_index) / grasp_test_behav.shape[0]
    grasp_test_reach_freq = np.sum(grasp_test_behav[:, 1] == reach_choice_index) / grasp_test_behav.shape[0]
    touch_test_behav = gen_test_behav['touch_data']
    touch_test_grasp_freq = np.sum(touch_test_behav[:, 1] == grasp_choice_index) / touch_test_behav.shape[0]
    touch_test_touch_freq = np.sum(touch_test_behav[:, 1] == touch_choice_index) / touch_test_behav.shape[0]
    touch_test_reach_freq = np.sum(touch_test_behav[:, 1] == reach_choice_index) / touch_test_behav.shape[0]
    reach_test_behav = gen_test_behav['reach_data']
    reach_test_grasp_freq = np.sum(reach_test_behav[:, 1] == grasp_choice_index) / reach_test_behav.shape[0]
    reach_test_touch_freq = np.sum(reach_test_behav[:, 1] == touch_choice_index) / reach_test_behav.shape[0]
    reach_test_reach_freq = np.sum(reach_test_behav[:, 1] == reach_choice_index) / reach_test_behav.shape[0]
    Elmo_behav_conf[gen_test_name] = np.array([
        [grasp_test_grasp_freq, grasp_test_touch_freq, grasp_test_reach_freq],
        [touch_test_grasp_freq, touch_test_touch_freq, touch_test_reach_freq],
        [reach_test_grasp_freq, reach_test_touch_freq, reach_test_reach_freq],
    ])
    print(gen_test_name, 'confusion matrix', Elmo_behav_conf[gen_test_name])
# print(Elmo_behav_conf.keys())
# Elmo_behav_conf_output_filename = os.path.join(output_path, 'Elmo_behavior_confusion_matrix.npz')
# print('------ Elmo behavior confusion matrix saved to:', Elmo_behav_conf_output_filename)
# np.savez(Elmo_behav_conf_output_filename, Results=Elmo_behav_conf)
# Save behavior confusion matrix in json
with open(os.path.join(output_path, "Elmo_behavior_confusion_matrix.json"), "w") as outfile:
    json.dump(Elmo_behav_conf, outfile, cls=NumpyArrayEncoder, indent=4)

# Plot confusion matrices
fig, axes = plt.subplots(1, len(test_video_name_list), figsize=(64, 4))
for sample_name, sample_dict in Elmo_behav_dict.items():
    conf_matrix = Elmo_behav_conf[sample_name]
    cur_ax = axes[test_video_name_list.index(sample_name)]
    cur_cax = cur_ax.imshow(conf_matrix, cmap='Blues', vmin=0, vmax=1)
    cur_ax.set_title(f'{sample_name}\n', size=28)
    cur_ax.set_xticks([0, 1, 2], ['grasp', 'touch', 'reach'], fontsize='xx-large')
    cur_ax.set_xlabel('Choice', fontsize='xx-large')
    cur_ax.set_yticks([0, 1, 2], ['grasp', 'touch', 'reach'], fontsize='xx-large')
    cur_ax.set_ylabel('Video', fontsize='xx-large')
# Add color bar
colorbar_ticks = [0, 0.25, 0.5, 0.75, 1]
color_labels = ['0       ', '0.25     ', '0.5      ', '0.75     ', '1       ']
colorBarPosition = [0.8, 0.0, 0.01, 0.85]
fig.subplots_adjust(bottom=0.0, top=0.85, left=0.0, right=0.8,
                    wspace=0.0, hspace=0.3)
cb_ax = fig.add_axes(colorBarPosition)
cbar = fig.colorbar(cur_cax, cax=cb_ax, ticks=colorbar_ticks, orientation='vertical')
cbar.set_ticklabels(color_labels, fontsize='large')
cbar.ax.set_ylabel('Choice frequency', loc='center', rotation=270, fontsize='xx-large')
fig.suptitle('Elmo Confusion Matrix', fontsize=40, x=0.4, y=1.3)
plt.savefig(os.path.join(output_path, 'vis_Elmo_behavior_confusion_matrix.png'), bbox_inches='tight', dpi=100)
plt.close()

# Check behavior bias (binomial test)
p_chance = 1 / len(category_order)
category_pairs = list(product(category_order, repeat=2))
Elmo_behav_binomtest_dict = {}
fig, axes = plt.subplots(1, len(test_video_name_list), figsize=(64, 4))
for sample_name, sample_dict in Elmo_behav_dict.items():
    Elmo_behav_binomtest_dict[sample_name] = {}
    binomtest_3x3 = np.zeros((len(category_pairs),))
    binomtest_sig_3x3 = np.zeros((len(category_pairs),))
    for pairI, pairTuple in enumerate(category_pairs):
        pairName = 'given_{}_choose_{}'.format(pairTuple[0], pairTuple[1])
        # print(pairName)
        category = pairTuple[0]
        choice = pairTuple[1]
        # category_index = category_order.index(category)
        choice_index = Elmo_choice_index_list.index(choice) + 1  # range [1,2,3]
        sample_category_test = Elmo_behav_dict[sample_name]['{}_data'.format(category)]
        sample_category_num_trial = len(sample_category_test)
        sample_category_num_choice = np.count_nonzero(sample_category_test[:, 1] == choice_index)
        # binomial test
        sample_category_binomtest = binomtest(
            sample_category_num_choice, sample_category_num_trial, p_chance, alternative='greater'
        ).pvalue
        # Elmo_behav_binomtest_dict[sample_name][pairName] = sample_category_binomtest
        # print(sample_name, category, 'choose', choice,
        #       'num_trial', sample_category_num_trial,
        #       'num_success', sample_category_num_choice,
        #       'if>p_chance', sample_category_binomtest, sample_category_binomtest < 0.05)
        binomtest_3x3[pairI] = sample_category_binomtest
        binomtest_sig_3x3[pairI] = int(sample_category_binomtest < 0.05)
    binomtest_reshape = binomtest_3x3.reshape(3, 3)
    binomtest_sig_reshape = binomtest_sig_3x3.reshape(3, 3)
    # print(sample_name, 'grasp/touch/reach binom test', binomtest_reshape)
    # print(sample_name, 'sig', binomtest_sig_reshape)
    Elmo_behav_binomtest_dict[sample_name]['binomtest_3x3'] = binomtest_reshape
    cur_ax = axes[test_video_name_list.index(sample_name)]
    cur_cax = cur_ax.imshow(binomtest_reshape, cmap='Blues', vmin=0, vmax=1)
    for rowI in range(binomtest_sig_reshape.shape[0]):
        for colI in range(binomtest_sig_reshape.shape[1]):
            if binomtest_sig_reshape[rowI, colI] == 1:
                cur_ax.text(colI, rowI, '*', horizontalalignment='center', verticalalignment='center',
                            color='red', fontsize=50)
    cur_ax.set_title(f'{sample_name}\n', size=28)
    cur_ax.set_xticks([0, 1, 2], ['grasp', 'touch', 'reach'], fontsize='xx-large')
    cur_ax.set_xlabel('Choice', fontsize='xx-large')
    cur_ax.set_yticks([0, 1, 2], ['grasp', 'touch', 'reach'], fontsize='xx-large')
    cur_ax.set_ylabel('Video', fontsize='xx-large')
# Add color bar
colorbar_ticks = [0, 0.25, 0.5, 0.75, 1]
color_labels = ['0       ', '0.25     ', '0.5      ', '0.75     ', '1       ']
colorBarPosition = [0.8, 0.0, 0.01, 0.85]
fig.subplots_adjust(bottom=0.0, top=0.85, left=0.0, right=0.8,
                    wspace=0.0, hspace=0.3)
cb_ax = fig.add_axes(colorBarPosition)
cbar = fig.colorbar(cur_cax, cax=cb_ax, ticks=colorbar_ticks, orientation='vertical')
cbar.set_ticklabels(color_labels, fontsize='large')
cbar.ax.set_ylabel('Binomial test p-value', loc='center', rotation=270, fontsize='xx-large')
fig.suptitle('Elmo Binomial test: accuracy v.s. chance', fontsize=40, x=0.4, y=1.3)
plt.savefig(os.path.join(output_path, 'vis_Elmo_behavior_bias.png'), bbox_inches='tight', dpi=100)
plt.close()
# Save bias significance in json
with open(os.path.join(output_path, "Elmo_behavior_bias_significance.json"), "w") as outfile:
    json.dump(Elmo_behav_binomtest_dict, outfile, cls=NumpyArrayEncoder, indent=4)


# Correction for behavior bias
# Bias toward touch in touch and reach tests
# Adjustment: label half of touch choices as reach in both tests
# Prerequisites for concave ROC curve:
#   (1) number of touch choices in reach test < number of total reach trials
#   (2) number of touch choices in touch test > 1/2 * number of total touch trials
# Result of adjustment:
#   AUC formula:
#     TPR = TP/(TP+FN)
#     FPR = FP/(FP+TN)
#     AUC = area below all sample ROC curve (TPR-FPR curve)
#   For individual samples:
#     x = number of touch choices in reach test adjusted to reach = the same number adjusted in touch test
#     Assume equal number of trials (n) in grasp/touch/reach, TP + FN = n, TN + FP = 2n
#     tp_touch = number of true positive choices in touch test
#     tp_reach = number of true positive choices in reach test
#     Prerequisite: tp_reach < x < n/2 < tp_touch
#     Result of adjustment in tp:
#       tp_touch_adjusted = tp_touch - x
#       tp_reach_adjusted = tp_reach + x
#  Result of adjustment in AUC (touch v.s. rest):
#    TPR_touch_adjusted = TPR_touch - x/n < TPR_touch - 0.5
#    FPR_touch_adjusted = FPR_touch - x/2n < FPR_touch - 0.25
#    More adjustment = more reduction in TPR_touch and FPR_touch = smaller AUC_touch
#  Result of adjustment in AUC (reach v.s. rest):
#     TPR_reach_adjusted = TPR_reach + x/n < TPR_reach + 0.5
#     FPR_reach_adjusted = FPR_reach + x/2n < FPR_reach + 0.25
#     More adjustment = more increase in TPR_reach and FPR_reach = bigger AUC_reach
Elmo_behav_bias_corrected_conf = correct_behav_conf(Elmo_behav_conf, Elmo_test_bias_dict)
Elmo_behav_bias_corrected_stats = get_behav_stats(Elmo_behav_dict, Elmo_test_bias_dict, Elmo_choice_index_list)
# # Save behavior conf bias corrected in json
# with open(os.path.join(output_path, "Elmo_behavior_stats_bias_corrected.json"), "w") as outfile:
#     json.dump(Elmo_behav_bias_corrected_stats, outfile, cls=NumpyArrayEncoder, indent=4)
# with open(os.path.join(output_path, "Elmo_confustion_matrix_bias_corrected.json"), "w") as outfile:
#     json.dump(Elmo_behav_bias_corrected_conf, outfile, cls=NumpyArrayEncoder, indent=4)

# Compute Youden index = TPR - FPR
Elmo_grasp_vs_rest_FPR = []
Elmo_touch_vs_rest_FPR = []
Elmo_reach_vs_rest_FPR = []
Elmo_grasp_vs_rest_TPR = []
Elmo_touch_vs_rest_TPR = []
Elmo_reach_vs_rest_TPR = []
for sample_name, sample_data in Elmo_behav_bias_corrected_stats.items():
    Elmo_grasp_vs_rest_FPR.append(sample_data['grasp_vs_rest']['FPR'])
    Elmo_touch_vs_rest_FPR.append(sample_data['touch_vs_rest']['FPR'])
    Elmo_reach_vs_rest_FPR.append(sample_data['reach_vs_rest']['FPR'])
    Elmo_grasp_vs_rest_TPR.append(sample_data['grasp_vs_rest']['TPR'])
    Elmo_touch_vs_rest_TPR.append(sample_data['touch_vs_rest']['TPR'])
    Elmo_reach_vs_rest_TPR.append(sample_data['reach_vs_rest']['TPR'])
Elmo_grasp_index = [Elmo_grasp_vs_rest_TPR[x] - Elmo_grasp_vs_rest_FPR[x] for x in range(len(test_video_name_list))]
Elmo_touch_index = [Elmo_touch_vs_rest_TPR[x] - Elmo_touch_vs_rest_FPR[x] for x in range(len(test_video_name_list))]
Elmo_reach_index = [Elmo_reach_vs_rest_TPR[x] - Elmo_reach_vs_rest_FPR[x] for x in range(len(test_video_name_list))]
# # Save behavior index in json
# with open(os.path.join(output_path, "Elmo_behavior_index.json"), "w") as outfile:
#     json.dump({
#         'grasp_Youden_index': Elmo_grasp_index, 
#         'touch_Youden_index': Elmo_touch_index,
#         'reach_Youden_index': Elmo_reach_index,
#     }, outfile, indent=4)

#######################################################
# Compute pairwise Youden index difference (dist map) #
# Note that the difference here is sum of two Youden index such that higher distance means both categories 
# can be classified well by the subject, and lower distance means there is confusion among the two categories
Elmo_behavior_index_dist_matrix = {}
Elmo_behavior_index_dist_vector = []
for videoI, video_name in enumerate(list(Elmo_behav_bias_corrected_stats.keys())):
    tmp = np.zeros((3, 3))
    tmp[0, 1] = abs(Elmo_grasp_index[videoI] + Elmo_touch_index[videoI])
    tmp[0, 2] = abs(Elmo_grasp_index[videoI] + Elmo_reach_index[videoI])
    tmp[1, 0] = abs(Elmo_touch_index[videoI] + Elmo_grasp_index[videoI])
    tmp[1, 2] = abs(Elmo_touch_index[videoI] + Elmo_reach_index[videoI])
    tmp[2, 0] = abs(Elmo_reach_index[videoI] + Elmo_grasp_index[videoI])
    tmp[2, 1] = abs(Elmo_reach_index[videoI] + Elmo_touch_index[videoI])
    Elmo_behavior_index_dist_matrix[video_name] = tmp
    Elmo_behavior_index_dist_vector.append(tmp[0,1])
    Elmo_behavior_index_dist_vector.append(tmp[0,2])
    Elmo_behavior_index_dist_vector.append(tmp[1,2])
    print('Elmo Youden index dist map:', video_name, tmp)
# Elmo_behav_dist_map_output_filename = os.path.join(output_path, 'Elmo_behavior_distance_map.npz')
# print('------ Elmo behavior distance map saved to:', Elmo_behav_dist_map_output_filename)
# np.savez(Elmo_behav_dist_map_output_filename, Results=Elmo_behavior_index_dist_matrix)
# Save behavior dist map in json
# Elmo_behav_dist_map_output_filename = json.dumps(Elmo_behavior_index_dist_matrix, indent = 4)
Elmo_distmap_tosave = copy.deepcopy(Elmo_behavior_index_dist_matrix)
Elmo_distmap_tosave['index_vector_video_order'] = test_video_name_list
Elmo_distmap_tosave['index_vector'] = Elmo_behavior_index_dist_vector
with open(os.path.join(output_path, "Elmo_behavior_index_distmap.json"), "w") as outfile:
    json.dump(Elmo_distmap_tosave, outfile, cls=NumpyArrayEncoder, indent=4)

# Plot behavior Youden index dist map
fig, axes = plt.subplots(1, len(Elmo_behav_bias_corrected_stats.keys()), figsize=(64, 4))
for sample_name, diff_data in Elmo_behavior_index_dist_matrix.items():
    cur_ax = axes[list(Elmo_behav_bias_corrected_stats.keys()).index(sample_name)]
    cur_cax = cur_ax.imshow(diff_data, cmap='Blues', vmin=0, vmax=1)
    cur_ax.set_title(f'{sample_name}\n', size=28)
    cur_ax.set_xticks([0, 1, 2], ['grasp', 'touch', 'reach'], fontsize='xx-large')
    cur_ax.set_yticks([0, 1, 2], ['grasp', 'touch', 'reach'], fontsize='xx-large')
# Add color bar
colorbar_ticks = [0, 0.5, 1, 1.5, 2]
color_labels = ['0       ', '0.5     ', '1      ', '1.5     ', '2       ']
colorBarPosition = [0.8, 0.0, 0.01, 0.85]
fig.subplots_adjust(bottom=0.0, top=0.85, left=0.0, right=0.8,
                    wspace=0.0, hspace=0.3)
cb_ax = fig.add_axes(colorBarPosition)
cbar = fig.colorbar(cur_cax, cax=cb_ax, ticks=colorbar_ticks, orientation='vertical')
cbar.set_ticklabels(color_labels, fontsize='large')
cbar.ax.set_ylabel('Sum of Youden index [0,2]', loc='center', rotation=270, fontsize='xx-large')
fig.suptitle('Elmo Behavior Distance Map (sum of Youden index [TPR-FPR])', fontsize=40, x=0.4, y=1.3)
plt.savefig(os.path.join(output_path, 'vis_Elmo_behavior_index_distmap.png'),  bbox_inches='tight', dpi=100)
plt.close()



##########
# Compute group mean Youden index distance map
Group_behavior_index_dist_matrix = {}
for video_name in list(Louis_behavior_index_dist_matrix.keys()):
    Louis_matrix = Louis_behavior_index_dist_matrix[video_name]
    Elmo_matrix = Elmo_behavior_index_dist_matrix[video_name]
    Group_matrix = (Louis_matrix + Elmo_matrix) / 2
    Group_behavior_index_dist_matrix[video_name] = Group_matrix
Group_behavior_index_dist_vector = [(Louis_behavior_index_dist_vector[x] + Elmo_behavior_index_dist_vector[x]) / 2 for x in range(len(Louis_behavior_index_dist_vector))]
Group_behavior_index_dist_matrix['index_vector_video_order'] = test_video_name_list
Group_behavior_index_dist_matrix['index_vector'] = Group_behavior_index_dist_vector
# Save index distmap in json
with open(os.path.join(output_path, "Group_behavior_index_distmap.json"), "w") as outfile:
    json.dump(Group_behavior_index_dist_matrix, outfile, cls=NumpyArrayEncoder, indent=4)



