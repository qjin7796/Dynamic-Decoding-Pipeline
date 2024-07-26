from utils_keops_chamfer_distance import *


# Define data path and batch size
data_input_folder = '/data/leuven/347/vsc34741/code/chd/Tail'
batch_size = 1
save_log_file = True
test_mode_on = False
# Videos
video_set_name_list = ['green_C_left_no', 'small_C_left_no', 'ring_C_left_no', 'big_C_left_yes', 'big_C_right_no', 'big_L_left_no', 'big_T_left_no', 'monkey_hand', 'monkey_tail']
video_name_list = ['grasp', 'touch', 'reach']
seg_name_list = ['arm-hand', 'arm', 'hand', 'body', 'body-arm-hand', 'body-arm']
dim_name_list = ['zero-excl', 'zero-incl']
# Features
local_feature_list = [
    'Sobel-gradients', 'Canny-edges', 'HOG-array', 'Gabor-mean-response', 
    'flow', 'motion-energy', 'HOF-array', 'MHI', 'shape', 'shape-moments', 'center', 'size', 'motion-degree', 'velocity'
]
local_feature_list_2 = ['Sobel-gradients', 'Gabor-mean-response', 'flow', 'motion-energy', 'MHI']
for dim_name in dim_name_list:
    data_output_folder = os.path.join(data_input_folder, f'output-{dim_name}')
    os.makedirs(data_output_folder, exist_ok=True)
    for seg_name in seg_name_list:
        ## Load point set dict
        if dim_name == 'zero-incl':
            feature_list = local_feature_list_2
        else:
            feature_list = local_feature_list
        point_set_file = os.path.join(data_input_folder, f'all-test-videos_feature-{seg_name}-{dim_name}_point-set.pickle')
        with open(point_set_file, 'rb') as f:
            point_set_dict = pickle.load(f)
        output_folder = os.path.join(data_output_folder, f'feature-{seg_name}-{dim_name}')
        os.makedirs(output_folder, exist_ok=True)
        log_file = os.path.join(output_folder, 'log.txt')
        ## Loop through videos
        for video_set_name in video_set_name_list:
            video_set_point_set = point_set_dict[video_set_name]
            point_set_chd_list_results = compute_batch_chamfer_distance(
                video_set_point_set, video_name_list, feature_list, batch_size, save_log_file, log_file, video_set_name, test_mode_on
            )
            ### Save json
            video_set_chd_json_file = os.path.join(output_folder, f'{video_set_name}_feature-{seg_name}-{dim_name}_chamfer-distance.json')
            with open(video_set_chd_json_file, 'w') as f:
                json.dump(point_set_chd_list_results, f, cls=NumpyArrayEncoder, indent=4)

