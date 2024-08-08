# Functions to visualize video/image features.
# @Qiuhan Jin, 15/08/2023


from utils_feature_extraction import *
from utils_flow_viz import *
import json
from json import JSONEncoder


class NumpyArrayEncoder(JSONEncoder):
    """ Special json encoder for numpy types.
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def visualize_segmentation(image_dict, segmentation_dict, segment_key=None, output_folder=None, output_prefix=None):
    """ Visualize segmentation contour, center, size on raw image.
        If segment_key is given, the segmentation_dict should be a dict of dicts, segment_key is combined with '_pred_masks' to get segmentation masks.
            Otherwise, the segmentation_dict should be a dict of segmentation masks.
        If output_folder is given, save each visualization images in the name of f'{output_prefixe}_frame_{}.png'.
    """
    assert isinstance(image_dict, dict), 'Input should be a dict of images.'
    assert isinstance(segmentation_dict, dict), 'Input should be a dict of segmentation results.'
    image_height, image_width = image_dict[list(image_dict.keys())[0]].shape[:2]
    vis_segmentation_dict = {}
    for frame_index in sorted(list(segmentation_dict.keys())):
        image = image_dict[frame_index]
        if segment_key is None:
            image_data = copy.deepcopy(image)
            seg_mask = segmentation_dict[frame_index]
            seg_center, seg_area, seg_contours, _, _ = get_mask_info(seg_mask, draw_contour=False)
            # Draw contour
            cv2.drawContours(image_data, seg_contours, -1, (0, 255, 0), 1)  # green
            # Draw center
            cv2.circle(image_data, tuple(seg_center.astype('int')), 2, (0, 0, 255), -1)  # red
            # Draw size
            annot_x = int(image_width * 0.1)
            annot_y = int(image_height * 0.1)
            cv2.putText(image_data, 'Size = {:.2f}'.format(seg_area), (annot_x, annot_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        else:
            seg_masks = segmentation_dict[frame_index][f'{segment_key}_pred_masks']
            if len(seg_masks) > 0:
                image_data = copy.deepcopy(image)
                for mask_index in range(len(seg_masks)):
                    seg_mask = seg_masks[mask_index].astype('uint8')
                    seg_center, seg_area, seg_contours, _, _ = get_mask_info(seg_mask, draw_contour=False)
                    # Draw contour
                    cv2.drawContours(image_data, seg_contours, -1, (0, 255, 0), 1)  # green
                    # Draw center
                    cv2.circle(image_data, tuple(seg_center.astype('int')), 2, (0, 0, 255), -1)  # red
                    # Draw size
                    annot_x = int(image_width * 0.1 + 30*mask_index)
                    annot_y = int(image_height * 0.1 + 30*mask_index)
                    cv2.putText(image_data, 'Size = {:.2f}'.format(seg_area), (annot_x, annot_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            else:
                image_data = image
        vis_segmentation_dict[frame_index] = image_data
        if output_folder is not None:
            if output_prefix is None:
                output_prefix = 'vis_segmentation'
            output_image_name = f'{output_prefix}_frame_{frame_index}.png'
            cv2.imwrite(os.path.join(output_folder, output_image_name), image_data)
    return vis_segmentation_dict


def visualize_flow(flow_dict, flow_threshold=0.1, output_folder=None, output_prefix=None):
    """ Visualize optical flow image on the wheel color scheme.
        If output_folder is given, save each visualization images in the name of f'{output_prefixe}_frame_{}.png'.
    """
    assert isinstance(flow_dict, dict), 'Input should be a dict of optical flow arrays.'
    flow_image_dict = {}
    for frame_index, flow_array in flow_dict.items():
        for flow_rowI in range(flow_array.shape[0]):
            for flow_colI in range(flow_array.shape[1]):
                flow_u = flow_array[flow_rowI, flow_colI, 0]
                flow_v = flow_array[flow_rowI, flow_colI, 1]
                if np.absolute(flow_u) < flow_threshold:
                    flow_array[flow_rowI, flow_colI, 0] = 0
                if np.absolute(flow_v) < flow_threshold:
                    flow_array[flow_rowI, flow_colI, 1] = 0
        # visualize flow
        flow_image = flow_to_image(flow_array)
        flow_image_dict[frame_index] = flow_image
        if output_folder is not None:
            if output_prefix is None:
                output_prefix = 'vis_flow'
            viz_fn = os.path.join(output_folder, f'{output_prefix}_frame_{frame_index}.png')
            cv2.imwrite(viz_fn, flow_image[:, :, [2,1,0]])
    return flow_image_dict


def visualize_features(image_dict, feature_dict, feature_list, output_folder, output_prefix=None):
    """ Visualize features on raw image.
        If output_image_folder is given, save each visualization images in the name of f'{output_image_prefixe}_frame_{}.png'.
        Supported features: 'Sobel-gradients', 'Canny-edges', 'FAST-corners', 'HOG-image', 'Gabor-mean-response', 
                            'flow', 'HOF-image', 'MHI', 'shape', 'mask-motion', 'mask-trajectory'.
        Note that feature_list may not contain all features in feature_dict, i.e. 'mask-motion', 'mask-trajectory' requires
            'mask-shape', 'center', 'motion-degree', 'velocity' in feature_dict.
    """
    assert len(feature_list) > 0, 'feature_list should not be empty.'
    assert isinstance(feature_list, list) and isinstance(feature_list[0], str), 'feature_list should be a list of strings.'
    assert not all(['feature-' in x for x in feature_dict.keys()]), 'feature_dict keys should only contain feature name.'
    for feature_name in feature_list:
        output_feature_folder = os.path.join(output_folder, feature_name)
        os.makedirs(output_feature_folder, exist_ok=True)
        if output_prefix is None:
            output_prefix = f'feature-{feature_name}'
        else:
            output_prefix = f'{output_prefix}_feature-{feature_name}'
        if feature_name in ['shape', 'HOG-image', 'HOF-image', 'Canny-edges', 'Gabor-mean-response', 'MHI']:
            # Feature_array can be directly plotted
            for frame_index, feature_array in feature_dict[feature_name].items():
                output_image_name = f'{output_prefix}_frame-{frame_index}.png'
                cv2.imwrite(os.path.join(output_feature_folder, output_image_name), feature_array)
        elif feature_name == 'Sobel-gradients':
            # Visualize Sobel gradients as squared sum of gradients in x and y axis
            for frame_index, feature_array in feature_dict[feature_name].items():
                output_image_name = f'{output_prefix}_frame-{frame_index}.png'
                output_image = np.sqrt(np.sum(feature_array**2, axis=2))
                cv2.imwrite(os.path.join(output_feature_folder, output_image_name), output_image)
        elif feature_name == 'FAST-corners':
            # Draw FAST corners as red dots on image
            for frame_index, image in image_dict.items():
                output_image_name = f'{output_prefix}_frame-{frame_index}.png'
                output_image = copy.deepcopy(image)
                for corner in feature_dict[feature_name][frame_index]:
                    cv2.circle(output_image, tuple(corner), 3, (0, 0, 255), -1)
                cv2.imwrite(os.path.join(output_feature_folder, output_image_name), output_image)
        elif feature_name == 'flow':
            # Visualize optical flow image using vis_flow.py
            flow_image_dict = visualize_flow(feature_dict[feature_name], flow_threshold=0.1, output_folder=None, output_prefix=None)
            for frame_index, flow_image in flow_image_dict.items():
                output_image_name = f'{output_prefix}_frame-{frame_index}.png'
                cv2.imwrite(os.path.join(output_feature_folder, output_image_name), flow_image)
        elif feature_name == 'mask-motion':
            # Visualize center as a black dot, motion degree as a black arrow, velocity as the length of the arrow
            #     and mask shape as a half-transparent polygon
            for frame_index, image in image_dict.items():
                output_image_name = f'{output_prefix}_frame-{frame_index}.png'
                output_image = np.zeros(image.shape, dtype='uint8')  # convert to BGR
                output_image = cv2.cvtColor(output_image, cv2.COLOR_GRAY2BGR)
                mask_shape = feature_dict['shape'][frame_index]
                mask_shape = cv2.cvtColor(mask_shape, cv2.COLOR_GRAY2BGR)
                mask_shape = (mask_shape * 0.5).astype('uint8')
                output_image = cv2.addWeighted(output_image, 1, mask_shape, 1, 0)
                ## Draw center
                mask_center = feature_dict['center'][frame_index]
                cv2.circle(output_image, tuple(mask_center.astype('int')), 2, (0, 0, 0), -1)
                ## Draw motion degree as arrow and velocity as arrow length
                motion_deg = feature_dict['motion-degree'][frame_index]
                velocity = feature_dict['velocity'][frame_index]
                if velocity > 0:
                    arrow_length = int(velocity / max(feature_dict['velocity'].values()) * 10) # from 0 to 10 pixels
                    arrow_end = (int(mask_center[0] + arrow_length * np.cos(motion_deg)), int(mask_center[1] + arrow_length * np.sin(motion_deg)))
                    cv2.arrowedLine(output_image, tuple(mask_center.astype('int')), arrow_end, (0, 0, 0), 2)
                cv2.imwrite(os.path.join(output_feature_folder, output_image_name), output_image)
        elif feature_name == 'mask-trajectory':
            # Visualize all mask centers as black dots, and first and last mask shapes as half-transparent polygons.
            output_image_name = f'{output_prefix}.png'
            output_image = np.zeros(image_dict[0].shape, dtype='uint8')  # convert to BGR
            output_image = cv2.cvtColor(output_image, cv2.COLOR_GRAY2BGR)
            for frame_index, image in image_dict.items():
                ## Draw first and last mask shapes
                if frame_index in [min(feature_dict['shape'].keys()), max(feature_dict['shape'].keys())]:
                    mask_shape = feature_dict['shape'][frame_index]
                    mask_shape = cv2.cvtColor(mask_shape, cv2.COLOR_GRAY2BGR)
                    mask_shape = (mask_shape * 0.5).astype('uint8')
                    output_image = cv2.addWeighted(output_image, 1, mask_shape, 1, 0)
                ## Draw mask centers
                mask_center = feature_dict['center'][frame_index]
                cv2.circle(output_image, tuple(mask_center.astype('int')), 2, (0, 0, 0), -1)
            cv2.imwrite(os.path.join(output_feature_folder, output_image_name), output_image)
        else:
            raise ValueError(f'Invalid feature name to visualize: {feature_name}')


def generate_video_from_images(input, output_folder, output_name):
    """ Generate a video from a list of images. 
        image_input can be a string of image folder, a list of images, or a dict of images.
    """
    if isinstance(input, str):
        image_list = sorted([os.path.join(input, f) for f in os.scandir(input) if f.name.endswith(('.png', '.jpg'))])
    elif isinstance(input, list):
        image_list = input
    elif isinstance(input, dict):
        image_list = []
        for frame_index in sorted(list(input.keys())):
            image_list.append(input[frame_index])
    # Write image_list to a video
    output_video_path = os.path.join(output_folder, f'{output_name}.mp4')
    output_video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (image_list[0].shape[1], image_list[0].shape[0]))
    for image in image_list:
        output_video.write(image)
    output_video.release()


def find_union_segmentation(list_of_image_dict, list_of_mask_dict, visualize=False, stack_type='h', output_folder=None, output_prefix=None, mask_label=None):
    """ For each frame across videos, find the union of segmentation masks.
        This function is used to find a spatial bounding box to filter image frames.
        All videos should have the same number of frames.
        Note that this step is often done after find_union_motion_clip().

        If output_folder is given, save one spatial_window_dict for all videos in json file.
        If visualize == True, also visualize the union bbox on each video and stack videos in one.
    """
    spatial_window_dict = {}
    num_frames = len(list_of_mask_dict[list(list_of_mask_dict.keys())[0]])
    for frame_i in range(num_frames):
        seg_mask_list = []
        for video_name in list(list_of_mask_dict.keys()):
            frame_index = sorted(list(list_of_mask_dict[video_name].keys()))[frame_i]
            seg_mask = np.asarray(list_of_mask_dict[video_name][frame_index], dtype=np.uint8)
            seg_mask_list.append(seg_mask)
        if len(seg_mask_list) > 0:
            union_mask = np.zeros_like(seg_mask_list[0])
            for seg_mask in seg_mask_list:
                union_mask = np.logical_or(union_mask, seg_mask)
            union_mask = union_mask.astype('uint8')
            spatial_window_dict[frame_i] = union_mask
        else:
            spatial_window_dict[frame_i] = np.zeros_like(list_of_image_dict[video_name][frame_index])
    if output_folder is not None:
        if output_prefix is None:
            output_prefix = 'PAO_'
        else:
            output_prefix = output_prefix + '_'
        if mask_label is None:
            mask_label = ''
        else:
            mask_label += '-'
        output_file_name = f'{output_prefix}rescaled-{mask_label}seg-union-mask.npz'
        output_file_path = os.path.join(output_folder, output_file_name)
        np.savez(output_file_path, Results=spatial_window_dict)
    # Visualize
    if visualize and output_folder is not None:
        list_of_vis_image_dict = {}
        list_of_frames_dict = {}
        if len(list_of_image_dict) > 5:
            num_keys = len(list_of_image_dict)
            one_third_num_keys = int(num_keys / 3)
            stacked_video_indices = [1, one_third_num_keys, 2 * one_third_num_keys]
        else:
            stacked_video_indices = list(range(len(list_of_image_dict)))
        # for video_name, video_image_dict in list_of_image_dict.items():
        for video_index in stacked_video_indices:
            video_name = list(list_of_image_dict.keys())[video_index]
            video_image_dict = list_of_image_dict[video_name]
            vis_image_dict = visualize_segmentation(video_image_dict, spatial_window_dict, segment_key=None, output_folder=None, output_prefix=None)
            list_of_vis_image_dict[video_name] = vis_image_dict
            list_of_frames_dict[video_name] = sorted(list(vis_image_dict.keys()))
            num_frames = len(list_of_frames_dict[video_name])
        ## Stack videos
        stacked_image_list = []
        if stack_type == 'h':
            for frame_i in range(num_frames):
                stacked_images = []
                spatial_window = spatial_window_dict[frame_i]
                for video_name, vis_image_dict in list_of_vis_image_dict.items():
                    frame_list = list_of_frames_dict[video_name]
                    vis_image = vis_image_dict[frame_list[frame_i]]
                    ### Draw bbox
                    (y1, y2, x1, x2) = crop_content_from_image(vis_image, mask=spatial_window, output_crop_bbox=True)
                    cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    stacked_images.append(vis_image)
                ### hstack
                stacked_image = np.hstack(stacked_images)
                stacked_image_list.append(stacked_image)
        elif stack_type == 'v':
            for frame_i in range(num_frames):
                stacked_images = []
                spatial_window = spatial_window_dict[frame_i]
                for video_name, vis_image_dict in list_of_vis_image_dict.items():
                    frame_list = list_of_frames_dict[video_name]
                    vis_image = vis_image_dict[frame_list[frame_i]]
                    ### Draw bbox
                    (y1, y2, x1, x2) = crop_content_from_image(vis_image, mask=spatial_window, output_crop_bbox=True)
                    cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    stacked_images.append(vis_image)
                ### vstack
                stacked_image = np.vstack(stacked_images)
                stacked_image_list.append(stacked_image)
        else:
            raise ValueError(f'Invalid stack_type: {stack_type}')
            # num_rows, num_cols = stack_type
            # for frame_i in range(num_frames):
            #     row_image_list = []
            #     spatial_window = spatial_window_dict[frame_i]
            #     for row_index in range(num_rows):
            #         col_image_list = []
            #         for col_index in range(num_cols):
            #             video_index = row_index * num_cols + col_index
            #             if video_index >= len(list_of_vis_image_dict):
            #                 break
            #             frame_list = list_of_frames_dict[video_name]
            #             vis_image = list_of_vis_image_dict[video_name][frame_list[frame_i]]
            #             ### Draw bbox
            #             (y1, y2, x1, x2) = crop_content_from_image(vis_image, mask=spatial_window, output_crop_bbox=True)
            #             cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
            #             col_image_list.append(vis_image)
            #         row_image = np.hstack(col_image_list)
            #         row_image_list.append(row_image)
            #     stacked_image = np.vstack(row_image_list)
            #     stacked_image_list.append(stacked_image)
        # Write videos
        output_video_name = f'{output_prefix}rescaled-{mask_label}seg-union-mask.mp4'
        output_video_path = os.path.join(output_folder, output_video_name)
        output_video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (stacked_image.shape[1], stacked_image.shape[0]))
        for frame_index in range(len(stacked_image_list)):
            output_video.write(stacked_image_list[frame_index])
        output_video.release()
    return spatial_window_dict


def find_union_motion_clip(list_of_image_dict, list_of_velocity_dict, duration, threshold, method='motion_end', full_length=False, 
                           visualize=False, stack_type='h', output_folder=None, output_prefix=None, mask_label=None):
    """ Realign frames of videos in list_of_image_dict (key: video name) by comparing one of the following from list_of_velocity_dict:
        (1) 'velocity', list_of_velocity_dict is the velocity of a segmentation mask,
            return a list of frame indices in which the segmentation velocity has passed the threshold,
            and consecutively last for a duration of frames.
        (2) 'flow', same method as (1) except list_of_velocity_dict is a dict of optical flow, and threshold for flow magnitude.
        
        If method == 'motion_end', realign videos such that the motion end frame is the same for all videos.
        If method == 'motion_start', realign videos such that the motion start frame is the same for all videos.
        If full_length == True, the list of frame indices will be extended to the possible maximum length for all videos,
        otherwise, only the clip of frames which all satisfy the threshold will be returned.

        If output_folder is given, save one temporal_window_dict for each video in json file.
        If visualize == True, also save the realigned videos.
            If visualize_type == 'h', stack frames of each video horizontally;
            If visualize_type == 'v', stack frames of each video vertically;
            If visualize_type is a tuple (num_rows, num_cols), stack frames by num_rows x num_cols.
    """
    num_frames_list = []
    list_of_frames_dict = {}
    for video_name, video_frames in list_of_image_dict.items():
        num_frames_list.append(len(video_frames))
        list_of_frames_dict[video_name] = sorted(list(video_frames.keys()))
    min_num_frames = min(num_frames_list)

    # Find motion start and end frames for all videos
    motion_start_frame_dict = {}
    motion_end_frame_dict = {}
    motion_start_frame_list = []
    motion_end_frame_list = []
    motion_duration_list = []
    for video_name, velocity_dict in list_of_velocity_dict.items():
        frame_list = sorted(list(velocity_dict.keys()))
        # Finds when motion starts
        motion_start_frame = 0
        for frame_i in frame_list:
            duration_info_list = []
            if frame_i - duration >= frame_list[0]:
                for frame_j in range(frame_i - duration, frame_i):
                    velocity = velocity_dict[frame_j]
                    duration_info_list.append(velocity)
            else:
                for frame_j in range(frame_list[0], frame_i):
                    velocity = velocity_dict[frame_j]
                    duration_info_list.append(velocity)
            if len(duration_info_list) > 0:
                if np.all(np.asarray(duration_info_list) < threshold) and velocity_dict[frame_i] >= threshold:
                    # frame_i is the first frame that satisfies the threshold
                    motion_start_frame = frame_i
                    break
        motion_start_frame_dict[video_name] = motion_start_frame
        motion_start_frame_list.append(motion_start_frame)
        # Finds when motion ends
        motion_end_frame = frame_list[-1]
        if motion_start_frame > 0:
            for frame_m in range(motion_start_frame, frame_list[-1] + 1):
                duration_info_list = []
                if frame_m + duration <= frame_list[-1] + 1:
                    for frame_n in range(frame_m + 1, frame_m + 1 + duration):
                        velocity = velocity_dict[frame_n]
                        duration_info_list.append(velocity)
                else:
                    for frame_n in range(frame_m + 1, frame_list[-1] + 1):
                        velocity = velocity_dict[frame_n]
                        duration_info_list.append(velocity)
                if len(duration_info_list) > 0:
                    if np.all(np.asarray(duration_info_list) < threshold) and velocity_dict[frame_m] >= threshold:
                        # frame_m is the first frame that satisfies the threshold
                        motion_end_frame = frame_m
                        break
        motion_end_frame_dict[video_name] = motion_end_frame
        motion_end_frame_list.append(motion_end_frame)
        # Append motion duration
        motion_duration_list.append(motion_end_frame - motion_start_frame)
    
    # Find the longest/shortest motion clip
    list_of_temporal_window_dict = {}
    if method == 'motion_end':
        if full_length:
            num_frames_to_earliest_motion_end = min(motion_end_frame_list)
            num_frames_from_latest_motion_end = min_num_frames - max(motion_end_frame_list)
            if num_frames_from_latest_motion_end < 0:
                raise ValueError('Cannot find a temporal window that satisfies the threshold, check if all videos have enough frames.')
            # motion_duration = num_frames_to_earliest_motion_end + num_frames_from_latest_motion_end
            ## Temporal window lasts for motion_duration, count from motion_end_frame, 
            ## before num_frames_to_earliest_motion_end frames, and after num_frames_from_latest_motion_end frames.
            for video_name, motion_end_frame in motion_end_frame_dict.items():
                temporal_window_end = motion_end_frame + num_frames_from_latest_motion_end
                temporal_window_start = motion_end_frame - num_frames_to_earliest_motion_end + 1
                list_of_temporal_window_dict[video_name] = np.arange(temporal_window_start, temporal_window_end)
        else:
            earliest_motion_end_frame = min(motion_end_frame_list)
            longest_motion_duration = max(motion_duration_list)
            while longest_motion_duration >= earliest_motion_end_frame - 1:
                longest_motion_duration -= 1  # mask sure duration is not longer than the earliest motion end
            # temporal window lasts for adjusted longest_motion_duration
            for video_name, motion_end_frame in motion_end_frame_dict.items():
                temporal_window_end = motion_end_frame
                temporal_window_start = motion_end_frame - longest_motion_duration
                list_of_temporal_window_dict[video_name] = np.arange(temporal_window_start, temporal_window_end + 1)
    elif method == 'motion_start':
        if full_length:
            raise ValueError('motion_start full_length not implemented yet.')
        else:
            latest_motion_start_frame = max(motion_start_frame_list)
            longest_motion_duration = max(motion_duration_list)
            while longest_motion_duration >= min_num_frames - latest_motion_start_frame:
                longest_motion_duration -= 1
            # temporal window lasts for adjusted longest_motion_duration
            for video_name, motion_start_frame in motion_start_frame_dict.items():
                temporal_window_start = motion_start_frame
                temporal_window_end = motion_start_frame + longest_motion_duration
                list_of_temporal_window_dict[video_name] = np.arange(temporal_window_start, temporal_window_end + 1)
    else:
        raise ValueError(f'Invalid method: {method}')
    if output_folder is not None:
        if output_prefix is None:
            output_prefix = 'video-set_'
        else:
            output_prefix = output_prefix + '_'
        if mask_label is None:
            mask_label = ''
        else:
            mask_label += '-'
        output_json_name = f'{output_prefix}{mask_label}temporal-window.json'
        output_json_path = os.path.join(output_folder, output_json_name)
        with open(output_json_path, 'w') as f:
            json.dump(list_of_temporal_window_dict, f, indent=4, cls=NumpyArrayEncoder)
    
    # Visualize realigned videos
    video_name_list = list(list_of_image_dict.keys())
    num_frames = len(list_of_temporal_window_dict[video_name_list[0]])
    if visualize and output_folder is not None:
        stacked_image_list = []
        if stack_type == 'h':
            for frame_i in range(num_frames):
                stacked_image = np.hstack([list_of_image_dict[video_name][list_of_temporal_window_dict[video_name][frame_i]] for video_name in video_name_list])
                stacked_image_list.append(stacked_image)
        elif stack_type == 'v':
            for frame_i in range(num_frames):
                stacked_image = np.vstack([list_of_image_dict[video_name][list_of_temporal_window_dict[video_name][frame_i]] for video_name in video_name_list])
                stacked_image_list.append(stacked_image)
        else:
            num_rows, num_cols = stack_type
            for frame_i in range(num_frames):
                row_image_list = []
                for row_index in range(num_rows):
                    col_image_list = []
                    for col_index in range(num_cols):
                        video_index = row_index * num_cols + col_index
                        if video_index >= len(video_name_list):
                            break
                        video_name = video_name_list[video_index]
                        col_image_list.append(list_of_image_dict[video_name][list_of_temporal_window_dict[video_name][frame_i]])
                    row_image = np.hstack(col_image_list)
                    row_image_list.append(row_image)
                stacked_image = np.vstack(row_image_list)
                stacked_image_list.append(stacked_image)
        output_video_name = f'{output_prefix}{mask_label}temporal-realigned.mp4'
        output_video_path = os.path.join(output_folder, output_video_name)
        output_video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 15, (stacked_image.shape[1], stacked_image.shape[0]))
        for stacked_image in stacked_image_list:
            output_video.write(stacked_image)
        output_video.release()
    return list_of_temporal_window_dict
# Example: Velocity is different for different videos in PAO dataset,
#      grasp/push = hand velocity, 
#      drop = object velocity, 
#      scratch = full video (velocity set to 0 for all frames),
# This function will align grasp/push to when hand motion finishes, 
#     drop when object motion finishes, 
#     and scratch simply count from video end.


def visualize_union_motion_clip(list_of_image_dict, list_of_temporal_window_dict, 
                                stack_type='h', output_folder=None, output_prefix=None, mask_label=None):
    """ Visualize all video frames in list_of_temporal_window_dict stacked together.
    """
    if output_folder is None:
        raise ValueError('Output folder not specified.')
    if output_prefix is None:
        output_prefix = 'video-set_'
    else:
        output_prefix = output_prefix + '_'
    if mask_label is None:
        mask_label = ''
    else:
        mask_label += '-'
    video_name_list = list(list_of_image_dict.keys())
    num_frames = len(list_of_temporal_window_dict[video_name_list[0]])
    stacked_image_list = []
    if stack_type == 'h':
        for frame_i in range(num_frames):
            stacked_image = np.hstack([list_of_image_dict[video_name][list_of_temporal_window_dict[video_name][frame_i]] for video_name in video_name_list])
            stacked_image_list.append(stacked_image)
    elif stack_type == 'v':
        for frame_i in range(num_frames):
            stacked_image = np.vstack([list_of_image_dict[video_name][list_of_temporal_window_dict[video_name][frame_i]] for video_name in video_name_list])
            stacked_image_list.append(stacked_image)
    else:
        num_rows, num_cols = stack_type
        for frame_i in range(num_frames):
            row_image_list = []
            for row_index in range(num_rows):
                col_image_list = []
                for col_index in range(num_cols):
                    video_index = row_index * num_cols + col_index
                    if video_index >= len(video_name_list):
                        break
                    video_name = video_name_list[video_index]
                    col_image_list.append(list_of_image_dict[video_name][list_of_temporal_window_dict[video_name][frame_i]])
                row_image = np.hstack(col_image_list)
                row_image_list.append(row_image)
            stacked_image = np.vstack(row_image_list)
            stacked_image_list.append(stacked_image)
    output_video_name = f'{output_prefix}{mask_label}temporal-realigned.mp4'
    output_video_path = os.path.join(output_folder, output_video_name)
    output_video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 15, (stacked_image.shape[1], stacked_image.shape[0]))
    for stacked_image in stacked_image_list:
        output_video.write(stacked_image)
    output_video.release()


