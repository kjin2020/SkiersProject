import sys
import os
# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)  # Use insert(0, ...) to prioritize it

from utils.load_data import load_json
from utils.dtw import compare_selected_cycles, extract_multivariate_series
from utils.feedback_utils import *
from utils.nets import LSTMNet, SimpleMLP
from utils.config import update_config
from utils.split_cycles import split_into_cycles
from utils.preprocess_signals import *
from utils.annotation_format import halpe26_to_coco
from alphapose.scripts.demo_inference import run_inference
from utils.feedback_utils import get_line_points, feedback_wide_legs, feedback_stiff_ankle
from utils.classify_angle import classify_angle
from utils.frame_extraction import get_image_by_id, extract_frame, reencode_to_all_keyframes_temp
from collections import defaultdict

import torch
import numpy as np
import cv2
import shutil
import time

# test comment


# Video ID
ID = "92"
# needed for evaluation of mistake
SKIER_ID = 12
# This is only needed when inference should not be run
INPUT_PATH = os.path.join(r"C:\awilde\britta\LTU\SkiingProject\SkiersProject\Data\Annotations\annotations_finetuned_v1\Mixed_level",  f"{ID}.json")
testing_with_inference = True
# if only the results should be evaluated this can be set to false
show_feedback = True



def main():
    
    # check and select device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device = {device}")
    
    # Step 1: Get Keypoints from AlphaPose 
    
    print("Loading config...")
    run_args = update_config("./feedback_system/config_feedback_pipe.yaml") # TODO Testing set up fix for full pipeline
    # for evaluation purposes
    evaluation_file = f'{run_args.FEEDBACK.OUTPUT_STATS}/evaluation_{run_args.FEEDBACK.MISTAKE_TYPE}.json'
    if testing_with_inference:
        start = time.time()
        output_path, results_list = run_inference(run_args)
        end = time.time()
        length = end - start
        print(f"Inference took {length} sec")
        
        # Convert keypoint data to coco format
        coco_data, _, _ = halpe26_to_coco(results_list)
        
    else:
        coco_data = load_json(INPUT_PATH)

    video_angle = classify_angle(coco_data)
    print(f"Video angle: {video_angle}")

    # Step 2: Split into cycles
    print("Splitting the data into cycles...")
    cycle_data = split_into_cycles(coco_data, run_args, visualize=False, video_angle=video_angle)

    # Load the checkpoint to the model as we need it for certain data
    print("Loading Model...")
    checkpoint = torch.load(run_args.CLS_GEAR.MODEL_PATH, map_location=device)
    
    # Extract the state_dict and custom parameters
    state_dict = checkpoint["state_dict"]
    custom_params = checkpoint["custom_params"]
    
    # convert to numpy arrays and stack them together
    input_data = []
    for cycle in cycle_data.values():

        # Extract joint data as (num_joints, time_steps)
        cycle_data_array = [np.array(cycle[joint], dtype=np.float32) for joint in custom_params["choosen_joints"]]

        # Stack into a (num_joints, time_steps) tensor
        cycle_tensor = np.stack(cycle_data_array)  # Shape: (num_joints, time_steps)

        input_data.append(cycle_tensor)
    # Step 3: Classify user cycles
    
    # Preprocess data based on train data
    print("Preprocessing data...")
    # padding
    input_data = pad_sequences(input_data, max_length=custom_params["train_max_length"], pad_value=float('nan'))
    # normalization
    if custom_params["normalization"]:
        print("Normalizing the data...")
        if custom_params["norm_type"] == "full_signal":
            input_data = normalize_full_signal(input_data, custom_params["train_mean"], custom_params["train_std"])
        else:
            input_data = normalize_per_timestamp(input_data, custom_params["train_mean"], custom_params["train_std"])
    
    # adjust padding    
    input_data = replace_nan_with_first_value(input_data)
    
    # smoothing
    if custom_params["smoothing"] > 0:
        input_data = gaussian_filter1d(input_data, sigma=custom_params["smoothing"], axis=2)
        
    # convert data to a tensor
    input_data = torch.tensor(input_data, dtype=torch.float32)
    
    # Extract input and output sizes
    input_channels = custom_params["input_channels"]
    output_channels = custom_params["output_channels"]

    # initialize the model
    if run_args.CLS_GEAR.NETTYPE.lower() == "lstm":

        print("Initializing LSTM...")
        net = LSTMNet(input_channels, 
                    custom_params["hidden_size"], 
                    output_channels, 
                    custom_params["num_layers"], 
                    custom_params["dropout"])

    else:  # MLP

        print("Initializing MLP...")
        net = SimpleMLP(input_channels, 
                        custom_params["hidden1"], 
                        custom_params["hidden2"], 
                        output_channels)

    # Load weights into the model
    net.load_state_dict(state_dict)
    net.to(device)
    net.eval()

    print(f"Loaded model with input={input_channels}, output={output_channels}")
    
    # reverse label dict to get predictions
    reversed_labels = {v: k for k, v in custom_params["labels"].items()}

    if run_args.FEEDBACK.SAVE_VIDEO:
        video_writer = 1
    else:
        video_writer = None # skips video writing

    # Create temp quick frame lookup video for user frames
    if show_feedback:
        user_temp_video = reencode_to_all_keyframes_temp(run_args.VIDEO_PATH)

    
    # classify each cycle
    # for evaluation purposes
    summary_feedback = generate_evaluation_dict(run_args.FEEDBACK.MISTAKE_TYPE)
    for i, cycle_input in enumerate(input_data):

        # this is done in the dataloader 
        cycle_input = cycle_input.T # (12, 97) => (97, 12), 12 joints, 97 timesteps
        # add batch size
        cycle_input = cycle_input.unsqueeze(0)
        
        if run_args.CLS_GEAR.NETTYPE.lower() == "mlp":
            cycle_input = cycle_input.contiguous().view(cycle_input.size(0), -1)
        cycle_input = cycle_input.to(device)
        outputs = net(cycle_input)
        outputs = outputs.squeeze(1)
            
        _, predicted = torch.max(outputs, 1)  # Get the class with the highest probability
        predicted_label = reversed_labels.get(predicted.item())
        print(f"Predicted class: {predicted_label}")
        
        # Step 4: Based on classification use DTW
        
        # based on cycle choose the expert data we want to compare the cycle to
        if predicted_label == "gear3":
            expert_path = "./data/expert_data/expert_cycles_gear3_real.json"
        elif predicted_label == "gear2":
            expert_path = "./data/expert_data/expert_cycles_gear2_real.json"
        else:
            print(f"The system cannot give feedback for {predicted_label}")
            continue

        expert_data = load_json(expert_path)
        
        # get mistake that we want to give feedback on
        mistake_type = run_args.FEEDBACK.MISTAKE_TYPE
        
        # Define joint triplets for angle comparisons
        # TODO to compare cycles we can either input joint triplets, then we need to set use_keypoints to false
        # otherwise we can input joints, then it will use raw keypoints for DTW
        joint_triplets = [("RHip", "RKnee", "RAnkle"), ("LHip", "LKnee", "LAnkle"), ("RShoulder", "RElbow", "RWrist"), ("LShoulder", "LElbow", "LWrist")]
        
        # define what joints we want to look at for the matching
        if mistake_type == "wide_legs":
            joints = ["RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle"]
        else:
            joints = ["RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist"]
        

        # send in data in json format
        cycle = cycle_data[f"Cycle {i+1}"]

        dtw_comparisons, path, expert_cycle = compare_selected_cycles(expert_data, cycle, joints, run_args.VIDEO_PATH, run_args.DTW.VIS_VID_PATH, use_keypoints=True, visualize=False)
        print("Chosen expert cycle id", expert_cycle.get("Video"))

        # Step 5: Give feedback
        
        direction = expert_cycle.get("Direction")
        if video_angle == "Front":
            if mistake_type == "wide_legs":
                joint_angles = []
                joints_distance = [("LAnkle", "RAnkle")]
                joints_lines_relative = []
                joints_lines_horizontal = []
            elif mistake_type == "biathlon":
                joint_angles = [("RElbow", "RShoulder", "RHip"), ("LElbow", "LShoulder", "LHip"),("RShoulder", "RElbow", "RWrist"), ("LShoulder", "LElbow", "LWrist")]
                joints_lines_horizontal = [("Hip", "Neck")]
                joints_distance = []
                joints_lines_relative = []
            elif mistake_type == "general":
                joint_angles, joints_distance, joints_lines_relative, joints_lines_horizontal = load_general_mode_feedback_config(run_args)
            else:
                print(f"We cannot give feedback for this mistake {mistake_type} from the front, please provide a video from the side or check the mistake type is supported.")
                break
            
        elif video_angle == "Left":
            
            if mistake_type == "stiff_ankle":
                joints_lines_relative = []
                joint_angles = [("RHip", "RKnee", "RAnkle"), # Right knee angle
                            ]
                joints_lines_horizontal = []
                joints_distance = []
            elif mistake_type == "general":
                joint_angles, joints_distance, joints_lines_relative, joints_lines_horizontal = load_general_mode_feedback_config(run_args)
            else:
                print(f"We cannot give feedback for this mistake {mistake_type} from the side, please provide a video from the front or check the mistake type is supported..")
                break

        elif video_angle == "Right":
            if mistake_type == "stiff_ankle":
                joint_angles = [("LHip", "LKnee", "LAnkle")]
                joints_lines_relative = []
                joints_lines_horizontal = []
                joints_distance = []
            elif mistake_type == "general":
                joint_angles, joints_distance, joints_lines_relative, joints_lines_horizontal = load_general_mode_feedback_config(run_args)
            else:
                print(f"We cannot give feedback for this mistake {mistake_type} from the side, please provide a video from the front or check the mistake type is supported..")
                break
        
        user_lines = []
        expert_lines = []
        user_angles = []
        expert_angles = []
        user_horizontal_lines = []
        expert_horizontal_lines = []
        user_distances = []
        expert_distances = []
        diff_lines_relative = []
        sim_lines_relative = []
        diff_angles = []
        sim_angles = []
        diff_lines_horizontal = []
        sim_lines_horizontal = []
        diff_distances = []
        sim_distances = []


        # Get the lines
        if len(joints_lines_relative):
            user_lines, _ = extract_multivariate_series_for_lines(cycle, joints_lines_relative, run_args)
            expert_lines, _ = extract_multivariate_series_for_lines(expert_cycle, joints_lines_relative, run_args)
            # Match using DTW and calculate difference in angle between the lines
            diff_lines_relative = calculate_differences(user_lines, expert_lines, path)
            sim_lines_relative = calculate_similarity(user_lines, expert_lines, path)

        if len(joint_angles):
            user_angles, _ = extract_multivariate_series(cycle, joint_angles, run_args)
            expert_angles, _ = extract_multivariate_series(expert_cycle, joint_angles, run_args)
             # Match using DTW and calculate difference in angle
            diff_angles = calculate_differences(user_angles, expert_angles, path)
            sim_angles = calculate_similarity(user_angles, expert_angles, path)

        if len(joints_lines_horizontal): 
            user_horizontal_lines, _ = extract_multivariate_series_for_single_lines(cycle, joints_lines_horizontal, run_args)
            expert_horizontal_lines, _= extract_multivariate_series_for_single_lines(expert_cycle, joints_lines_horizontal, run_args)
             # Match using DTW and calculate difference in angle between the lines
            diff_lines_horizontal = calculate_differences(user_horizontal_lines, expert_horizontal_lines, path)
            sim_lines_horizontal = calculate_similarity(user_horizontal_lines, expert_horizontal_lines, path)
            
        if len(joints_distance):
            user_distances, _ = extract_multivariate_series_for_distances(cycle, joints_distance, run_args)
            expert_distances, _ = extract_multivariate_series_for_distances(expert_cycle, joints_distance, run_args)
            
            diff_distances = calculate_differences(user_distances, expert_distances, path)
            sim_distances = calculate_similarity(user_distances, expert_distances, path)

        feedback_range = 3
        if mistake_type == "wide_legs":
            feedback_per_frame, feedback_per_category = feedback_wide_legs(expert_distances, user_distances, diff_distances, path, feedback_range)
            for category, feedbacks in feedback_per_category.items():
                    for sentiment, count in feedbacks.items():
                        summary_feedback["with_self_matches"][predicted_label][category][sentiment] += count
                        summary_feedback["with_self_matches"][predicted_label]["all"][sentiment] += count
            if SKIER_ID != expert_cycle.get("Skier_id"):
                for category, feedbacks in feedback_per_category.items():
                    for sentiment, count in feedbacks.items():
                        summary_feedback["no_self_matches"][predicted_label][category][sentiment] += count
                        summary_feedback["no_self_matches"][predicted_label]["all"][sentiment] += count
                        
        elif mistake_type == "stiff_ankle":
            feedback_per_frame, feedback_per_category = feedback_stiff_ankle(joint_angles, user_angles, expert_angles, path)
            for category, count in feedback_per_category.items():
                    summary_feedback["with_self_matches"][predicted_label][category] += count
            if SKIER_ID != expert_cycle.get("Skier_id"):
                for category, count in feedback_per_category.items():
                    summary_feedback["no_self_matches"][predicted_label][category] += count

        if show_feedback:
            # Create temp quick frame lookup video for expert frames
            expert_start_frame = expert_cycle.get("Start_frame")
            vid_id = int(expert_cycle.get("Video"))
            expert_video = os.path.join(run_args.DTW.VIS_VID_PATH, f"DJI_{vid_id:04d}" + ".mp4")
            # for the expert frame a -1 is added to the index as we get the image ids here and they start at 1 but frames actually 
            # start at 0. For the user id we can get the frame id directly as we have the full json file. 
            temp_dir="./temp_videos"
            expert_temp_video = reencode_to_all_keyframes_temp(expert_video, temp_dir)

            # Loops through the DTW match pair and shows lines on user video
            for i, (frame1, frame2) in enumerate(path):
                # get acual frame
                imgage_id = cycle["Image_ids"][frame1]
                image_video_id = get_image_by_id(coco_data["images"], imgage_id)
                image_video_id = int(image_video_id["file_name"].split('.')[0])
                user_frame = extract_frame(user_temp_video, image_video_id)
                # Get expert frame
                expert_frame = extract_frame(expert_temp_video, frame2 + expert_start_frame - 1)    
                
                # draw lines on to each frame
                user_points_lines = get_line_points(cycle, joints_lines_relative, frame1, run_args)
                expert_points_lines = get_line_points(expert_cycle, joints_lines_relative, frame2, run_args)
                
                user_points_angles = get_line_points(cycle, joint_angles, frame1, run_args)
                expert_points_angles = get_line_points(expert_cycle, joint_angles, frame2, run_args)

                user_points_horizontal_lines = get_line_points(cycle, joints_lines_horizontal, frame1, run_args)
                expert_points_horizontal_lines = get_line_points(expert_cycle, joints_lines_horizontal, frame2, run_args)
                
                user_points_distances = get_line_points(cycle, joints_distance, frame1, run_args)
                expert_points_distances = get_line_points(expert_cycle, joints_distance, frame2, run_args)

                # Draw lines
                draw_joint_relative_lines(joints_lines_relative, user_frame, user_points_lines)
                draw_joint_relative_lines(joints_lines_relative, expert_frame, expert_points_lines)
                draw_joint_angles(joint_angles, user_frame, user_points_angles)
                draw_joint_angles(joint_angles, expert_frame, expert_points_angles)
                draw_joint_single_lines(joints_lines_horizontal, user_frame, user_points_horizontal_lines)
                draw_joint_single_lines(joints_lines_horizontal, expert_frame, expert_points_horizontal_lines)
                draw_joint_single_lines(joints_distance, user_frame, user_points_distances)
                draw_joint_single_lines(joints_distance, expert_frame, expert_points_distances)
                

                height, width, channels = user_frame.shape
                empty_image = np.zeros((height,width,channels), np.uint8)

                info_image = draw_table(empty_image, 
                                        (joint_angles, user_angles, expert_angles, diff_angles, sim_angles),
                                        (joints_lines_relative, user_lines, expert_lines, diff_lines_relative, sim_lines_relative),
                                        (joints_lines_horizontal, user_horizontal_lines, expert_horizontal_lines, diff_lines_horizontal, sim_lines_horizontal),
                                        (joints_distance, user_distances, expert_distances, diff_distances, sim_distances),
                                        (frame1, frame2), 
                                        i)
                
                plot_image = draw_plots(np.zeros((height,width,channels), 
                                                np.uint8), 
                                                (user_angles, expert_angles, joint_angles,),
                                                (user_lines, expert_lines, joints_lines_relative),
                                                (user_horizontal_lines, expert_horizontal_lines, joints_lines_horizontal),
                                                (user_distances, expert_distances, joints_distance),
                                                path, 
                                                frame1, 
                                                frame2)
                
                # print feedback
                if mistake_type != "general" and frame2 in list(feedback_per_frame.keys()):
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1.2
                    font_thickness = 2
                    text_color = (255, 255, 255)
                    draw_multiline_text(info_image, feedback_per_frame[frame2], (20, 250), font, font_scale, text_color, font_thickness)


                side_image = cv2.vconcat([info_image, plot_image])
                stacked_frame = cv2.vconcat([user_frame, expert_frame])
                stacked_frame = cv2.hconcat([stacked_frame, side_image])

                resize_frame = cv2.resize(stacked_frame, None, fx=0.5, fy=0.5)

                if True:
                    cv2.imshow("User video", resize_frame)

                if video_writer == 1:
                    # Define the video codec and output file
                    os.makedirs(run_args.FEEDBACK.OUTPUT_PATH, exist_ok=True)
                    output_video_path = os.path.join(run_args.FEEDBACK.OUTPUT_PATH, "output_video_" + run_args.VIDEO_PATH.split("\\")[-1].split(".")[0] + ".mp4")
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
                    video_output_size = (resize_frame.shape[1], resize_frame.shape[0])  # Set the desired output size
                    fps = 5
                    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, video_output_size)

                if run_args.FEEDBACK.SAVE_VIDEO:
                    video_writer.write(resize_frame)
            
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    if video_writer:
        video_writer.release()

    if show_feedback:
        cv2.destroyAllWindows()
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    # for evaluation purposes
    if run_args.FEEDBACK.SAVE_STATISTICS:
        save_summary_for_video(ID, SKIER_ID,  evaluation_file, summary_feedback)
    

if __name__ == '__main__':
    main()