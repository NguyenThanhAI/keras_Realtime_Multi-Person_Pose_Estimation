import os
import argparse
import json
import time

from tqdm import tqdm

import numpy as np

import cv2
import mmcv

from scipy.optimize import linear_sum_assignment

from processing import extract_parts, draw
from config_reader import config_reader
from model.cmu_model import get_testing_model


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_dir", type=str, default=r"D:\MMAct\videos", help="Path to dataset")
    parser.add_argument("--output_dir", type=str, default=r"D:\MMAct_annotator", help="Path to save json files")
    parser.add_argument('--model', type=str, default='model/keras/model.h5', help='path to the weights file')
    parser.add_argument("--run_format", type=str, choices=("videos", "images"), default="videos")

    args  = parser.parse_args()

    return args


def diff_list(li1, li2):
    return li1[len(li2):]


def enumerate_videos(args):
    videos_list = []
    for dirs, _, files in os.walk(args.dataset_dir):
        for file in files:
            if file.endswith((".mp4", ".avi", ".mkv")):
                videos_list.append(os.path.join(dirs, file))
    #print(videos_list)
    return videos_list


def enumerate_videos_folder(args):
    """
    Use if dataset dir is collection of frames extracted from videos
    :param args:
    :return: List of videos folder
    """
    videos_dict  = [] # keys: name of videos folder, values: list of image path in this folder
    for dirs, _, files in os.walk(args.dataset_dir):
        subfiles_list = os.listdir(dirs)
        is_videos = True # a bool var to indicate if this dir is a video contains only images of frames
        for subfile in subfiles_list:
            if os.path.isdir(os.path.join(dirs, subfile)):
                is_videos = False
                break

        if not is_videos:
            continue

        videos_dict.append(dirs)
    #print(videos_dict)
    return videos_dict


if __name__ == '__main__':

    args = get_args()

    keypoint_to_label = {'nose': 0, 'neck': 1, 'right_shoulder': 2, 'right_elbow': 3, 'right_wrist': 4,
                         'left_shoulder': 5, 'left_elbow': 6, 'left_wrist': 7, 'right_hip': 8, 'right_knee': 9,
                         'right_ankle': 10, 'left_hip': 11, 'left_knee': 12, 'left_ankle': 13, 'right_eye': 14,
                         'left_eye': 15, 'right_ear': 16, 'left_ear': 17}
    label_to_keypoint = {v: k for k, v in keypoint_to_label.items()}
    keras_weights_file = args.model

    check_id_classes = ["talking", "transferring_object"]

    model = get_testing_model()
    model.load_weights(keras_weights_file)

    # load config
    params, model_params = config_reader()

    dataset_dir_compose = list(filter(None, args.dataset_dir.split(os.sep)))

    if args.run_format == "videos":
        videos_list =  enumerate_videos(args)

        if not os.path.exists("videos_file.txt"):
            start_index = 0

        else:
            with open("videos_file.txt", "r") as f:
                content = f.read()
                content = content.split("\n")
                content = list(filter(None, content))

                assert len(content) == 1

                start_index = videos_list.index(content[0])

                f.close()
        print(start_index)
        for i in tqdm(range(start_index, len(videos_list))):
            video = videos_list[i]
            video_class = os.path.basename(video).split(".")[0].lower()
            json_filename = video_class + ".json"

            dirname_compose = list(filter(None, os.path.dirname(video).split(os.sep)))
            relative_dirname = os.path.join(*diff_list(dirname_compose, dataset_dir_compose))
            out_dirname = os.path.join(args.output_dir, relative_dirname)
            #print(dirname_compose, relative_dirname, out_dirname)

            if not os.path.exists(out_dirname):
                os.makedirs(out_dirname, exist_ok=True)

            with open("videos_file.txt", "w") as f:
                f.write(video + "\n")
                f.close()
            video_info = {}
            cam = cv2.VideoCapture(video)
            input_fps = cam.get(cv2.CAP_PROP_FPS)
            ret_val, orig_image = cam.read()
            height, width = orig_image.shape[:2]
            video_length = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))

            #

            scale_search = [1]  # [.5, 1, 1.5, 2]

            params['scale_search'] = scale_search

            j = 0  # default is 0
            while (cam.isOpened()) and ret_val is True:
                tic = time.time()
                input_image = cv2.cvtColor(orig_image, cv2.COLOR_RGB2BGR)
                input_image = cv2.resize(input_image, (640, 360))
                draw_image = input_image.copy()[:, :, ::-1]
                tic_cvclars = time.time()
                # generate image with body parts
                body_parts, all_peaks, subset, candidate = extract_parts(input_image, params, model, model_params)
                #print("body_parts:", body_parts, "all_peaks:", all_peaks, "subset:", subset, "candidate", candidate)
                tic_scekp = time.time()
                #canvas = draw(draw_image, all_peaks, subset, candidate) # From orig_image to draw_image
                #print('Processing frame: ', j)
                frame_info = {}
                if video_class in check_id_classes:
                    if len(subset) > 0:
                        subset = np.array(subset, dtype=np.int32)
                        #print(subset.shape)
                        candidate = np.array(candidate, dtype=np.float32)
                        #print(candidate.shape)
                        predicted_coords = np.where(subset[:, :18, np.newaxis] > 0, candidate[:, :3][subset[:, :18]], -1 * np.ones_like(subset[:, :18])[:, :, np.newaxis])
                        predicted_coords[:, :, :2] = np.where(predicted_coords[:, :, :2] > 0., (predicted_coords[:, :, :2] + np.array([1])[np.newaxis, np.newaxis, :]) / np.array([width, height])[np.newaxis, np.newaxis, :], predicted_coords[:, :, :2])
                        #print("coords:", predicted_coords)
                        predicted_reid_dict = dict(zip(range(len(list(predicted_coords))), list(predicted_coords)))
                        #print(predicted_reid_dict)
                        if j == 0:
                            previous_reid_dict = predicted_reid_dict
                            active_ids = list(range(len(predicted_reid_dict.keys())))
                            num_id = len(predicted_reid_dict.keys())
                            col_to_id = dict(zip(range(num_id), active_ids))
                        else:
                            dist_matrix = np.zeros(shape=[len(predicted_reid_dict.keys()), len(previous_reid_dict.keys())])
                            for d, d_key in enumerate(predicted_reid_dict.keys()):
                                for v, v_key in enumerate(previous_reid_dict.keys()):
                                    sub = np.where(np.logical_and(predicted_reid_dict[d_key][:, :2] > 0., previous_reid_dict[v_key][:, :2] > 0), (100 * predicted_reid_dict[d][:, :2] - 100 * previous_reid_dict[v][:, :2]), np.zeros_like(predicted_reid_dict[d][:, :2]))
                                    sqr = np.sum(np.square(sub), axis=-1, keepdims=False)
                                    dist = np.mean(np.sqrt(sqr))
                                    dist_matrix[d, v] = dist
                            print("dist matrix", dist_matrix)
                            row_ind, col_ind = linear_sum_assignment(dist_matrix)

                            row_ind = list(row_ind)
                            col_ind = list(col_ind)
                            match = dict(zip(row_ind, col_ind))
                            row_to_id = {k: col_to_id[v] for k, v in match.items()}

                            if dist_matrix.shape[0] == dist_matrix.shape[1]:
                                predicted_reid_dict = {row_to_id[k]: v for k, v in predicted_reid_dict.items()}
                                previous_reid_dict = predicted_reid_dict
                                col_to_id = row_to_id
                            elif dist_matrix.shape[0] < dist_matrix.shape[1]:
                                predicted_reid_dict = {row_to_id[k]: v for k, v in predicted_reid_dict.items()}
                                miss_cols = list(filter(lambda l: l not in col_ind, list(range(dist_matrix.shape[1]))))
                                miss_ids = [col_to_id[col] for col in miss_cols]
                                active_ids = filter(lambda l: l not in miss_ids, active_ids)
                                previous_reid_dict = predicted_reid_dict
                                col_to_id = row_to_id
                            else:
                                miss_rows = list(filter(lambda l: l not in row_ind, list(range(dist_matrix.shape[0]))))
                                for m, row in enumerate(miss_rows):
                                    row_to_id[row] = num_id + m
                                active_ids = active_ids + list(range(num_id, num_id + len(miss_rows)))
                                num_id += len(miss_rows)
                                predicted_reid_dict = {row_to_id[k]: v for k, v in predicted_reid_dict.items()}
                                previous_reid_dict = predicted_reid_dict
                                col_to_id = row_to_id
                        for k in predicted_reid_dict.keys():
                            frame_info["person_" + str(k).zfill(2)] = {}



                    else:
                        continue


                else:
                    for k, person in enumerate(subset):
                        frame_info["person_" + str(k).zfill(2)] = {}
                        for h, point in enumerate(person):
                            if h == 18:
                                break
                            if point == -1.:
                                frame_info["person_" + str(k).zfill(2)][label_to_keypoint[h]] = {"x": -1.0, "y": -1.0, "prob": -1.0}
                                continue
                            x = int(candidate[point.astype(int), 0])
                            y = int(candidate[point.astype(int), 1])
                            prob = candidate[point.astype(int), 2]
                            frame_info["person_" + str(k).zfill(2)][label_to_keypoint[h]] = {"x": round((x + 1) / width, 3),
                                                                                             "y": round((y + 1) / height,3),
                                                                                             "prob": round(prob, 3)}

                npart, dpart = str(round((j / input_fps), 3)).split(".")
                video_info[npart.zfill(3) + ":" + dpart.zfill(3)] = frame_info
                toc = time.time()
                total = toc - tic
                print("Processing frame %d, processing time is %.5f, time rate of convert to rgb and resize  %.3f percent, source code to extract keypoints %.3f percent, write to dict %.3f percent" % (j, total, ((tic_cvclars - tic) * 100. / total), ((tic_scekp - tic_cvclars) * 100. / total), (toc - tic_scekp) * 100. / total))
                j += 1
                ret_val, orig_image = cam.read()
                #print(video_info)
                #
                # with open(os.path.join(out_dirname, json_filename), "w+") as f:
                #    json.dump(video_info, f, indent=4)
                #
                #cv2.imshow("canvas", canvas)
                #cv2.waitKey(2000)
                #cv2.destroyWindow("canvas")

            with open(os.path.join(out_dirname, json_filename), "w+") as f:
                json.dump(video_info, f, indent=4)

    else:
        videos_dict = enumerate_videos_folder(args)

        if not os.path.exists("images_file.txt"):
            start_index = 0

        else:
            with open("images_file.txt", "r") as f:
                content = f.read()
                content = content.split("\n")
                content = list(filter(None, content))

                assert len(content) == 1

                start_index = videos_dict.index(content[0])

                f.close()
        #print(start_index)
        for i in tqdm(range(start_index, len(videos_dict))):
            video = videos_dict[i]

            json_filename = video.split(os.sep)[-1] + ".json"
            #(json_filename)
            dirname_compose = video.split(os.sep)[:-1] # [:-1] important
            relative_dirname = os.path.join(*diff_list(dirname_compose, dataset_dir_compose))
            out_dirname = os.path.join(args.output_dir, relative_dirname)
            #print(dirname_compose, relative_dirname, out_dirname)

            if not os.path.exists(out_dirname):
                os.makedirs(out_dirname, exist_ok=True)

            with open("images_file.txt", "w") as f:
                f.write(video + "\n")
                f.close()
            video_info = {}
            images_list = os.listdir(video)
            #print(images_list)

            for j, image in enumerate(images_list):
                assert image.endswith((".jpg", ".png", ".jpeg"))
                #print(j, image)

                image_path = os.path.join(video, image)

                img = cv2.imread(image_path)
                height, width = img.shape[:2]

                img_bgr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                body_parts, all_peaks, subset, candidate = extract_parts(img_bgr, params, model, model_params)

                #canvas = draw(img, all_peaks, subset, candidate)
                frame_info = {}
                for k, person in enumerate(subset):
                    frame_info["person_" + str(k).zfill(2)] = {}
                    for h, point in enumerate(person):
                        if h == 18:
                            break

                        if point == -1.:
                            frame_info["person_" + str(k).zfill(2)][label_to_keypoint[h]] = {"x": -1.0, "y": -1.0, "prob": -1.0}
                            continue

                        x = int(candidate[point.astype(int), 0])
                        y = int(candidate[point.astype(int), 1])
                        prob = candidate[point.astype(int), 2]
                        frame_info["person_" + str(k).zfill(2)][label_to_keypoint[h]] = {"x": round((x + 1) / width, 3), "y": round((y + 1) / height, 3), "prob": round(prob, 3)}

                video_info["frame_" + str(j).zfill(5)] = frame_info
                #print(video_info)
#
                #cv2.imshow("canvas", canvas)
                #cv2.waitKey(2000)
                #cv2.destroyWindow("canvas")

            with open(os.path.join(out_dirname, json_filename), "w+") as f:
                json.dump(video_info, f, indent=4)
