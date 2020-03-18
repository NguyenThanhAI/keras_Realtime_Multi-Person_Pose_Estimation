import os
import argparse
import json
import time

from tqdm import tqdm

import numpy as np

import cv2
import mmcv

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

    model = get_testing_model()
    model.load_weights(keras_weights_file)

    # load config
    params, model_params = config_reader()

    dataset_dir_compose = args.dataset_dir.split(os.sep)

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
        #print(start_index)
        for i in tqdm(range(start_index, len(videos_list))):
            video = videos_list[i]
            print("Video ", i)
            json_filename = os.path.basename(video).split(".")[0] + ".json"
            print(json_filename)
            dirname_compose = os.path.dirname(video).split(os.sep)
            relative_dirname = os.path.join(*diff_list(dirname_compose, dataset_dir_compose))
            out_dirname = os.path.join(args.output_dir, relative_dirname)
            #print(dirname_compose, relative_dirname, out_dirname)

            if not os.path.exists(out_dirname):
                os.makedirs(out_dirname, exist_ok=True)

            frames = mmcv.VideoReader(video)

            with open("videos_file.txt", "w") as f:
                f.write(video + "\n")
                f.close()
            video_info = {}
            for j, frame in enumerate(frames):
                print("frame ", j, len(frames))
                height, width = frame.shape[:2]
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                body_parts, all_peaks, subset, candidate = extract_parts(frame_bgr, params, model, model_params)

                canvas = draw(frame, all_peaks, subset, candidate)

                frame_info = {}
                for k, person in enumerate(subset):
                    frame_info["person_" + str(k).zfill(2)] = {}
                    for h, point in enumerate(person):
                        if h == 18:
                            break

                        if point == -1.:
                            frame_info["person_" + str(k).zfill(2)][label_to_keypoint[h]] = {"x": 0., "y": 0., "prob": 0.}
                            continue

                        x = int(candidate[point.astype(int), 0])
                        y = int(candidate[point.astype(int), 1])
                        prob = candidate[point.astype(int), 2]
                        frame_info["person_" + str(k).zfill(2)][label_to_keypoint[h]] = {"x": round((x + 1) / width, 3), "y": round((y + 1) / height, 3), "prob": round(prob, 3)}

                video_info["frame_" + str(j).zfill(5)] = frame_info
                #print(video_info)
#
                #with open(os.path.join(out_dirname, json_filename), "w+") as f:
                #    json.dump(video_info, f, indent=4)
#
                #cv2.imshow("canvas", canvas)
                cv2.imwrite("image_to_vide/image" + str(j).zfill(5) + ".jpg", canvas)
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

                canvas = draw(img, all_peaks, subset, candidate)
                frame_info = {}
                for k, person in enumerate(subset):
                    frame_info["person_" + str(k).zfill(2)] = {}
                    for h, point in enumerate(person):
                        if h == 18:
                            break

                        if point == -1.:
                            frame_info["person_" + str(k).zfill(2)][label_to_keypoint[h]] = {"x": 0., "y": 0., "prob": 0.}
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
