import os
import argparse
import json
import time
import math
import re

from tqdm import tqdm

import numpy as np

import cv2
import mmcv

from scipy.optimize import linear_sum_assignment

from processing import extract_parts, draw
from config_reader import config_reader
from model.cmu_model import get_testing_model
import util


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_dir", type=str, default=r"D:\MMAct\videos", help="Path to dataset")
    parser.add_argument("--output_dir", type=str, default=r"D:\MMAct_annotator", help="Path to save json files")
    parser.add_argument('--model', type=str, default='model/keras/model.h5', help='path to the weights file')
    parser.add_argument("--extract_every_num_frames", type=int, default=1, help="Extract keypoints every specified of frames")
    parser.add_argument("--is_save_video", type=bool, default=False)
    parser.add_argument("--video_save_dir", type=str, default=r"D:\MMAct_save_videos")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)

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


def draw_canvas_with_info(input_image, all_peaks, subset, candidate, resize_fac=3):
    canvas = input_image.copy()
    height, width = canvas.shape[:2]

    subset_array = np.array(subset, dtype=np.int32)
    candidate_array = np.array(candidate, dtype=np.float32)
    persons = np.where(subset_array[:, :18, np.newaxis] > 0, candidate_array[:, :3][subset_array[:, :18]], -1. * np.ones_like(subset_array[:, :18])[:, :, np.newaxis])
    persons[:, :, :2] = np.where(persons[:, :, :2] > 0, persons[:, :, :2] * resize_fac, persons[:, :, :2])
    persons[:, :, :2] = np.where(persons[:, :, :2] > 0, (persons[:, :, :2] + np.array([1])[np.newaxis, np.newaxis, :]) / np.array([width, height])[np.newaxis, np.newaxis, :], persons[:, :, :2])
    xy_min = np.min(persons[:, :, :2], axis=1, keepdims=False)
    xy_max = np.max(persons[:, :, :2], axis=1, keepdims=False)

    wh = (xy_max - xy_min) * 100.

    persons = persons.tolist()

    for i, person in enumerate(persons):
        w, h = wh[i]
        for j, point in enumerate(person):
            x, y, prob = point
            cv2.circle(canvas, (int(x * width), int(y * height)), radius=1, color=util.colors[j], thickness=-1)
            cv2.putText(canvas,
                        text="x: " + str(round(x, 3)) + " y: " + str(round(y, 3)) + " prob: " + str(round(prob, 3)), org=(int(x * width) + 3, int(y * height) - 3),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX,
                        fontScale=0.27,
                        color=util.colors[j],
                        thickness=1)

    stickwidth = 4

    for i in range(17):
        for s in subset:
            index = s[np.array(util.limbSeq[i]) - 1]
            if -1 in index:
                continue
            cur_canvas = canvas.copy()
            y = candidate[index.astype(int), 0]
            x = candidate[index.astype(int), 1]
            # print("i:", i, "np.array(util.limbSeq[i])-1:", np.array(util.limbSeq[i]) - 1, "index:", index, "y:", y, "x:", x)
            m_x = np.mean(x)
            m_y = np.mean(y)
            length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(x[0] - x[1], y[0] - y[1]))
            polygon = cv2.ellipse2Poly((int(m_y * resize_fac), int(m_x * resize_fac)),
                                       (int(length * resize_fac / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, util.colors[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
            # cv2.imshow("canvas", canvas)
            # cv2.waitKey(0)

    return canvas

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
    print("Start with video from index:", start_index)
    for i in tqdm(range(start_index, len(videos_list))):
        video = videos_list[i]
        video_class = os.path.basename(video).split(".")[0].lower()
        json_filename = video_class + ".json"
        dirname_compose = list(filter(None, os.path.dirname(video).split(os.sep)))
        relative_dirname = os.path.join(*diff_list(dirname_compose, dataset_dir_compose))
        out_dirname = os.path.join(args.output_dir, relative_dirname)
        #print(dirname_compose, relative_dirname, out_dirname)
        subject_name = list(filter(None, relative_dirname.split(os.sep)))[0]
        assert subject_name.startswith("subject")
        #subject_int = int(re.findall(r"\d+", subject_name)[0])
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
        if args.is_save_video:
            out_video_filename = os.path.basename(video).split(".")[0] + ".avi"
            out_video_dirname = os.path.join(args.video_save_dir, relative_dirname)
            if not os.path.exists(out_video_dirname):
                os.makedirs(out_video_dirname, exist_ok=True)
            out_video = cv2.VideoWriter(os.path.join(out_video_dirname, out_video_filename),
                                        cv2.VideoWriter_fourcc(*'DIVX'), input_fps, (width, height))
        #
        scale_search = [1]  # [.5, 1, 1.5, 2]
        params['scale_search'] = scale_search
        j = 0  # default is 0
        while (cam.isOpened()) and ret_val is True:
            if j % args.extract_every_num_frames == 0:
                tic = time.time()
                input_image = cv2.cvtColor(orig_image, cv2.COLOR_RGB2BGR)
                input_image = cv2.resize(input_image, (args.width, args.height))
                draw_image = input_image.copy()[:, :, ::-1]
                tic_cvclars = time.time()
                # generate image with body parts
                body_parts, all_peaks, subset, candidate = extract_parts(input_image, params, model, model_params)
                #print("body_parts:", body_parts, "all_peaks:", all_peaks, "subset:", subset, "candidate", candidate)
                tic_scekp = time.time()
                #canvas = draw(draw_image, all_peaks, subset, candidate) # From orig_image to draw_image
                if args.is_save_video:
                    #print("Visualize")
                    canvas = draw_canvas_with_info(orig_image, all_peaks, subset, candidate, resize_fac=int(width / args.width))
                    out_video.write(canvas)
                    cv2.imshow("canvas", canvas)
                    cv2.waitKey(1000)
                #print('Processing frame: ', j)
                frame_info = {}
                if video_class in check_id_classes:
                    if len(subset) > 0:
                        subset = np.array(subset, dtype=np.int32)
                        #print(subset.shape)
                        candidate = np.array(candidate, dtype=np.float32)
                        #print(candidate.shape)
                        predicted_coords = np.where(subset[:, :18, np.newaxis] > 0, candidate[:, :3][subset[:, :18]], -1 * np.ones_like(subset[:, :18])[:, :, np.newaxis])
                        predicted_coords[:, :, :2] = np.where(predicted_coords[:, :, :2] > 0., (predicted_coords[:, :, :2] + np.array([1])[np.newaxis, np.newaxis, :]) / np.array([args.width, args.height])[np.newaxis, np.newaxis, :], predicted_coords[:, :, :2])
                        #print("coords:", predicted_coords)
                        predicted_reid_dict = dict(zip(range(len(list(predicted_coords))), list(predicted_coords)))
                        #print("First predicted_reid_dict", predicted_reid_dict)
                        #print(predicted_reid_dict)
                        if j == 0:
                            previous_reid_dict = predicted_reid_dict
                            active_ids = list(range(len(predicted_reid_dict.keys())))
                            num_id = len(predicted_reid_dict.keys())
                            col_to_id = dict(zip(range(num_id), active_ids))
                        else:
                            dist_matrix = np.zeros(shape=[len(predicted_reid_dict.keys()), len(previous_reid_dict.keys())])
                            for d, d_key in enumerate(predicted_reid_dict.keys()):
                                #print("d, d_key", d, d_key)
                                for v, v_key in enumerate(previous_reid_dict.keys()):
                                    #print("v, v_key", v, v_key)
                                    sub = np.where(np.logical_and(predicted_reid_dict[d_key][:, :2] > 0., previous_reid_dict[v_key][:, :2] > 0), (100 * predicted_reid_dict[d_key][:, :2] - 100 * previous_reid_dict[v_key][:, :2]), np.zeros_like(predicted_reid_dict[d][:, :2])) # Huhu, miss dkey->d, v_key->v and now fixed
                                    sqr = np.sum(np.square(sub), axis=-1, keepdims=False)
                                    dist = np.mean(np.sqrt(sqr))
                                    #print("d, v, d_key, v_key", d, v, d_key, v_key, dist)
                                    dist_matrix[d, v] = dist # d -> d_key, v -> v_key
                            #print("dist matrix", dist_matrix)
                            row_ind, col_ind = linear_sum_assignment(dist_matrix)
                            row_ind = list(row_ind)
                            col_ind = list(col_ind)
                            match = dict(zip(row_ind, col_ind))
                            #print("match row to col ind", match)
                            row_to_id = {k: col_to_id[v] for k, v in match.items()}
                            #print("row index to id", row_to_id)
                            if dist_matrix.shape[0] == dist_matrix.shape[1]:
                                #print("Case 1")
                                predicted_reid_dict = {row_to_id[k]: v for k, v in predicted_reid_dict.items()}
                                previous_reid_dict = predicted_reid_dict
                                col_to_id = row_to_id # It's wrong if not dist_matrix[d, v] = dist instead of dist_matrix[d_key, v_key] = dist
                                #col_to_id = dict(zip(range(dist_matrix.shape[0]), sorted(predicted_reid_dict.keys())))
                                #sorted(previous_reid_dict.keys())
                                #previous_reid_dict = {previous_reid_dict[k] for k in sorted(previous_reid_dict.keys())}
                                #print("col_to_id", col_to_id)
                            elif dist_matrix.shape[0] < dist_matrix.shape[1]:
                                #print("Case 2")
                                predicted_reid_dict = {row_to_id[k]: v for k, v in predicted_reid_dict.items()}
                                miss_cols = list(filter(lambda l: l not in col_ind, list(range(dist_matrix.shape[1]))))
                                miss_ids = [col_to_id[col] for col in miss_cols]
                                active_ids = list(filter(lambda l: l not in miss_ids, active_ids))
                                previous_reid_dict = predicted_reid_dict
                                col_to_id = row_to_id
                            else:
                                #print("Case 3")
                                miss_rows = list(filter(lambda l: l not in row_ind, list(range(dist_matrix.shape[0]))))
                                for m, row in enumerate(miss_rows):
                                    row_to_id[row] = num_id + m
                                active_ids = active_ids + list(range(num_id, num_id + len(miss_rows)))
                                num_id += len(miss_rows)
                                predicted_reid_dict = {row_to_id[k]: v for k, v in predicted_reid_dict.items()}
                                previous_reid_dict = predicted_reid_dict
                                col_to_id = row_to_id
                        for k in predicted_reid_dict.keys():
                            frame_info[subject_name + str(k + 1)] = {k: v for k, v in zip(list(map(lambda x: label_to_keypoint[x], list(range(predicted_reid_dict[k].shape[0])))), list(map(lambda x: dict(zip(["x", "y", "prob"], list(map(lambda y: round(y, 3), x)))), predicted_reid_dict[k].tolist())))}
                            #print(type(predicted_reid_dict[k].tolist()[0]), list(map(lambda x: dict(zip(["x", "y", "z"], x)), predicted_reid_dict[k].tolist())))
                    #else:
                    #    continue
                else:
                    for k, person in enumerate(subset):
                        frame_info[subject_name + str(k + 1)] = {}
                        for h, point in enumerate(person):
                            if h == 18:
                                break
                            if point == -1.:
                                frame_info[subject_name + str(k + 1)][label_to_keypoint[h]] = {"x": -1.0, "y": -1.0, "prob": -1.0}
                                continue
                            x = int(candidate[point.astype(int), 0])
                            y = int(candidate[point.astype(int), 1])
                            prob = candidate[point.astype(int), 2]
                            frame_info[subject_name + str(k + 1)][label_to_keypoint[h]] = {"x": round((x + 1) / args.width, 3),
                                                                                           "y": round((y + 1) / args.height,3),
                                                                                           "prob": round(prob, 3)}
                npart, dpart = str(format((j / input_fps), ".3f")).split(".")
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
        if args.is_save_video:
            out_video.release()
            #print("Save video with info")
        with open(os.path.join(out_dirname, json_filename), "w+") as f:
            json.dump(video_info, f, indent=4)
