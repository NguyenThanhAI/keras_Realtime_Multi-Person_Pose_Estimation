import os
import argparse
import time
import json

import numpy as np

import cv2

from processing import extract_parts, draw
from config_reader import config_reader
from model.cmu_model import get_testing_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='input image')
    parser.add_argument('--output', type=str, default='result.png', help='output image')
    parser.add_argument('--model', type=str, default='model/keras/model.h5', help='path to the weights file')

    args = parser.parse_args()
    image_path = args.image
    output = os.path.join(os.path.dirname(image_path), os.path.basename(image_path).split(".")[0] + "_" + args.output)
    keras_weights_file = args.model

    keypoint_to_label = {'nose': 0, 'neck': 1, 'right_shoulder': 2, 'right_elbow': 3, 'right_wrist': 4, 'left_shoulder': 5, 'left_elbow': 6, 'left_wrist': 7, 'right_hip': 8, 'right_knee': 9, 'right_ankle': 10, 'left_hip': 11, 'left_knee': 12, 'left_ankle': 13, 'right_eye': 14, 'left_eye': 15, 'right_ear': 16, 'left_ear': 17}
    label_to_keypoint = {v: k for k, v in keypoint_to_label.items()}

    tic = time.time()
    print('start processing...')

    # load model

    # authors of original model don't use
    # vgg normalization (subtracting mean) on input images
    model = get_testing_model()
    model.load_weights(keras_weights_file)

    # load config
    params, model_params = config_reader()
    
    input_image = cv2.imread(image_path)  # B,G,R order
    height, width = input_image.shape[:2]
    
    body_parts, all_peaks, subset, candidate = extract_parts(input_image, params, model, model_params)
    #print("body_parts:", body_parts, "all_peaks:", len(all_peaks), "subset:", subset, "candidate:", candidate)
    canvas = draw(input_image, all_peaks, subset, candidate)
    
    toc = time.time()
    print('processing time is %.5f' % (toc - tic))

    draw_image = input_image.copy()
    #for i, key in enumerate(body_parts.keys()):
    #    cv2.circle(draw_image, body_parts[key], 2, colors[i])
    #colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
    #for i in range(18):
    #    peak = all_peaks[i]
    #    print(len(peak))
    #    for j in range(len(peak)):
    #        #print(j)
    #        #print(peak[j][0], peak[j][1], colors[j])
    #        cv2.circle(draw_image, (peak[j][0], peak[j][1]), 3, colors[j], thickness=-1)
    results = {}
    for j, person in enumerate(subset):
        color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        results["person_" + str(j)] = {}
        for i, point in enumerate(person):
            if i == 18:
                break
            if point == -1.:
                results["person_" + str(j)][label_to_keypoint[i]] = {"x": 0., "y": 0.}
                continue
            x = int(candidate[point.astype(int), 0])
            y = int(candidate[point.astype(int), 1])
            cv2.circle(draw_image, (x, y), 3, color, thickness=-1)
            print(label_to_keypoint[i])
            results["person_" + str(j)][label_to_keypoint[i]] = {"x": round((x + 1) / width, 3), "y": round((y + 1) / height, 3)}
            #cv2.imshow("draw_img", draw_image)
            #cv2.waitKey(0)
    with open(os.path.join(os.path.dirname(image_path), os.path.basename(image_path).split(".")[0] + ".json"), "w+") as f:
        json.dump(results, fp=f)

    #cv2.imshow("", draw_image)
    #cv2.waitKey(0)

    cv2.imwrite(output, canvas)

    cv2.destroyAllWindows()
