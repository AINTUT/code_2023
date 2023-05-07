import copy
import json

import cv2
import numpy as np

from my_filter import draw_sample


def save_json(file_path, data, indent=2):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=indent)


def refine_samples(samples):
    refined_samples = copy.deepcopy(samples)

    cv2.namedWindow("image", cv2.WINDOW_NORMAL)

    for refined_sample in refined_samples:
        img = cv2.imread(refined_sample["image"])
        img = draw_sample(refined_sample)
        print(refined_sample["condition"]["target_labels"])
        rect = cv2.selectROI('image', img, fromCenter=False, showCrosshair=True)

        print((img.shape[1], img.shape[0]))
        print(rect)

        if rect != (0, 0, 0, 0):
            roi = {
                "type": "rectangle",
                "region": {
                    "xmin": int(rect[0]),
                    "ymin": int(rect[1]),
                    "xmax": int(rect[0] + rect[2]),
                    "ymax": int(rect[1] + rect[3]),
                }
            }
            refined_sample["condition"]["roi"] = roi
            print(roi)

    return refined_samples


def main():
    data_path = "./output/data_refined1.json"
    output_path = "./output/data_refined2.json"

    with open(data_path, "r") as file_obj:
        samples = json.load(file_obj)

    refined_samples = refine_samples(samples)

    save_json(output_path, refined_samples)


if __name__ == "__main__":
    main()
