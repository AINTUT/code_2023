import copy
import json

import cv2
import numpy as np
from EasyROI import EasyROI

from my_filter import draw_sample


def save_json(file_path, data, indent=2):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=indent)


def refine_samples(samples):
    output_path = "./output/data_refined3.json"

    refined_samples = copy.deepcopy(samples)

    roi_helper = EasyROI(verbose=True)
    cv2.namedWindow("Draw 1 Polygon(s)", cv2.WINDOW_NORMAL)

    for refined_sample in refined_samples:
        roi = refined_sample["condition"]["roi"]
        if roi is not None:
            continue

        img = cv2.imread(refined_sample["image"])
        img = draw_sample(refined_sample)
        print(refined_sample["condition"]["target_labels"])
        print((img.shape[1], img.shape[0]))

        polygon_roi = roi_helper.draw_polygon(img)

        if polygon_roi:
            roi = {
                "type": "polygon",
                "region": np.array(polygon_roi["roi"][0]["vertices"]).tolist(),
            }
            refined_sample["condition"]["roi"] = roi
            print(roi)

        save_json(output_path, refined_samples)

    return refined_samples


def main():
    data_path = "./output/data_refined2.json"
    data_path = "./output/data_refined3.json"

    with open(data_path, "r") as file_obj:
        samples = json.load(file_obj)

    refine_samples(samples)


if __name__ == "__main__":
    main()
