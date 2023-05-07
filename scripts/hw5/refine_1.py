import copy
import json

import cv2
import numpy as np


np.random.seed(999887)


def save_json(file_path, data, indent=2):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=indent)


def refine_samples(samples):
    refined_samples = copy.deepcopy(samples)

    for refined_sample in refined_samples:
        contained_labels = set([d["label"] for d in refined_sample["detections"]])
        if len(contained_labels) <= 1:
            refined_sample["condition"]["target_labels"] = None

        confidences = [d["confidence"] for d in refined_sample["detections"]]
        confidence_mean = np.mean(confidences)
        if confidence_mean >= 0.5:
            confidence_threshold = 0.25
            '''
            if (len(contained_labels) > 1) and np.random.rand() <= 0.1:
                confidence_threshold = None
                print("====!!!!!!===")
            '''
        else:
            confidence_threshold = (confidence_mean / 2) + np.random.rand() * (confidence_mean / 2)
            confidence_threshold = round(confidence_threshold, 2)
        print(refined_sample["image"], confidence_mean, confidence_threshold)
        refined_sample["condition"]["confidence_threshold"] = confidence_threshold

        im_height, im_width, _ = cv2.imread(refined_sample["image"]).shape
        refined_sample["condition"]["roi"] = None
        '''
        refined_sample["condition"]["roi"] = {
            "type": "polygon",
            "region": [
                (int(im_width * 0.2), int(im_height * 0.2)),
                (int(im_width * 0.3), int(im_height * 0.1)),
                (int(im_width * 0.9), int(im_height * 0.5)),
                (int(im_width * 0.5), int(im_height * 0.8)),
                (int(im_width * 0.15), int(im_height * 0.7)),
                (int(im_width * 0.1), int(im_height * 0.5)),
            ]
        }
        refined_sample["condition"]["roi"] = {
            "type": "rectangle",
            "region": {
                "xmin": int(im_width * 0.2),
                "ymin": int(im_height * 0.2),
                "xmax": int(im_width * 0.8),
                "ymax": int(im_height * 0.8),
            }
        }
        '''

    return refined_samples


def main():
    data_path = "./output/data.json"
    output_path = "./output/data_refined1.json"

    data_path = "./output_2/data.json"
    output_path = "./output_2/data_refined1.json"

    with open(data_path, "r") as file_obj:
        samples = json.load(file_obj)

    refined_samples = refine_samples(samples)

    save_json(output_path, refined_samples)


if __name__ == "__main__":
    main()
