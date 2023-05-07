import json
import pathlib
import os
from glob import glob

import cv2
from imutils import resize
from ultralytics import YOLO
from tqdm import tqdm


def mkdir_p(dir_path):
    pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)


def save_json(file_path, data, indent=2):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=indent)


def result2dict(img_path, result, cls2label):
    category_name = img_path.split("/")[-2]
    category_name2traget_labels = {
        "vehicle": ["motorcycle", "bus", "truck", "car"],
        "motorcycle": ["motorcycle"],
        "person": ["person"],
        "bicycle": ["bicycle"],
        "cow": ["cow"],
        "umbrella": ["umbrella"],
    }

    sample = {
        "image": img_path,
        "condition": {
            "target_labels": category_name2traget_labels[category_name],
        },
        "detections": [],
    }

    for bbox in result.boxes:
        xyxy = bbox.xyxy.cpu().numpy()[0]
        conf = bbox.conf.cpu().item()
        cls = bbox.cls.item()
        sample["detections"].append({
            "bbox": {
                "xmin": int(xyxy[0]),
                "ymin": int(xyxy[1]),
                "xmax": int(xyxy[2]),
                "ymax": int(xyxy[3]),
            },
            "confidence": float(conf),
            "label": cls2label[int(cls)],
            "filtered": False,
        })

    return sample


def main():
    model_arch = "yolov8n"
    model_arch = "yolov8x"
    model = YOLO(model_arch)

    if True:
        # output_root = "output"
        output_root = "output_2"

        mkdir_p(output_root)

        samples = []
        # img_paths = glob("/home/feabries/data/tmp_crowd/data/**/*")
        img_paths = glob("/home/feabries/data/tmp_crowd/data_2/**/*")

        for img_path in tqdm(img_paths):
            category_name = img_path.split("/")[-2]
            tgt_dir = os.path.join(output_root, category_name)
            tgt_img_path = os.path.join(tgt_dir, os.path.basename(img_path))
            mkdir_p(tgt_dir)

            result = model.predict(img_path, conf=0.1)[0]
            res_plotted = result.plot()
            cv2.imwrite(tgt_img_path, res_plotted)

            sample = result2dict(img_path, result, model.model.names)
            samples.append(sample)

        save_json(os.path.join(output_root, "data.json"), samples)

        return

    img_path = "/home/feabries/data/tmp_crowd/data/person/"
    img_path = "/home/feabries/data/tmp_crowd/data/person/1531755250.png"
    img_path = "/home/feabries/data/tmp_crowd/data/person/1532001826.png"
    img_path = "/home/feabries/data/tmp_crowd/data/person/shopping-street-g5f2047311_1280.jpg"
    img_path = "/home/feabries/data/tmp_crowd/data/person/1535190491.png"
    img_path = "/home/feabries/data/tmp_crowd/data/umbrella/people-g3214b926e_1280.jpg"
    img_path = "/home/feabries/data/tmp_crowd/data/person/street-g260dd8fe2_1280.jpg"
    img_path = "/home/feabries/data/tmp_crowd/released-dataset/images/0139.jpg"
    img_path = "/home/feabries/data/tmp_crowd/data/vehicle/car-6810885_1280.jpg"
    img_path = "/home/feabries/data/tmp_crowd/data/person/crowd-4987227_1280.jpg"
    img_path = "/home/feabries/data/tmp_crowd/data/bicycle/c1s2_053796.jpg"
    img_path = "/home/feabries/data/tmp_crowd/data/sheep/sheep-4504992_1280.jpg"
    img_path = "/home/feabries/data/tmp_crowd/data/cow/steppe-3807616_1280.jpg"
    img_path = "/home/feabries/data/tmp_crowd/data/umbrella/umbrellas-2862071_1280.jpg"
    img_path = "/home/feabries/data/tmp_crowd/data/person/1532100613.png"
    img_path = "/home/feabries/data/tmp_crowd/data/person/1532001826.png"
    img_path = "/home/feabries/data/tmp_crowd/data/vehicle/nascar-334706_1920.jpg"
    img_path = "/home/feabries/data/tmp_crowd/data/person/geese-908291_1920.jpg"
    img_path = "/home/feabries/data/tmp_crowd/data/person/women-5963960_1920.jpg"

    # results = model.predict(img_path, save=True)
    results = model.predict(img_path, conf=0.1)

    result = results[0]
    res_plotted = result.plot()
    cv2.imshow("result", resize(res_plotted, height=800))
    cv2.waitKey(0)

    print(json.dumps(result2dict(img_path, result, model.model.names), indent=2))
    '''
    for bbox in result.boxes[:]:
        print("===========")
        print("xyxy", bbox.xyxy)
        print("xywh", bbox.xywh)
        print("conf", bbox.conf)
        print("cls", bbox.cls)
    '''


if __name__ == "__main__":
    main()
