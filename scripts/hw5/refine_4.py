import copy
import pathlib
import json
import os
from shutil import copyfile

from PIL import Image


def mkdir_p(dir_path):
    pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)


def save_json(file_path, data, indent=2):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=indent)


def refine_samples(samples):
    imgs_dir = os.path.join("images")
    mkdir_p(imgs_dir)

    sub_samples_lst = (
        [s for s in samples if s["condition"]["roi"] is None],
        [s for s in samples if s["condition"]["roi"] and s["condition"]["roi"]["type"] == "rectangle"],
        [s for s in samples if s["condition"]["roi"] and s["condition"]["roi"]["type"] == "polygon"],
    )
    name_prefixs = ("none", "rect", "poly")

    refined_samples = []
    for sub_samples, name_prefix in zip(sub_samples_lst, name_prefixs):
        sub_samples = sorted(sub_samples, key=lambda s: len(s["detections"]))
        refined_sub_samples = []
        for idx, sample in enumerate(sub_samples):
            refined_sample = {}
            img_src_path = sample["image"]
            img_tgt_path = os.path.join(imgs_dir, "{}_{}.jpg".format(
                name_prefix,
                str(idx).zfill(2),
            ))
            refined_sample["image"] = img_tgt_path
            refined_sample["detections"] = sample["detections"]
            refined_sample["condition"] = sample["condition"]

            if img_src_path != img_tgt_path:
                if img_src_path.endswith(".jpg"):
                    copyfile(img_src_path, img_tgt_path)
                else:
                    Image.open(img_src_path).save(img_tgt_path)

            refined_sub_samples.append(refined_sample)
            print(len(refined_sample["detections"]))

        refined_samples += refined_sub_samples
        print(len(refined_samples))

    save_json(os.path.join("samples_before.json"), refined_samples)


def main():
    data_path = "./output/data_refined3.json"
    data_path = "./output_2/data_fin.json"

    with open(data_path, "r") as file_obj:
        samples = json.load(file_obj)

    refine_samples(samples)


if __name__ == "__main__":
    main()
