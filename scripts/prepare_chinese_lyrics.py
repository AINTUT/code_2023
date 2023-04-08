import glob
import os
import pathlib

import opencc


def load_text(file_path):
    with open(file_path, "r") as f:
        return f.read()


def save_text(file_path, text):
    with open(file_path, "w") as f:
        f.write(text)


def mkdir_p(dir_path):
    pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)


def main():
    input_root = "Chinese_Lyrics"
    output_root = "chinese_lyrics"

    converter = opencc.OpenCC("s2t")

    src_paths = glob.glob(os.path.join(input_root, "**", "*.txt"))

    for src_path in src_paths:
        _, src_singer, src_basename = src_path.split("/")

        simp_singer = src_singer.split("_")[0]
        trad_singer = converter.convert(simp_singer)

        src_name = os.path.splitext(src_basename)[0]
        simp_title = "_".join(src_name.split("_")[:-1])
        trad_title = converter.convert(simp_title)

        simp_text = load_text(src_path)
        trad_text = converter.convert(simp_text)

        tgt_dir = os.path.join(output_root, trad_singer)
        tgt_path = os.path.join(tgt_dir, "{}.txt".format(trad_title))

        mkdir_p(tgt_dir)
        save_text(tgt_path, trad_text)


if __name__ == "__main__":
    main()
