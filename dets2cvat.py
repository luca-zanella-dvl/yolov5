from pathlib import Path
import argparse
import os
import shutil

COCO = "coco"
CROWDHUMAN = "crowdhuman"
COCO_CLASSES = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    12: "stop sign",
    13: "parking meter",
    14: "bench",
    15: "bird",
    16: "cat",
    17: "dog",
    18: "horse",
    19: "sheep",
    20: "cow",
    21: "elephant",
    22: "bear",
    23: "zebra",
    24: "giraffe",
    25: "backpack",
    26: "umbrella",
    27: "handbag",
    28: "tie",
    29: "suitcase",
    30: "frisbee",
    31: "skis",
    32: "snowboard",
    33: "sports ball",
    34: "kite",
    35: "baseball bat",
    36: "baseball glove",
    37: "skateboard",
    38: "surfboard",
    39: "tennis racket",
    40: "bottle",
    41: "wine glass",
    42: "cup",
    43: "fork",
    44: "knife",
    45: "spoon",
    46: "bowl",
    47: "banana",
    48: "apple",
    49: "sandwich",
    50: "orange",
    51: "broccoli",
    52: "carrot",
    53: "hot dog",
    54: "pizza",
    55: "donut",
    56: "cake",
    57: "chair",
    58: "couch",
    59: "potted plant",
    60: "bed",
    61: "dining table",
    62: "toilet",
    63: "tv",
    64: "laptop",
    65: "mouse",
    66: "remote",
    67: "keyboard",
    68: "cell phone",
    69: "microwave",
    70: "oven",
    71: "toaster",
    72: "sink",
    73: "refrigerator",
    74: "book",
    75: "clock",
    76: "vase",
    77: "scissors",
    78: "teddy bear",
    79: "hair drier",
    80: "toothbrush",
}
CROWDHUMAN_CLASSES = {
    1: "head",
    2: "person",
}


def main(args):
    labels_path = args.labels_path
    dataset_type = args.dataset_type

    ann_files = [
        f
        for f in os.listdir(labels_path)
        if os.path.isfile(os.path.join(labels_path, f))
    ]

    ann_dirs = [x.rpartition("_")[0] for x in ann_files]
    ann_dirs = set(ann_dirs)

    for ann_dir in ann_dirs:
        archive_path = os.path.join(labels_path, ann_dir)
        Path(archive_path).mkdir(parents=True, exist_ok=True)

        with open(os.path.join(archive_path, "obj.data"), "w") as f:
            if dataset_type == COCO:
                f.write("classes = 80\n")
            elif dataset_type == CROWDHUMAN:
                f.write("classes = 2\n")
            f.write("names = data/obj.names\n")
            f.write("train = data/train.txt")

        with open(os.path.join(archive_path, "obj.names"), "w") as f:
            if dataset_type == COCO:
                for value in COCO_CLASSES.values():
                    f.write(f"{value}\n")
            elif dataset_type == CROWDHUMAN:
                for value in CROWDHUMAN_CLASSES.values():
                    f.write(f"{value}\n")

        data_path = os.path.join(archive_path, "obj_train_data")
        Path(data_path).mkdir(parents=True, exist_ok=True)
        cur_ann_files = [f for f in ann_files if f.startswith(ann_dir)]
        frame_idxs = []
        for f in cur_ann_files:
            old_name = os.path.join(labels_path, f)
            print(old_name)
            frame_idx = int(os.path.splitext(f.rpartition("_")[-1])[0]) - 1
            frame_idxs.append(frame_idx)
            new_name = os.path.join(data_path, f"frame_{frame_idx:06d}.txt")
            if not os.path.exists(new_name):
                shutil.copy(old_name, new_name)

        with open(os.path.join(archive_path, "train.txt"), "w") as f:
            for frame_idx in sorted(frame_idxs):
                f.write(f"data/obj_train_data/frame_{frame_idx:06d}.PNG\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels_path", type=str, help="path to folder containing yolo labels"
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="coco",
        help="type of dataset: coco, crowdhuman",
    )
    args = parser.parse_args()
    main(args)
