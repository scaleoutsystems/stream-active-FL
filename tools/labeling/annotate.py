import os
import json
import cv2
from torch.utils.data import Dataset
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

from zod import ZodSequences
from zod.visualization.lidar_on_image import visualize_lidar_on_image
from tqdm import tqdm


cfg = get_cfg()

cfg.merge_from_file(
    model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
)
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
)

cfg.MODEL.DEVICE = "cuda"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4

predictor = DefaultPredictor(cfg)


class ZOD(Dataset):
    def __init__(self, dataset_root, version="full", transform=None):
        self.transform = transform
        self.sequences = ZodSequences(dataset_root, version)

        sequence_frames_mapping = {}
        for sequence in self.sequences:
            sequence_id = sequence.info.id
            sequence_frames = sequence.info.get_camera_frames()
            sequence_frames_mapping[sequence_id] = sequence_frames

        self.sequence_list = list(sequence_frames_mapping.items())

    def __len__(self):
        return len(self.sequence_list)  # number of sequences

    def __getitem__(self, index):
        if isinstance(index, tuple):
            seq_idx, frame_idx = index
            seq_id, sequence_frames = self.sequence_list[seq_idx]
            frame = sequence_frames[frame_idx]
            # read image
            img = frame.read()
            if self.transform:
                img = self.transform(img)
            return img
        else:
            # return all frames in sequence
            seq_id, sequence_frames = self.sequence_list[index]
            return sequence_frames

    def get_stream(self, sequence_idx, frame_idx, n_frames_before, n_frames_after):
        """Get temporal window of frames around a specific frame"""
        seq_id, sequence_frames = self.sequence_list[sequence_idx]

        start_idx = max(0, frame_idx - n_frames_before)
        end_idx = min(len(sequence_frames), frame_idx + n_frames_after + 1)

        stream_frames = []
        for idx in range(start_idx, end_idx):
            img = self[sequence_idx, idx]
            stream_frames.append(img)

        return stream_frames


def label_sequences(dataset_root, directory="annotations"):
    dataset = ZOD(dataset_root, version="full")

    os.makedirs(directory, exist_ok=True)

    # change detectron object ids
    COCO_TO_CUSTOM = {
        0: 0,  # person -> person
        2: 1,  # car -> car
        9: 2,  # traffic lights
    }

    for seq_idx, sequence in enumerate(tqdm(dataset, desc="Processing sequences")):
        seq_id, _ = dataset.sequence_list[seq_idx]

        output_file = f"{directory}/{seq_id}.json"
        if os.path.exists(output_file):
            print(f"Skipping sequence {seq_id} (already processed)")
            continue

        annotations = {
            "images": [],
            "annotations": [],
            "categories": [{"id": 0, "name": "person"}, {"id": 1, "name": "car"}, {"id": 2, "name": "traffic_light"}],
        }

        img_id = 0
        ann_id = 0

        for frame_idx, frame in enumerate(sequence):
            filepath = frame.filepath
            img = cv2.imread(filepath)
            if img is None:
                print("file doesn't exist, skipping image")
                continue

            h, w = frame.height, frame.width

            outputs = predictor(img)

            boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
            scores = outputs["instances"].scores.cpu().numpy()
            classes = outputs["instances"].pred_classes.cpu().numpy()

            # Add image metadata with unique filename
            annotations["images"].append(
                {
                    "id": img_id,
                    "file_name": os.path.basename(filepath),
                    "frame_idx": frame_idx,
                    "width": w,
                    "height": h,
                }
            )

            # Add bounding boxes
            for box, cls, score in zip(boxes, classes, scores):
                if int(cls) not in COCO_TO_CUSTOM:
                    continue

                custom_cls = COCO_TO_CUSTOM[int(cls)]
                x1, y1, x2, y2 = box

                annotations["annotations"].append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "frame_idx": frame_idx,
                        "category_id": custom_cls,
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": float(score),
                        "area": float((x2 - x1) * (y2 - y1)),
                        "iscrowd": 0,
                    }
                )
                ann_id += 1

            img_id += 1

        # Save one file per sequence
        output_file = f"{directory}/{seq_id}.json"
        with open(output_file, "w") as f:
            json.dump(annotations, f, indent=4)

        print(f"Saved sequence {seq_idx}: {img_id} images, {ann_id} annotations")


if __name__ == "__main__":
    dataset_root = "/mnt/ZOD_clone_2018_scaleout_zenseact/" # path to original ZOD images
    # dataset_root = "/mnt/pr_2018_scaleout_workdir/ZODCropped/" # path to downsampled + cropped images

    output_directory = "../../data/annotations_original_imgs"

    label_sequences(dataset_root, directory=output_directory)
