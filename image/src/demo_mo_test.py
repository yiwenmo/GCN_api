# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import multiprocessing as mp
import os

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import tempfile
import time
import warnings

import cv2
import numpy as np
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from mask2former import add_maskformer2_config
from predictor import VisualizationDemo


# constants
WINDOW_NAME = "mask2former demo"

# 20221113 edit

# reserve
import copy
import json
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json, load_sem_seg
# from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from detectron2.utils.file_io import PathManager

def get_dicts(file_dir):
      with open(file_dir) as f:
          dicts = json.load(f)
      return(dicts)

def get_metadata(categories_json_dir):
    
    COCO_CATEGORIES = get_dicts(categories_json_dir)
    
    meta = {}
    
    # The following metadata maps contiguous id from [0, #thing categories +
    # #stuff categories) to their names and colors. We have to replica of the
    # same name and color under "thing_*" and "stuff_*" because the current
    # visualization function in D2 handles thing and class classes differently
    # due to some heuristic used in Panoptic FPN. We keep the same naming to
    # enable reusing existing visualization functions.
    thing_classes = [k["name"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    stuff_classes = [k["name"] for k in COCO_CATEGORIES]
    stuff_colors = [k["color"] for k in COCO_CATEGORIES]

    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors

    # Convert category id for training:
    #   category id: like semantic segmentation, it is the class id for each
    #   pixel. Since there are some classes not used in evaluation, the category
    #   id is not always contiguous and thus we have two set of category ids:
    #       - original category id: category id in the original dataset, mainly
    #           used for evaluation.
    #       - contiguous category id: [0, #classes), in order to train the linear
    #           softmax classifier.
    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}

    for i, cat in enumerate(COCO_CATEGORIES):
        if cat["isthing"]:
            thing_dataset_id_to_contiguous_id[cat["id"]] = i
        # else:
        #     stuff_dataset_id_to_contiguous_id[cat["id"]] = i

        # in order to use sem_seg evaluator
        stuff_dataset_id_to_contiguous_id[cat["id"]] = i

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

    return meta

def load_coco_panoptic_json(image_dir, png_dir, ann_json, meta):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/coco/train2017".
        gt_dir (str): path to the raw annotations. e.g., "~/coco/panoptic_train2017".
        json_file (str): path to the json file. e.g., "~/coco/annotations/panoptic_train2017.json".
    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """

    def _convert_category_id(segment_info, meta):
        if segment_info["category_id"] in meta["thing_dataset_id_to_contiguous_id"]:
            segment_info["category_id"] = meta["thing_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
            segment_info["isthing"] = True
        else:
            segment_info["category_id"] = meta["stuff_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
            segment_info["isthing"] = False
        return segment_info

    with PathManager.open(ann_json) as f:
        json_info = json.load(f)

    ret = []
    for i in range(len(json_info["annotations"])):
        ann = json_info["annotations"][i]
        img = json_info["images"][i]
        if ann["image_id"] == img["id"]:
            image_id = int(ann["image_id"])
            image_file = os.path.join(image_dir, os.path.splitext(ann["file_name"])[0] + ".png")
            label_file = os.path.join(png_dir, ann["file_name"])
            segments_info = [_convert_category_id(x, meta) for x in ann["segments_info"]]
            im_width = int(img["width"])
            im_height = int(img["height"])         
            ret.append(
                {
                    "file_name": image_file,
                    "image_id": image_id,
                    "height": im_height,
                    "width": im_width,
                    "pan_seg_file_name": label_file,
                    "segments_info": segments_info,
                }
            )
    # for ann in json_info["annotations"]:
    #     image_id = int(ann["image_id"])
    #     # TODO: currently we assume image and label has the same filename but
    #     # different extension, and images have extension ".jpg" for COCO. Need
    #     # to make image extension a user-provided argument if we extend this
    #     # function to support other COCO-like datasets.
    #     image_file = os.path.join(image_dir, os.path.splitext(ann["file_name"])[0] + ".png")
    #     label_file = os.path.join(gt_dir, ann["file_name"])
    #     # sem_label_file = os.path.join(semseg_dir, ann["file_name"])
    #     segments_info = [_convert_category_id(x, meta) for x in ann["segments_info"]]
    #     ret.append(
    #         {
    #             "file_name": image_file,
    #             "image_id": image_id,
    #             "pan_seg_file_name": label_file,
    #             # "sem_seg_file_name": sem_label_file,
    #             "segments_info": segments_info,
    #         }
    #     )
    assert len(ret), f"No images found in {image_dir}!"
    assert PathManager.isfile(ret[0]["file_name"]), ret[0]["file_name"]
    assert PathManager.isfile(ret[0]["pan_seg_file_name"]), ret[0]["pan_seg_file_name"]
    # assert PathManager.isfile(ret[0]["sem_seg_file_name"]), ret[0]["sem_seg_file_name"]
    return ret

def register_coco_panoptic_annos_sem_seg(dataset_name, image_root, png_root, ann_json, metadata):
    panoptic_name = dataset_name
    # delattr(MetadataCatalog.get(panoptic_name), "thing_classes")
    # delattr(MetadataCatalog.get(panoptic_name), "thing_colors")
    # MetadataCatalog.get(panoptic_name).set(
    #     thing_classes=metadata["thing_classes"],
    #     thing_colors=metadata["thing_colors"],
        # thing_dataset_id_to_contiguous_id=metadata["thing_dataset_id_to_contiguous_id"],
    # )

    # the name is "coco_2017_train_panoptic_with_sem_seg" and "coco_2017_val_panoptic_with_sem_seg"
    # semantic_name = name + "_with_sem_seg"
    semantic_name = panoptic_name
    DatasetCatalog.register(
        semantic_name,
        lambda: load_coco_panoptic_json(image_root, png_root, ann_json, metadata),
    )
    MetadataCatalog.get(semantic_name).set(
        # sem_seg_root=sem_seg_root,
        panoptic_root=png_root,
        image_root=image_root,
        panoptic_json=ann_json,
        # json_file=instances_json,
        evaluator_type="coco_panoptic_seg",
        ignore_label=255,
        label_divisor=1000,
        **metadata,
    )

gsv_data_dir = '/workspace/Mask2Former/data/gsv_data'
# gsv_data_dir = '/content/drive/MyDrive/Mask2Former/data/gsv_data'
ann_json_dir= gsv_data_dir + '/pano'
im_dir= gsv_data_dir + '/gsv'
pano_dir= gsv_data_dir + '/pano'
cat_json= gsv_data_dir + '/categories.json'
for d in ['train', 'val', 'test']:
    pano_png_dir = pano_dir + '/' + d
    pano_json_file = ann_json_dir + '/' + d + '.json'
    dataset_name = 'gsv_' + d
    # get_meta = get_metadata(cat_json)
    # load_coco_panoptic_json(im_dir, pano_png_dir, pano_json_file, get_meta
    # print(dataset_name, get_meta, im_dir, pano_png_dir, pano_json_file)
    register_coco_panoptic_annos_sem_seg(dataset_name, im_dir, pano_png_dir, pano_json_file, get_metadata(cat_json))

# add end line

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="maskformer2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg)

    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            print('line301', segments_info)

            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )
            # 20221202 add
            def save_tensor_seg(pre_tensor, pre_seg_info, save_path, image_name):
                tensor_path = save_path
                tensor_name = image_name.split('.')[0]+'.pt'
                tensor_save = tensor_path + '/' + tensor_name
                import torch
                torch.save(pre_tensor, tensor_save)
                print('OK!', tensor_save)
                seg_path = save_path
                seg_name = image_name.split('.')[0]+'.json'
                seg_save = seg_path + '/' + seg_name
                print('OK!', seg_save)
                with open(seg_save, "w") as f:
                    f.write(json.dumps(pre_seg_info))
            # add end

            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                    save = save_tensor_seg(panoptic_seg, segments_info, args.output, os.path.basename(path)) # 20221202 add
                    # print(save)
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                    save = save_tensor_seg(panoptic_seg, segments_info, os.path.dirname(args.output), os.path.basename(path)) # 20221202 add
                    # print(save)
                visualized_output.save(out_filename)
            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit
    # elif args.webcam:
    #     assert args.input is None, "Cannot have both --input and --webcam!"
    #     assert args.output is None, "output not yet supported with --webcam!"
    #     cam = cv2.VideoCapture(0)
    #     for vis in tqdm.tqdm(demo.run_on_video(cam)):
    #         cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    #         cv2.imshow(WINDOW_NAME, vis)
    #         if cv2.waitKey(1) == 27:
    #             break  # esc to quit
    #     cam.release()
    #     cv2.destroyAllWindows()
    # elif args.video_input:
    #     video = cv2.VideoCapture(args.video_input)
    #     width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    #     height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #     frames_per_second = video.get(cv2.CAP_PROP_FPS)
    #     num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    #     basename = os.path.basename(args.video_input)
    #     codec, file_ext = (
    #         ("x264", ".mkv") if test_opencv_video_format("x264", ".mkv") else ("mp4v", ".mp4")
    #     )
    #     if codec == ".mp4v":
    #         warnings.warn("x264 codec not available, switching to mp4v")
    #     if args.output:
    #         if os.path.isdir(args.output):
    #             output_fname = os.path.join(args.output, basename)
    #             output_fname = os.path.splitext(output_fname)[0] + file_ext
    #         else:
    #             output_fname = args.output
    #         assert not os.path.isfile(output_fname), output_fname
    #         output_file = cv2.VideoWriter(
    #             filename=output_fname,
    #             # some installation of opencv may not support x264 (due to its license),
    #             # you can try other format (e.g. MPEG)
    #             fourcc=cv2.VideoWriter_fourcc(*codec),
    #             fps=float(frames_per_second),
    #             frameSize=(width, height),
    #             isColor=True,
    #         )
    #     assert os.path.isfile(args.video_input)
    #     for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
    #         if args.output:
    #             output_file.write(vis_frame)
    #         else:
    #             cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
    #             cv2.imshow(basename, vis_frame)
    #             if cv2.waitKey(1) == 27:
    #                 break  # esc to quit
    #     video.release()
    #     if args.output:
    #         output_file.release()
    #     else:
    #         cv2.destroyAllWindows()
