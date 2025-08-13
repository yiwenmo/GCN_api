"""
Image Processing Script
Author: MO, YI-WEN

This script processes several files such as json and pt.
It performs the following operations:
1. Reads .pt and JSON files from specified directories
2. Processes images to extract bounding box and segmentation information
3. Calculates intersections between bounding boxes and segmentations
4. Generates new JSON files with updated information
5. Optionally saves processed images

Usage:
python image_processing.py [arguments]

Arguments:
--img_dir: Directory containing .pt and .json files (default: "data")
--txt_dir: Directory containing .txt files (default: "Label")
--save_path: Output directory for processed files (default: "./data")
--workers: Number of worker processes for parallel processing (default: 1)

For more information, use: python image_processing.py --help
"""

import os
import torch
import cv2
import math
import time
import json
import numpy as np
import argparse
import logging
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Get the width and height of the image
def get_image_dimensions(pt_path):
    """
    Get the dimensions of the image from the pt file
    
    Args:
        pt_path (str): Path to the .pt file
        
    Returns:
        tuple: (width, height) of the image
    """
    try:
        # read the pt file
        pano_id = torch.load(pt_path, map_location=torch.device('cpu'))
        # get the dimensions
        height, width = pano_id.shape
        return width, height
    except Exception as e:
        logging.error(f"Error getting dimensions from {pt_path}: {str(e)}")
        return None, None

def preprocess_txt_files(input_dir, output_dir):
    """
    Preprocess .txt files by adding a label column based on the score value.
    Rules:
    - First row always gets label 0
    - second row always gets label 1
    - If only one bbox exists, add a fake bbox with label 1
    - For other rows:
        - If score >= 0.6: label = 1
        - If score < 0.6: label = 0
        - Special case: If score = 0.0: label = 0
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all .txt files in input directory
    txt_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
    
    for txt_file in txt_files:
        input_path = os.path.join(input_dir, txt_file)
        output_path = os.path.join(output_dir, txt_file)
        
        try:
            # Read and process the file
            with open(input_path, 'r') as f:
                lines = f.readlines()
            
            processed_lines = []

            # for line in lines:
            #     # Split the line into values
            #     values = line.strip().split()
            #     if len(values) >= 6:  # Ensure we have enough values
            #         score = float(values[5])
            #         # Apply the labeling rules
            #         label = 1 if (score >= 0.6 or score == 0.0) else 0
            #         # Add the label to the line
            #         new_line = f"{' '.join(values)} {label}\n"
            #         processed_lines.append(new_line)
            
            # Check if we need to add a fake bbox
            if len(lines) == 1:
                # Process the single real bbox
                values = lines[0].strip().split()
                if len(values) >= 6:
                    # Add original bbox with label 1
                    processed_lines.append(f"{' '.join(values[:6])} 1\n")
                    
                    # Add fake bbox with label 0
                    # Using slightly modified coordinates and lower confidence score
                    fake_bbox = [
                        values[0],  # same class_id
                        "0.5000",   # center_x
                        "0.5000",   # center_y
                        "0.2000",   # width
                        "0.2000",   # height
                        "0.3000"    # lower confidence score
                    ]
                    processed_lines.append(f"{' '.join(fake_bbox)} 0\n")
            else:
                # Process multiple bboxes normally
                for i, line in enumerate(lines):
                    values = line.strip().split()
                    if len(values) >= 6:
                        if i == 0:  # First row gets label 1
                            label = 1
                        elif i == 1:  # Second row gets label 0
                            label = 0
                        else:  # Other rows follow the original rules
                            score = float(values[5])
                            label = 1 if (score >= 0.6 or score == 0.0) else 0
                        
                        processed_lines.append(f"{' '.join(values[:6])} {label}\n")
            # Write the processed lines to the new file
            with open(output_path, 'w') as f:
                f.writelines(processed_lines)
                
            logging.info(f"Processed {txt_file} successfully")
            
        except Exception as e:
            logging.error(f"Error processing {txt_file}: {str(e)}")

def read_arcade_result(result_path, image_width, image_height):
    """
    Read and process arcade result file using actual image dimensions
    """
    records = []
    with open(result_path, "r") as file:
        for line in file:
            line = line.strip()
            record = [float(value) for value in line.split()]
            records.append(record)
    records = sort_records_and_calculate_area(records, image_width, image_height)
    return records

def sort_records_and_calculate_area(records, image_width, image_height):
    sorted_records = sorted(records, key=lambda x: x[1])
    output_records = []
    for record in sorted_records:
        bbox_area = record[3] * image_width * record[4] * image_height
        output_record = {
            'class': record[0],
            'bbox_center_x': record[1],
            'bbox_center_y': record[2],
            'bbox_width': record[3],
            'bbox_height': record[4],
            'bbox_area': bbox_area,
            'score': record[5],
            # 'label': record[6],
            'label': 1 if record[5] >= 0.6 else 0
        }

        # label = 1 if record[5] >= 0.6 else 0
        if len(record) > 6:
            output_record['label'] = record[6]
        elif record[5] >= 0.6:
            output_record['label'] = 1
        else:
            output_record['label'] = 0
            
        output_records.append(output_record)
    return output_records

def id2rgb(id_map):
    # covert id to rgb
    if isinstance(id_map, np.ndarray):
        id_map_copy = id_map.copy()
        rgb_shape = tuple(list(id_map.shape) + [3])
        rgb_map = np.zeros(rgb_shape, dtype=np.uint8)
        for i in range(3):
            rgb_map[..., i] = id_map_copy % 256
            # id_map_copy //= 256
            id_map_copy = id_map_copy // 256
        return rgb_map
    color = []
    for _ in range(3):
        color.append(id_map % 256)
        # id_map //= 256
        id_map = id_map // 256
    return color

def rgb2id(color):
    # covert rgb to id
    if isinstance(color, np.ndarray) and len(color.shape) == 3:
        if color.dtype == np.uint8:
            color = color.astype(np.int32)
        return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
    return int(color[0] + 256 * color[1] + 256 * 256 * color[2])

def read_seg_json(json_path):
    # read seg json file
    with open(json_path, 'r') as f:
        seg = json.load(f)
    for item in seg:
        item['category_id'] += 1 # 在原始json，catergory_id都比設定少1
    return seg

def read_tensor_pt(pt_path):
    # read seg pt file
    pano_id = torch.load(pt_path, map_location=torch.device('cpu')).numpy().astype(np.uint8)
    return pano_id

def pt2img(pt_path):
    # covert pt file to image
    pano_id = read_tensor_pt(pt_path)
    rgb_map = id2rgb(pano_id)
    bgr_map = cv2.cvtColor(rgb_map, cv2.COLOR_RGB2BGR)
    return bgr_map

def seg_info(bbox_intersected_region, seg, id_seg_dict, records):
    # calculate the intersection of a bbox with seg info
    collected_seg = []
    rgb_image = cv2.cvtColor(bbox_intersected_region, cv2.COLOR_BGR2RGB)
    id_image = rgb2id(rgb_image).astype(np.uint8)
    unique, counts = np.unique(id_image, return_counts=True)
    seg_dict = dict(zip(unique, counts))
    for subseg in seg:
        if subseg['id'] in unique:
            temp = subseg.copy()
            temp['intersection'] = int(seg_dict[temp['id']])
            # temp['seg_area'] = int(id_seg_dict[temp['id']])  # 在output的json file有多的
            seg_percent = int(seg_dict[subseg['id']]) / int(id_seg_dict[subseg['id']])
            temp['seg_percent'] = seg_percent

            # get corresponding bbox_area
            bbox_area = None
            for record in records:
                if record['bbox_area'] is not None and record['bbox_area'] > 0:
                    bbox_area = record['bbox_area']
                    break
            
            # Calculate the percentage of each category in the corresponding bounding box
            if bbox_area is not None:
                # calculate bbox_percent
                temp['bbox_percent'] = temp['intersection'] / bbox_area
            else:
                temp['bbox_percent'] = 0.0
            
            collected_seg.append(temp)
    return collected_seg

def bbox2mask(seg, pano_id, image, records, image_width, image_height):
    # Calculate the intersection area of bbox and seg, and output list
    unique_id, id_counts = np.unique(pano_id, return_counts=True)
    id_seg_dict = dict(zip(unique_id, id_counts))

    mask = np.zeros_like(image)
    bbox_info = []
    for record in records:
        bbox_mask = np.zeros_like(image)
        x_center = record['bbox_center_x'] * image_width
        y_center = record['bbox_center_y'] * image_height
        width = record['bbox_width'] * image_width
        height = record['bbox_height'] * image_height
        
        # Top-left and bottom-right corners of each bbox
        x_min = int(x_center - width / 2)
        y_min = int(y_center - height / 2)
        x_max = int(x_center + width / 2)
        y_max = int(y_center + height / 2)

        # Draw each bbox_mask
        cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), (255, 255, 255), -1)
        cv2.rectangle(bbox_mask, (x_min, y_min), (x_max, y_max), (255, 255, 255), -1)

        # Draw the outline of each bbox
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 255, 255), 1)
        
        # Draw the intersection area of each bbox
        bbox_intersected_region = cv2.bitwise_and(image, bbox_mask)

        # Process the image id and mapping category_id for each bbox
        if bbox_intersected_region is not None:
            record['seg'] = seg_info(bbox_intersected_region, seg, id_seg_dict, records)
            bbox_info.append(record)
    intersected_region = cv2.bitwise_and(image, mask)    
    return [intersected_region, bbox_info]


def extract_by_id(intersected_region, image):
    # Extract all intersected seg blocks
    colors = np.unique(intersected_region.reshape(-1, 3), axis=0)
    mask = np.zeros_like(image, dtype=np.uint8)
    for color in colors:
        color_mask = np.all(image == color, axis=2)
        mask[color_mask] = (255, 255, 255)
    extracted_region = cv2.bitwise_and(image, mask)
    return extracted_region

def bboxinfo2json(image_name, bbox_info, save=False, save_path='./'):
    # Write the intersection seg information to json
    to_json = dict()
    to_json['image'] = image_name
    to_json['bbox_info'] = bbox_info
    if save is True:
        with open(os.path.join(save_path, image_name+'_bbox_info.json'), 'w') as file:
            json.dump(to_json, file)
    return to_json

def view_result(image, image_name, intersected_region, extracted_region, save=False, save_path='./'):
    if save is True:
        cv2.imwrite(os.path.join(save_path, image_name+'_with_bbox.png'), image)
        cv2.imwrite(os.path.join(save_path, image_name+'_intersected_with_bbox.png'), intersected_region)
        cv2.imwrite(os.path.join(save_path, image_name+'_extracted_by_bgr.png'), extracted_region)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()
    # cv2.waitKey(3000) # Line for macOS

def pt_centroid(pt_path):
    # Convert pt to array
    pano_id_array = torch.load(pt_path, map_location=torch.device('cpu'))
    data_np = pano_id_array.numpy()
    # Get all ids in the 640*640 numpy array
    unique_ids = np.unique(data_np)
    # print(unique_ids)
    # create a dict for saving the sum of location and number of pixel for each object
    object_positions = {}
    # for loop objects
    for object_id in unique_ids:
        # ignore the background
        if object_id == 0:
            continue
        
        # get all the pixel with same id
        pixels_x, pixels_y = np.where(data_np == object_id)
        
        # calculate the mean value for centroid
        centroid_x = int(np.mean(pixels_y))
        centroid_y = int(np.mean(pixels_x))
        # print(centroid_x, centroid_y)
        # save
        object_positions[object_id] = (centroid_x, centroid_y)

    return object_positions


def process_jsons(input_dir, output_dir, image_dimensions):
    """
    Process JSON files using actual image dimensions
    """
    img_files = [os.path.join(input_dir, filename) 
            for filename in os.listdir(input_dir) 
            if filename.endswith('_bbox_info.json')]

    
    for img_file in img_files:
        with open(img_file, 'r') as json_file:
            data = json.load(json_file)
        
        img_name = os.path.splitext(os.path.basename(img_file))[0]

        # handle _bbox_info
        parts = img_name.split('_bbox_info')
        img_name_without_bbox_info = parts[0] + parts[1]  # if parts[1] is empty, it will be ''
        # pt_centroid_dict = pt_centroid(f'./data/{img_name_without_bbox_info}.pt') # input pt file
        
        # Get actual image dimensions from the dictionary
        image_width, image_height = image_dimensions.get(img_name_without_bbox_info, (None, None))
        if image_width is None or image_height is None:
            logging.error(f"Could not find dimensions for {img_name_without_bbox_info}")
            continue

        pt_file_path = os.path.join(input_dir, f'{img_name_without_bbox_info}.pt')
        pt_centroid_dict = pt_centroid(pt_file_path)  # use path

        # 更新json
        for bbox_info in data['bbox_info']:
            bbox_center_x = bbox_info['bbox_center_x']
            bbox_center_y = bbox_info['bbox_center_y']

            for seg_item in bbox_info['seg']:
                id = seg_item['id']
                centroid_x, centroid_y = pt_centroid_dict.get(id, (None, None))
                
                if centroid_x is not None and centroid_y is not None:
                    center_x = bbox_center_x * image_width  # image_width = 640
                    center_y = bbox_center_y * image_height  # image_height =640
                    distance = math.sqrt((centroid_x - center_x) ** 2 + (centroid_y - center_y) ** 2)
                    
                    seg_item['distance'] = distance
                    seg_item['centroid_x'] = centroid_x/image_width
                    seg_item['centroid_y'] = centroid_y/image_height
                else:
                    seg_item['distance'] = None
                    seg_item['centroid_x'] = None
                    seg_item['centroid_y'] = None

        output_path = os.path.join(output_dir, f'{img_name_without_bbox_info}_bbox_info_updated.json')
        with open(output_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)


def process_single_image(img_file, img_dir, bbox_dir, save_path):
    """Process a single image file with automatic dimension detection"""
    img_name = os.path.splitext(os.path.basename(img_file))[0]
    try:
        start_time = time.time()

        # get the dimensions of the image
        image_width, image_height = get_image_dimensions(img_file)
        if image_width is None or image_height is None:
            raise ValueError(f"Failed to get image dimensions for {img_file}")
        
        logging.info(f"Processing {img_name} with dimensions {image_width}x{image_height}")
        seg = read_seg_json(os.path.join(img_dir, img_name + '.json'))
        pano_id = read_tensor_pt(img_file)
        image = pt2img(img_file)
        records = read_arcade_result(os.path.join(bbox_dir, img_name + '.txt'), image_width, image_height)
        intersected_region, bbox_info = bbox2mask(seg, pano_id, image, records, image_width, image_height)
        extracted_region = extract_by_id(intersected_region, image)
        bbox_info_json = bboxinfo2json(img_name, bbox_info, save=True, save_path=save_path)
        
        process_time = time.time() - start_time
        logging.info(f"{img_file} processing time: {process_time:.2f} sec")
        return img_name, True, (image_width, image_height)
    except Exception as e:
        logging.error(f"Error processing {img_file}: {str(e)}")
        return img_name, False, (None, None)

def process_images(img_dir, bbox_dir, save_path, num_workers=1):
    """Process multiple images using multiprocessing"""
    img_files = [os.path.join(img_dir, filename) for filename in os.listdir(img_dir) if filename.endswith('.pt')]
    logging.info(f"Start processing {len(img_files)} image files...")

    image_dimensions = {}

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_single_image, img_file, img_dir, bbox_dir, save_path) for img_file in img_files]
        
        successful = 0
        failed = 0
        for future in as_completed(futures):
            img_name, success, dimensions = future.result()
            if success:
                successful += 1
                image_dimensions[img_name] = dimensions
            else:
                failed += 1
    
    logging.info(f"Image processing completed. Successful: {successful}, Failed: {failed}")
    return image_dimensions

def main(args):
    # Set parameters
    img_dir = args.img_dir
    txt_dir = args.txt_dir
    save_path = args.save_path
    num_workers = args.workers

    # Ensure output directory exists
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(txt_dir, exist_ok=True)

    # Preprocess txt files
    logging.info("Preprocessing txt files...")
    preprocess_txt_files(img_dir, txt_dir)

    # Process images and get image dimensions
    logging.info("Processing images...")
    image_dimensions = process_images(img_dir, txt_dir, save_path, num_workers)

    # Process JSON files
    logging.info("Proccessing json files...")
    process_jsons(save_path, save_path, image_dimensions)

    logging.info("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image preprocessing script")
    parser.add_argument("--img_dir", default="data", help="Directory contains .pt and .json")
    parser.add_argument("--txt_dir", default="Label", help="Directory contains .txt")
    parser.add_argument("--save_path", default="./data", help="output directory")
    parser.add_argument("--workers", type=int, default=1, help="number of workers")
    args = parser.parse_args()

    main(args)