"""
Arcade Data Processing Script
Author: MO, YI-WEN

This script processes JSON files containing object detection and segmentation data,
and generates various CSV and TXT files for use in graph-based tasks.

Key functionalities:
1. Convert JSON data to CSV format (graphs.csv)
2. Process adjacency information (graphs_adj.csv)
3. Generate node information (nodes_without_adj.csv, nodes.csv)
4. Merge graph and node data (merged_data.csv)
5. Perform additional processing steps for generating Arcade_v3 directory and corresponding txt files:
   - Filter and encode data
   - Generate node labels and attributes
   - Process adjacency and create edge attributes
   - Convert graph to bi-directional representation
   - One-hot encode node and edge attributes

Usage:
python json2txt.py --input_dir <input_directory> --output_dir <output_directory> --final_output <final_output_directory>
"""

import os
import re
import json
import torch
import csv
import pandas as pd
import numpy as np
import warnings
import argparse


# bbox image coordinates and adjacency between two bboxes
def calculate_coordinates(bbox_center_x, bbox_center_y, bbox_width, bbox_height, image_width, image_height, graph_id):
    x1 = int((bbox_center_x - 0.5 * bbox_width) * image_width)
    y1 = int((bbox_center_y - 0.5 * bbox_height) * image_height)
    x2 = int((bbox_center_x + 0.5 * bbox_width) * image_width)
    y2 = int((bbox_center_y + 0.5 * bbox_height) * image_height)
    return x1, y1, x2, y2, graph_id

def are_adjacent(bbox1, bbox2, threshold):
    x1_min, y1_min, x1_max, y1_max, graph_id1 = bbox1
    x2_min, y2_min, x2_max, y2_max, graph_id2 = bbox2
   
    intersection_area = max(0, min(x1_max, x2_max) - max(x1_min, x2_min)) * max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    horizontal_dist = abs(x2_min - x1_max)
   
    if intersection_area > 0 or horizontal_dist <= threshold:
        return True, graph_id1, graph_id2
    else:
        return False, None, None


# json2csv
# graphs.csv
def process_json_to_graphscsv(input_folder, output_folder, output_filename):
    try:
        # Create an empty list to store all data
        all_data = []
        # Initialize graph_id
        graph_id = 1
        # for loop for json file
        json_files = [f for f in os.listdir(input_folder) if f.endswith('bbox_info_updated.json')]
        
        if not json_files:
            print(f"No JSON files found in {input_folder}")
            return False

        for filename in json_files:
            json_file_path = os.path.join(input_folder, filename)
            print(f"Processing file: {json_file_path}")  # Debug print
            # read each file in output
            with open(json_file_path, 'r') as json_file:
                data = json.load(json_file)
            image_id = data['image']
            image_counts = 1
            # get each bbox information and set into list
            for bbox_info in data['bbox_info']:
                if not bbox_info.get('seg'):
                    continue # skip bbox without seg
               
                row = {
                    'image_id': f"{image_id}_{image_counts}",
                    'graph_id': graph_id,
                    'bbox_center_x': bbox_info['bbox_center_x'],
                    'bbox_center_y': bbox_info['bbox_center_y'],
                    'bbox_width': bbox_info['bbox_width'],
                    'bbox_height': bbox_info['bbox_height'],
                    'bbox_area': bbox_info['bbox_area'],
                    'score': bbox_info['score'],
                    'label': bbox_info['label']
                }
                image_counts += 1
                all_data.append(row)
                graph_id += 1
        
        if not all_data:
            print(f"No valid data found in JSON files in {input_folder}")
            return False
           
        # create dataframe
        df = pd.DataFrame(all_data)
        # save dataframe as csv
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, output_filename)
        df.to_csv(output_path, index=False)
        print(f'Data has been written to {output_filename} with graph_id')
        print(f'Number of rows written: {len(df)}')  # Debug print
        return True
    except Exception as e:
        print(f"An error occurred in process_json_to_graphscsv: {str(e)}")
        print(traceback.format_exc())  # This will print the full traceback
        return False


# add distance threshold
def process_graph_adj(input_file, output_file):
    try:
        data = pd.read_csv(input_file)
        if data.empty:
            print(f"The input file {input_file} is empty.")
            return False
            
        image_width = 640
        image_height = 640
        data['adjacency'] = 0
        image_groups = {}
        
        for image_id in data['image_id']:
            key = '$'.join(image_id.split('$')[:2])
            if key not in image_groups:
                image_groups[key] = []
            image_groups[key].append(image_id)
        
        for image_id_group in image_groups.values():
            image_data = data[data['image_id'].isin(image_id_group)]
            
            # determine distance
            try:
                distance_str = image_data['image_id'].apply(lambda x: re.sub(r'_\d+$', '', x.split('$')[2]))
                distance = float(distance_str.iloc[0])
            except (ValueError, IndexError):
                # if the distance cannot be determined, set it to a default value
                distance = 5
                
            threshold = 125.4465301465138 if distance <= 5 else 87.02154489564495
            
            for i in range(len(image_data)):
                bbox1 = calculate_coordinates(image_data.iloc[i]["bbox_center_x"], 
                                           image_data.iloc[i]["bbox_center_y"],
                                           image_data.iloc[i]["bbox_width"], 
                                           image_data.iloc[i]["bbox_height"],
                                           image_width, 
                                           image_height, 
                                           image_data.iloc[i]["graph_id"])
                                           
                for j in range(i + 1, len(image_data)):
                    bbox2 = calculate_coordinates(image_data.iloc[j]["bbox_center_x"], 
                                               image_data.iloc[j]["bbox_center_y"],
                                               image_data.iloc[j]["bbox_width"], 
                                               image_data.iloc[j]["bbox_height"],
                                               image_width, 
                                               image_height, 
                                               image_data.iloc[j]["graph_id"])
                    
                    adjacent, graph_id1, graph_id2 = are_adjacent(bbox1, bbox2, threshold)
                    
                    if adjacent:
                        data.loc[(data['image_id'].isin(image_id_group)) & 
                                ((data['graph_id'] == graph_id1) | 
                                 (data['graph_id'] == graph_id2)), 'adjacency'] = 1
        
        data.to_csv(output_file, index=False)
        print(f'Data has been written to {output_file} with adjacency information')
        print(f'Number of rows written: {len(data)}')
        return True
        
    except Exception as e:
        print(f"An error occurred in process_adjacency: {str(e)}")
        print(traceback.format_exc())
        return False

# nodes_without_adj.csv
def generate_nodes_without_adj(input_folder, output_folder):
    try:
        json_files = [f for f in os.listdir(input_folder) if f.endswith('bbox_info_updated.json')]
        
        if not json_files:
            print(f"No JSON files found in {input_folder}")
            return False

        fieldnames = ['image_id', 'bbox_id', 'node_id', 'seg_id', 'category_id', 'id', 'intersection', 'seg_area', 'seg_percent', 'bbox_percent', 'distance', 'centroid_x', 'centroid_y']
        
        nodes_without_adj_path = os.path.join(output_folder, 'nodes_without_adj.csv')
        
        with open(nodes_without_adj_path, 'w', newline='') as nodes_without_adj_file:
            writer = csv.DictWriter(nodes_without_adj_file, fieldnames=fieldnames)
            writer.writeheader()
            
            bbox_idx = 1
            count = 1
            node_idx = 0
            rows_written = 0
            
            for json_file in json_files:
                json_file_path = os.path.join(input_folder, json_file)
                img_name = os.path.splitext(os.path.basename(json_file_path))[0]
                with open(json_file_path, 'r') as json_file:
                    data = json.load(json_file)
                count_bbox = 1
                
                for bbox in data["bbox_info"]:
                    segs = bbox.get("seg", [])
                    if not segs:
                        continue
                    
                    node_idx = bbox_idx
                    for seg in segs:
                        id = int(seg['id'])
                        category_id = int(seg["category_id"])
                        seg_id = f"{category_id}_{id}"
                        intersection = seg["intersection"]
                        seg_percent = seg["seg_percent"]
                        bbox_percent = seg["bbox_percent"]
                        seg_area = seg["area"]
                        distance = seg.get("distance", None)
                        centroid_x = seg.get("centroid_x", None)
                        centroid_y = seg.get("centroid_y", None)
                        
                        node_idx += 1
                        image_bbox_id = f"{img_name.replace('_bbox_info_updated', '')}_{count_bbox}"
                        
                        row = {
                            'image_id': image_bbox_id, 'bbox_id': bbox_idx, 'node_id': node_idx,
                            'seg_id': seg_id, 'category_id': category_id, 'id': id,
                            'intersection': intersection, 'seg_area': seg_area,
                            'seg_percent': seg_percent, 'bbox_percent': bbox_percent,
                            'distance': distance, 'centroid_x': centroid_x, 'centroid_y': centroid_y
                        }
                        
                        writer.writerow(row)
                        rows_written += 1
                    bbox_idx = node_idx + 1
                    count_bbox += 1
                    count += 1

        if rows_written == 0:
            print(f"No data was written to nodes_without_adj.csv. Check if the JSON files contain the expected data.")
            return False
        else:
            print(f'Data has been written to nodes_without_adj.csv. {rows_written} rows were written.')
            return True
    except Exception as e:
        print(f"An error occurred in generate_nodes_without_adj: {str(e)}")
        return False

# nodes.csv
def generate_nodes(input_folder, output_folder, graph_filename):
    try:
        json_files = [f for f in os.listdir(input_folder) if f.endswith('bbox_info_updated.json')]
        
        if not json_files:
            print(f"No JSON files found in {input_folder}")
            return False

        fieldnames = ['image_id', 'bbox_id', 'node_id', 'seg_id', 'category_id', 'id', 'intersection', 'seg_area', 'seg_percent', 'bbox_percent', 'distance', 'centroid_x', 'centroid_y']
        
        nodes_path = os.path.join(output_folder, 'nodes.csv')
        
        # Read adjacency information
        adjacency_dict = {}
        with open(graph_filename, 'r') as graph_file:
            reader = csv.DictReader(graph_file)
            for row in reader:
                adjacency_dict[row['image_id']] = {
                    'adjacency': int(row['adjacency']),
                    'bbox_center_x': row.get('bbox_center_x', 'NaN'),
                    'bbox_center_y': row.get('bbox_center_y', 'NaN')
                }

        with open(nodes_path, 'w', newline='') as nodes_file:
            writer = csv.DictWriter(nodes_file, fieldnames=fieldnames)
            writer.writeheader()
            
            bbox_idx = 1
            count = 1
            node_idx = 0
            rows_written = 0
            
            for json_file in json_files:
                json_file_path = os.path.join(input_folder, json_file)
                img_name = os.path.splitext(os.path.basename(json_file_path))[0]
                with open(json_file_path, 'r') as json_file:
                    data = json.load(json_file)
                count_bbox = 1
                
                for bbox in data["bbox_info"]:
                    segs = bbox.get("seg", [])
                    if not segs:
                        continue
                    
                    node_idx = bbox_idx
                    image_bbox_id = f"{img_name.replace('_bbox_info_updated', '')}_{count_bbox}"
                    
                    # Get adjacency and bbox center coordinates from graphs_adj.csv
                    bbox_info = adjacency_dict.get(image_bbox_id, {})
                    adjacency = bbox_info.get('adjacency', 0)
                    bbox_center_x = bbox_info.get('bbox_center_x', 'NaN')
                    bbox_center_y = bbox_info.get('bbox_center_y', 'NaN')

                    if adjacency == 1:
                        node_idx += 1
                        writer.writerow({
                            'image_id': image_bbox_id, 'bbox_id': bbox_idx, 'node_id': node_idx,
                            'seg_id': 0, 'category_id': 0, 'id': 'NaN', 'intersection': 'NaN',
                            'seg_area': 'NaN', 'seg_percent': 'NaN', 'bbox_percent': 'NaN',
                            'distance': 'NaN', 'centroid_x': bbox_center_x,
                            'centroid_y': bbox_center_y
                        })
                        rows_written += 1

                    for seg in segs:
                        id = int(seg['id'])
                        category_id = int(seg["category_id"])
                        seg_id = f"{category_id}_{id}"
                        intersection = seg["intersection"]
                        seg_percent = seg["seg_percent"]
                        bbox_percent = seg["bbox_percent"]
                        seg_area = seg["area"]
                        distance = seg.get("distance", None)
                        centroid_x = seg.get("centroid_x", None)
                        centroid_y = seg.get("centroid_y", None)
                        
                        node_idx += 1
                        
                        row = {
                            'image_id': image_bbox_id, 'bbox_id': bbox_idx, 'node_id': node_idx,
                            'seg_id': seg_id, 'category_id': category_id, 'id': id,
                            'intersection': intersection, 'seg_area': seg_area,
                            'seg_percent': seg_percent, 'bbox_percent': bbox_percent,
                            'distance': distance, 'centroid_x': centroid_x, 'centroid_y': centroid_y
                        }
                        
                        writer.writerow(row)
                        rows_written += 1
                    bbox_idx = node_idx + 1
                    count_bbox += 1
                    count += 1

        if rows_written == 0:
            print(f"No data was written to nodes.csv. Check if the JSON files contain the expected data.")
            return False
        else:
            print(f'Data has been written to nodes.csv. {rows_written} rows were written.')
            return True
    except Exception as e:
        print(f"An error occurred in generate_nodes: {str(e)}")
        return False

# merged_data.csv
def merge_data(nodes_file, graphs_file, output_file):
    try:
        nodes_df = pd.read_csv(nodes_file)
        graphs_df = pd.read_csv(graphs_file)

        if nodes_df.empty or graphs_df.empty:
            print("One of the input DataFrames is empty. Cannot perform merge.")
            return False

        merged_df = pd.merge(graphs_df, nodes_df, on='image_id')
        merged_df.to_csv(output_file, index=False)
        print(f'Merged data has been written to {output_file}')
        print(f'Number of rows in merged data: {len(merged_df)}')
        return True
    except Exception as e:
        print(f"An error occurred in merge_data: {str(e)}")
        print(traceback.format_exc())
        return False

# csv2txt
def generate_node_labels(csv_file, output_directory):

    node_attributes = {}
    
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            node_id = int(row['node_id'])
            bbox_id = int(row['bbox_id'])
            category_id = int(row['category_id'])
            label = int(float(row['label']))
            if label == 0:
                label = -1
            # else:
            #     label = 0
            
            node_attributes[node_id] = 0
            node_attributes[bbox_id] = label
        # Create output directory if it doesn't exist
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
    
    with open(f"{output_directory}/Arcade_node_labels.txt", "w") as file:
        max_id = max(max(node_attributes.keys()), 1)
        for i in range(1, max_id + 1):
            category_id = node_attributes.get(i, None)
            file.write(f"{category_id}\n")

def generate_node_attributes(csv_file, output_directory):
    node_attributes = {}
    
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            node_id = int(row['node_id'])
            bbox_id = int(row['bbox_id'])
            category_id = int(row['category_id'])
            score = float(row['score'])
            node_attributes[node_id] = (category_id, 0)
            node_attributes[bbox_id] = (31, score)
    

    # find the biggest node/bbox id 2024/11/20 add
    max_valid_id = max(node_attributes.keys())

    with open(f"{output_directory}/Arcade_node_attributes.txt", "w") as file:
        # max_id = max(max(node_attributes.keys()), 31)
        
        # for i in range(1, max_id + 1):
        #     category_id, score = node_attributes.get(i, (None, None))
        #     file.write(f"{category_id}, {score}\n")

        # 2024/11/20 add
        for i in range(1, max_valid_id+1):
            if i in node_attributes:
                category_id, score = node_attributes[i]
                file.write(f"{category_id}, {score}\n")

def compare_adjacent_bboxes(data, threshold):
    image_width = 640
    image_height = 640
    new_rows = []
    
    for graph_id, group in data.groupby('graph_id'):
        bbox_ids = group['bbox_id'].unique()
        
        for i in range(len(bbox_ids)):
            for j in range(i + 1, len(bbox_ids)):
                bbox_id1 = bbox_ids[i]
                bbox_id2 = bbox_ids[j]
                
                bbox1 = group[group['bbox_id'] == bbox_id1].iloc[0]
                bbox2 = group[group['bbox_id'] == bbox_id2].iloc[0]
                

                bbox1_coords = calculate_coordinates(bbox1['bbox_center_x'], bbox1['bbox_center_y'], bbox1['bbox_width'], bbox1['bbox_height'], image_width, image_height, bbox1['bbox_id'])
                bbox2_coords = calculate_coordinates(bbox2['bbox_center_x'], bbox2['bbox_center_y'], bbox2['bbox_width'], bbox2['bbox_height'], image_width, image_height, bbox2['bbox_id'])
                
                adj, adj_bbox_id1, adj_bbox_id2 = are_adjacent(bbox1_coords, bbox2_coords, threshold)

                if adj:
                    new_row = bbox2.copy()
                    new_row['node_id'] = adj_bbox_id2
                    new_row['bbox_id'] = adj_bbox_id1
                    new_row['category_id'] = 31
                    new_row['seg_id'] = str(new_row['graph_id']) + '_' + str(new_row['category_id'])
                    # print(new_row)
                    new_rows.append(new_row)
    
    new_data = pd.concat([data, pd.DataFrame(new_rows)], ignore_index=True)
    # print(new_data)
    new_data['id'] = new_data.index
    return new_data

# Arcade_graph_labels.txt, Arcade_graph_indicator.txt, Arcade_A.txt
def csv2txt(csv_file, output_directory):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Extract columns 'bbox_id' and 'node_id' 2024/4/24 edit
    arcade_A = df[['bbox_id', 'node_id']]
    
    # Check if the output directory exists, if not, create it
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Write the extracted data to a specified file
    output_file = f"{output_directory}/Arcade_A.txt"
    with open(output_file, 'w') as f:
        for _, row in arcade_A.iterrows():
            f.write(f"{row['bbox_id']}, {row['node_id']}\n")

    # Extract unique graph_ids
    unique_graph_ids = df['graph_id'].unique()
    graph_indicator_list = []
    
    # 2024/4/24 edit
    index = 0
    for graph_id in unique_graph_ids:
        max_bbox_id = df[df['graph_id'] == graph_id]['bbox_id'].max()
        graph_indicator_list.extend([graph_id] * (max_bbox_id-index))
        index = max_bbox_id

    # Write the graph indicator data to a specified file
    output_graph_indicator_file = f"{output_directory}/Arcade_graph_indicator.txt"
    with open(output_graph_indicator_file, 'w') as f:
        for item in graph_indicator_list:
            f.write(f"{item}\n")

    # 2024/3/15
    # Extract unique graph labels
    unique_graph_labels = df.groupby('graph_id')['label'].first()
    
    # Replace label 0 with -1
    # unique_graph_labels = unique_graph_labels.replace({0: -1})
    unique_graph_labels = unique_graph_labels.apply(lambda x: -1) # let graph all for -1

    # Write the graph labels to a specified file
    output_graph_labels_file = f"{output_directory}/Arcade_graph_labels.txt"
    with open(output_graph_labels_file, 'w') as f:
        for graph_id, label in unique_graph_labels.items():
            f.write(f"{int(label)}\n")

def generate_edge_attributes(csv_file, output_directory):
    df = pd.read_csv(csv_file)

    with open(f"{output_directory}/Arcade_edge_attributes.txt", "w") as edge_file:
        
        for _, row in df.iterrows():
            bbox_center_x = row['bbox_center_x']
            bbox_center_y = row['bbox_center_y']
            bbox_width = row['bbox_width']
            bbox_height = row['bbox_height']
            category_id = row['category_id']
            graph_id = row['graph_id']

            image_width = 640
            image_height = 640
            
            x_min, y_min, x_max, y_max, _ = calculate_coordinates(bbox_center_x, bbox_center_y, bbox_width, bbox_height, image_width, image_height, graph_id)
            centroid_x = row['centroid_x'] * image_width
            centroid_y = row['centroid_y'] * image_height

            # Predicate rules
            if category_id == 31:
                ans = 3 # Adj (Arcade predicate arcade)
            
            # 2024/5/2 edit
            elif (x_min <= centroid_x <= x_max) and (y_min <= centroid_y <= y_max) and category_id != 0:
                ans = 0  # Contain 0 (Arcade predicate 26 categories)
            else:
                ans = 2  # overlap (Arcade predicate 26 categories) 2024/9/8 add
            
            edge_file.write(f"{ans}\n")

# create bi-directed graph
def data2bidirected(output_directory):
    # Read Arcade_A.txt
    raw_data_file = f"{output_directory}/Arcade_A.txt"
    with open(raw_data_file, 'r') as file:
        raw_data = [tuple(map(int, line.strip().split(','))) for line in file]

    # Read edge_attributes.txt
    edge_attributes_file = f"{output_directory}/Arcade_edge_attributes.txt"
    with open(edge_attributes_file, 'r') as file:
        edge_attributes_data = [line.strip() for line in file]

    # Create dict for edge and its values
    edge_attributes_dict = {}
    for edge, attribute in zip(raw_data, edge_attributes_data):
        edge_attributes_dict[edge] = int(attribute)

    # Add opposite direction
    for edge in raw_data:
        reverse_edge = (edge[1], edge[0])
        if edge_attributes_dict[edge] == 0:  # 0 = contain
            reverse_attribute = 1  # 1 = inside
        elif edge_attributes_dict[edge] == 3:  # 3 = adj
            reverse_attribute = 3  # Keep adj for both directions
        else:
            reverse_attribute = edge_attributes_dict[edge]  # 2 = Overlap
        edge_attributes_dict[reverse_edge] = reverse_attribute

    # Sort values
    converted_data = list(edge_attributes_dict.keys())
    converted_data.sort(key=lambda x: (x[0], x[1]))

    # Write updated Arcade_A.txt
    output_file = f"{output_directory}/Arcade_A.txt"
    with open(output_file, 'w') as file:
        for edge in converted_data:
            file.write(f"{edge[0]}, {edge[1]}\n")

    # Write updated edge_attributes.txt
    output_edge_attributes_file = f"{output_directory}/Arcade_edge_attributes.txt"
    with open(output_edge_attributes_file, 'w') as file:
        for edge in converted_data:
            file.write(f"{edge_attributes_dict[edge]}\n")

    print("Data has been written to", output_edge_attributes_file)
    print("Data has been written to", output_file)


# one_hot_encode
def one_hot_encode_txt_file(file_path, output_file_path, num_classes=None, include_scores=False):
    # Read data from .txt file
    with open(file_path, "r") as file:
        data = file.readlines()

    # Split the data into category_id and score, and convert to appropriate types
    categories = []
    scores = []
    for line in data:
        parts = line.strip().split(', ')
        if len(parts) == 2:
            category_id, score = parts
            categories.append(int(category_id))
            scores.append(float(score))
        else:
            categories.append(int(parts[0]))
            scores.append(np.nan)

    # Convert categories to PyTorch tensor
    categories_tensor = torch.tensor(categories)

    # Determine the number of unique classes in the data
    if num_classes is None:
        num_classes = len(torch.unique(categories_tensor))
        
    # Determine the maximum class value
    max_class_value = torch.max(categories_tensor).item()

    # Adjust num_classes if necessary to accommodate all class values
    if max_class_value >= num_classes:
        num_classes = max_class_value + 1

    # Perform one-hot encoding on the categories
    one_hot_encoded_categories = torch.nn.functional.one_hot(categories_tensor, num_classes)

    # Save the one-hot encoded data and scores to file
    with open(output_file_path, "w") as file:
        for one_hot_category, score in zip(one_hot_encoded_categories, scores):
            one_hot_category_str = ", ".join(map(str, one_hot_category.tolist()))
            
            if include_scores and not np.isnan(score):
                file.write(f"{one_hot_category_str}, {score}\n")
            else:
                file.write(f"{one_hot_category_str}\n")

    print("Data has been written to", output_file_path)
    return one_hot_encoded_categories


# filtered_data.csv, encoded_data_v3.csv, merged_data_v3.csv, Arcade_edge_attributes.txt, Arcade_node_attributes.txt, Arcade_node_labels.txt
# 2024/12/13
def additional_processing(csv_folder, output_directory):
    # Read merged_data.csv
    df = pd.read_csv(f'{csv_folder}/merged_data.csv')
    
    # Filter out rows where seg_id is "0"
    df_filtered = df[df['seg_id'] != "0"]
    df_filtered.to_csv(f'{csv_folder}/filtered_data.csv', index=False)
    
    # Read filtered_data.csv
    df = pd.read_csv(f'{csv_folder}/filtered_data.csv')
    
    # Extract image_name with more flexible handling
    def extract_image_base(image_id):
        # if image_id contains $
        if '$' in image_id:
            parts = image_id.split('$')
            if len(parts) >= 2:
                return '$'.join(parts[:2])
        
        # if no $, just return the base name
        base_name = re.sub(r'_\d+$', '', image_id)
        return base_name
    
    # apply the function to the 'image_id' column
    df['image_name'] = df['image_id'].apply(extract_image_base)
    
    # Create unique graph_id for each image_name
    graph_id_dict = {}
    graph_id = 1
    current_image_name = None
    
    for index, row in df.iterrows():
        if row['image_name'] != current_image_name:
            current_image_name = row['image_name']
            graph_id_dict[current_image_name] = graph_id
            graph_id += 1
    
    # Update graph_id and seg_id
    df['graph_id'] = df['image_name'].map(graph_id_dict)
    df['seg_id'] = df['graph_id'].astype(str) + '_' + df['seg_id'].astype(str)
    
    # Drop image_name column
    df.drop('image_name', axis=1, inplace=True)
    
    # Encode seg_id
    encoded_seg_ids = {}
    def encode_seg_id(seg_id):
        if seg_id in encoded_seg_ids:
            return encoded_seg_ids[seg_id]
        else:
            new_id = len(encoded_seg_ids) + 1
            encoded_seg_ids[seg_id] = new_id
            return new_id
    
    df['encoded_seg_id'] = df['seg_id'].apply(encode_seg_id)
    df.to_csv(f'{csv_folder}/encoded_data_v3.csv', index=False)
    # Process node_id and bbox_id
    data = pd.read_csv(f'{csv_folder}/encoded_data_v3.csv')

    # 處理 node_id
    node_id_dict = {}
    for graph_id, group in data.groupby('graph_id'):
        if graph_id == 1:
            last_image_count = 0 # start value
        else:
            last_image_count = data[data['graph_id'] < graph_id]['image_id'].nunique()
        for index, row in group.iterrows():
            encode_seg_id = row['encoded_seg_id']
            image_id = row['image_id']
            if graph_id not in node_id_dict:
                node_id_dict[graph_id] = {}
            if encode_seg_id not in node_id_dict[graph_id]:
                node_id_dict[graph_id][encode_seg_id] = 1
            node_id = encode_seg_id + last_image_count
            node_id_dict[graph_id][encode_seg_id] = node_id
            data.at[index, 'node_id'] = node_id

    # 處理 bbox_id
    bbox_id_dict = {}
    for graph_id, group in data.groupby('graph_id'):
        if graph_id == 1:
            last_image_count = 0 # start value
        else:
            last_image_count = data[data['graph_id'] < graph_id]['image_id'].nunique()
        max_node_id = group['node_id'].max()
        if graph_id not in bbox_id_dict:
            bbox_id_dict[graph_id] = {}
        for image_id, sub_group in group.groupby('image_id'):
            if image_id not in bbox_id_dict[graph_id]:
                bbox_id_dict[graph_id][image_id] = max_node_id + 1
                max_node_id += 1
            bbox_id = bbox_id_dict[graph_id][image_id]
            data.loc[sub_group.index, 'bbox_id'] = bbox_id

    data.to_csv(f'{csv_folder}/merged_data_v3.csv', index=False)
    
    # Generate node labels txt
    generate_node_labels(f'{csv_folder}/merged_data_v3.csv', output_directory)
    
    # Generate node attributes txt
    generate_node_attributes(f'{csv_folder}/merged_data_v3.csv', output_directory)
    
    # Process adjacency
    threshold = 125.4465301465138 # quartile method
    new_data = compare_adjacent_bboxes(data, threshold)
    new_data.to_csv(f'{csv_folder}/merged_data_v3.csv', index=False)
    
    # Generate other required files
    csv2txt(f'{csv_folder}/merged_data_v3.csv', output_directory)
    
    # Generate edge attributes
    generate_edge_attributes(f'{csv_folder}/merged_data_v3.csv', output_directory)

    # let the graph be bidirected
    data2bidirected(output_directory)

    # One-hot encode node attributes
    node_file_path = f"{output_directory}/Arcade_node_attributes.txt"
    node_num_classes = None
    node_output_file_path = f"{output_directory}/Arcade_node_attributes_encoded.txt"
    one_hot_encode_txt_file(node_file_path, node_output_file_path, node_num_classes, include_scores=False)

    # One-hot encode edge attributes
    edge_file_path = f"{output_directory}/Arcade_edge_attributes.txt"
    edge_num_classes = None
    edge_output_file_path = f"{output_directory}/Arcade_edge_attributes_encoded.txt" # 看要不要覆蓋以前的檔案
    one_hot_encode_txt_file(edge_file_path, edge_output_file_path, edge_num_classes, include_scores=False)


def main():
    parser = argparse.ArgumentParser(description="Process JSON files and create CSV outputs.")
    parser.add_argument("--input_dir", default="data", help="Directory containing input JSON files")
    parser.add_argument("--output_dir", default="csv", help="Directory to save output CSV files")
    parser.add_argument("--final_output", default="Arcade_v3", help="Directory for final output files")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    
    json_folder = os.path.abspath(args.input_dir)
    csv_folder = os.path.abspath(args.output_dir)
    final_output = os.path.abspath(args.final_output)
    
    print(f"Input directory: {json_folder}")
    print(f"Output directory: {csv_folder}")
    print(f"Final output directory: {final_output}")
    
    os.makedirs(csv_folder, exist_ok=True)
    os.makedirs(final_output, exist_ok=True)
    
    # Generate graphs.csv
    if not process_json_to_graphscsv(json_folder, csv_folder, 'graphs.csv'):
        print("Failed to process JSON to graphs.csv. Exiting.")
        return
    print("Successfully processed JSON to graphs.csv.")
    
    # Generate graphs_adj.csv
    if not process_graph_adj(os.path.join(csv_folder, 'graphs.csv'), os.path.join(csv_folder, 'graphs_adj.csv')):
        print("Failed to process adjacency. Exiting.")
        return
    print("Successfully processed adjacency.")

    # Generate nodes_without_adj.csv
    if not generate_nodes_without_adj(json_folder, csv_folder):
        print("Failed to generate nodes_without_adj.csv. Exiting.")
        return
    print("Successfully generated nodes_without_adj.csv.")

    # Generate nodes.csv
    graph_adj_file = os.path.join(csv_folder, 'graphs_adj.csv')
    if not generate_nodes(json_folder, csv_folder, graph_adj_file):
        print("Failed to generate nodes.csv. Exiting.")
        return
    print("Successfully generated nodes.csv.")

    # Generate merged_data.csv
    if not merge_data(os.path.join(csv_folder, 'nodes.csv'), 
                      os.path.join(csv_folder, 'graphs.csv'), 
                      os.path.join(csv_folder, 'merged_data.csv')):
        print("Failed to merge data. Exiting.")
        return
    print("Successfully merged data.")


    # Additional processing steps
    additional_processing(csv_folder, final_output)
    print("Additional processing completed.")

    print("All processing steps completed successfully.")

if __name__ == "__main__":
    main()