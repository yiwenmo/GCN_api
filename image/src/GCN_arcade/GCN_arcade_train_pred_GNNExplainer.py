import argparse
import torch
import os
import pandas as pd
import numpy as np
from torch_geometric.data import DataLoader
from torch_geometric.explain import Explainer, GNNExplainer
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

# custom modules from other py files
from utils.data_utils_node import CustomDataset
from utils.data_display import display_misclassified_images
from model.model_node import GCN, GCN_batchnorm


def get_image_id_mapping(csv_file):
    df = pd.read_csv(csv_file)
    image_id_mapping = {row['bbox_id']: row['image_id'] for index, row in df.iterrows()}
    return image_id_mapping


def visualize_results(y_true, y_pred, output_folder):
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(output_folder, 'confusion_matrix.png'))
    plt.close()

    # Classification Report
    report = classification_report(y_true, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(os.path.join(output_folder, 'classification_report.csv'))


def main(args):
    # Determine the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the dataset
    new_dataset = CustomDataset(root='.', filepath=args.filepath, name=f'pt/{args.dir_name}')
    new_loader = DataLoader(new_dataset, batch_size=64, shuffle=False)  # Fixed batch size
    input_dim = new_dataset.num_node_features
    output_dim = new_dataset.num_classes

    # Define and load the model
    model = GCN_batchnorm(input_dim, 64, output_dim)  # Fixed hidden dim
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Create explainer
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),  # Fixed number of epochs
        explanation_type='phenomenon',  # Fixed explanation type
        node_mask_type='common_attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='log_probs',
        ),
    )

    # Get image ID mapping
    image_id_mapping = get_image_id_mapping(args.merged_data_path)
    os.makedirs(args.output_folder, exist_ok=True)

    # Read node labels
    with open(args.label_file_path, 'r') as file:
        labels = [int(line.strip()) for line in file]
    target_nodes = [i for i, label in enumerate(labels) if label in [-1, 1]]

    # Generate explanations
    for i, target_node in enumerate(target_nodes):
        x, edge_index = new_dataset.x.to(device), new_dataset.edge_index.to(device)
        target = new_dataset.y.to(device)  # Move target to the same device
        
        try:
            explanation = explainer(x, edge_index, index=target_node, target=target)
        
            bbox_id = target_node + 1
            image_id = image_id_mapping.get(bbox_id)
            
            if image_id is None:
                print(f"No image ID found for target node {bbox_id}")
                continue
            
            # Feature importance
            feature_importance_path = os.path.join(args.output_folder, f'{image_id}_FI_{bbox_id}.png')
            explanation.visualize_feature_importance(feature_importance_path, top_k=5)
            
            # Save subgraph
            subgraph_path = os.path.join(args.output_folder, f'{image_id}_subgraph_{bbox_id}.png')
            explanation.visualize_graph(subgraph_path)
        except RuntimeError as e:
            print(f"Error processing target node {target_node}: {str(e)}")
            continue

    # Process node labels and predictions
    node_labels = pd.read_csv(args.label_file_path, header=None, names=["label"])
    node_labels.reset_index(inplace=True)
    node_labels.rename(columns={'index': 'node_id'}, inplace=True)
    node_labels['node_id'] += 1
    node_labels['label'] += 1

    # Generate predictions
    predictions = []
    with torch.no_grad():
        for data in new_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            predictions.extend(pred.cpu().numpy())

    assert len(predictions) == len(node_labels), "Predictions length does not match node_labels length"

    node_labels['Predict'] = predictions

    # Save node labels with predictions
    node_labels.to_csv(os.path.join(args.filepath, args.node_labels_output_filename), index=False)

    # Merge data and save results
    node_labels_preds = node_labels[['node_id', 'label', 'Predict']]
    merged_data = pd.read_csv(args.merged_data_path)
    merged_data = merged_data.drop_duplicates(subset='bbox_id', keep='first')
    merged = pd.merge(merged_data, node_labels_preds, left_on='bbox_id', right_on='node_id')

    result = merged[['image_id', 'graph_id', 'bbox_center_x', 'bbox_center_y', 'bbox_width', 'bbox_height', 'bbox_area', 'score', 'label_y', 'bbox_id', 'Predict']]
    result = result.rename(columns={'label_y': 'label', 'Predict': 'predict'})

    result['label'] = result['label'] - 1
    result['predict'] = result['predict'] - 1

    result.to_csv(os.path.join(args.filepath, args.result_output_filename), index=False)

    # Visualize results
    visualize_results(result['label'], result['predict'], args.output_folder)

    # Print class counts
    class_counts = Counter(labels)
    print("Class counts:", class_counts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GNN Explainer Script")
    parser.add_argument("--dir_name", type=str, required=True, help="Name of the test directory")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model file")
    parser.add_argument("--output_folder", type=str, required=True, help="Name of the output image folder")
    parser.add_argument("--node_labels_output_filename", type=str, required=True, help="Output filename for node labels")
    parser.add_argument("--result_output_filename", type=str, required=True, help="Output filename for results")
    parser.add_argument("--merged_data_path", type=str, required=True, help="Path to merged_data file")
    parser.add_argument("--label_file_path", type=str, required=True, help="Path to label file")
    parser.add_argument("--filepath", type=str, required=True, help="Path to dataset")

    args = parser.parse_args()
    main(args)