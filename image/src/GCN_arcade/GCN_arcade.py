import argparse
import torch
import os
import pandas as pd
from torch_geometric.data import DataLoader
from collections import Counter

import warnings
warnings.filterwarnings("ignore")

# custom modules from other py files
from utils.data_utils_node import CustomDataset
from model.model_node import GCN_batchnorm

def predict_nodes(model, loader, device):
    predictions = []
    model.eval()
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            predictions.extend(pred.cpu().numpy())
    return predictions

def main(args):
    # Determine the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the dataset
    new_dataset = CustomDataset(root='.', filepath=args.filepath, name=f'pt/{args.dir_name}')
    new_loader = DataLoader(new_dataset, batch_size=64, shuffle=False)
    input_dim = new_dataset.num_node_features
    output_dim = new_dataset.num_classes

    # Define and load the model
    model = GCN_batchnorm(input_dim, 64, output_dim)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Get predictions
    predictions = predict_nodes(model, new_loader, device)
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'node_id': range(1, len(predictions) + 1),
        'predict': predictions
    })
    
    # Merge with metadata
    merged_data = pd.read_csv(args.merged_data_path)
    merged_data = merged_data.drop_duplicates(subset='bbox_id', keep='first')
    merged = pd.merge(merged_data, results_df, left_on='bbox_id', right_on='node_id')

    # Select and rename columns
    result = merged[['image_id', 'graph_id', 'bbox_center_x', 'bbox_center_y', 
                    'bbox_width', 'bbox_height', 'bbox_area', 'score', 
                    'bbox_id', 'predict']]
    
    # Adjust predictions to match original scale (-1, 0, 1)
    result['predict'] = result['predict'] - 1

    # Save results
    os.makedirs(args.output_folder, exist_ok=True)
    result.to_csv(os.path.join(args.filepath, args.result_output_filename), index=False)

    print(f"Predictions saved to {args.result_output_filename}")
    print("Prediction counts:", Counter(predictions))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GNN Prediction Script")
    parser.add_argument("--dir_name", type=str, required=True, help="Name of the test directory")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model file")
    parser.add_argument("--output_folder", type=str, required=True, help="Name of the output folder")
    parser.add_argument("--result_output_filename", type=str, required=True, help="Output filename for results")
    parser.add_argument("--merged_data_path", type=str, required=True, help="Path to merged_data file")
    parser.add_argument("--filepath", type=str, required=True, help="Path to dataset")
    
    args = parser.parse_args()
    main(args)