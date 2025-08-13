import os
import cv2
import json
import subprocess
from pathlib import Path
import shutil
import pandas as pd

class GCNProcessor:
    def __init__(self):
        # set up paths
        self.project_root = Path('/workspace/Mask2Former')
        self.src_dir = self.project_root / 'image' / 'src'
        self.output_dir = self.project_root / 'output'

        # data paths
        self.csv_dir = self.output_dir / 'csv'
        self.arcade_v3_dir = self.output_dir / 'Arcade_v3'
        self.gcn_results_dir = self.output_dir / 'gcn_results'
        
        # gcn script paths
        self.image_processing_script = self.src_dir / 'preprocessing' / 'image_processing.py'
        self.json2txt_script = self.src_dir / 'bidirected_arcade_scene_graph' / 'json2txt.py'
        self.gcn_script = self.src_dir / 'GCN_arcade' / 'GCN_arcade.py'
        self.model_path = self.src_dir / 'GCN_arcade' / 'best_model_node_200epochs_GCN64_bn_128batch_final_4spr.pth'
        
    def prepare_directories(self):
        """create directories"""
        dirs = [self.csv_dir, self.arcade_v3_dir, self.gcn_results_dir]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)

    def step1_preprocessing(self, json_path):
        """Step 1: using image_processing.py"""
        try:
            cmd = [
                'python', str(self.image_processing_script),
                '--img_dir', str(self.output_dir),
                '--txt_dir', str(self.output_dir),
                '--save_path', str(self.output_dir)
            ]
            subprocess.run(cmd, check=True)
            return True
        except Exception as e:
            print(f"Preprocessing failed: {str(e)}")
            return False

    def step2_json2txt(self):
        """Step 2: using json2txt.py for JSON to TXT conversion"""
        try:
            cmd = [
                'python', str(self.json2txt_script),
                '--input_dir', str(self.output_dir),
                '--output_dir', str(self.csv_dir),
                '--final_output', str(self.arcade_v3_dir)
            ]
            subprocess.run(cmd, check=True)
            return True
        except Exception as e:
            print(f"JSON to TXT conversion failed: {str(e)}")
            return False

    def step3_gcn_prediction(self):
        """Step 3: execute GCN prediction"""
        try:
            result_file = self.gcn_results_dir / 'gcn_results.csv'
            cmd = [
                'python', str(self.gcn_script),
                '--dir_name', 'Arcade_v3',
                '--model_path', str(self.model_path),
                '--output_folder', str(self.gcn_results_dir),
                '--result_output_filename', str(result_file),
                '--merged_data_path', str(self.csv_dir / 'merged_data_v3.csv'),
                '--filepath', str(self.arcade_v3_dir)
            ]
            subprocess.run(cmd, check=True)
            return result_file
        except Exception as e:
            print(f"GCN prediction failed: {str(e)}")
            return None

    def visualize_gcn_results(self, json_path):
        """show gcn results on image"""
        try:
            # read gcn results from csv
            result_file = self.gcn_results_dir / 'gcn_results.csv'
            results_df = pd.read_csv(result_file)
            
            # get gsv and output path
            base_name = Path(json_path).stem.replace('_bbox_info', '')
            gsv_path = self.output_dir / f"{base_name}_gsv.png"
            output_path = self.output_dir / f"{base_name}_gcn_result.png"
            
            # Check if gsv file exists, if not, try to find and copy the original image
            if not gsv_path.exists():
                print(f"GSV file not found: {gsv_path}, searching for original image...")
                
                # Try to find the original image in various locations
                original_image_path = None
                for folder in [self.upload_dir, Path("host_upload"), Path("static/uploads")]:
                    for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG']:
                        test_path = folder / f"{base_name}{ext}"
                        if test_path.exists():
                            original_image_path = test_path
                            print(f"Found original image: {original_image_path}")
                            break
                    if original_image_path:
                        break
                
                # If original image found, copy it to gsv_path
                if original_image_path:
                    import shutil
                    shutil.copy2(original_image_path, gsv_path)
                    print(f"Copied original image to GSV path: {gsv_path}")


            # read image
            img = cv2.imread(str(gsv_path))
            if img is None:
                raise FileNotFoundError(f"Cannot find image file: {gsv_path}")
            
            img_cp = img.copy()
            
            # process gcn results
            for _, row in results_df.iterrows():
                # Skip fake bbox (identified by its fixed dimensions)
                if row['bbox_width'] == 0.2 and row['bbox_height'] == 0.2:
                    continue

                # get bbox info
                bbox_center_x = float(row['bbox_center_x'])
                bbox_center_y = float(row['bbox_center_y'])
                bbox_width = float(row['bbox_width'])
                bbox_height = float(row['bbox_height'])
                prediction = int(row['predict'])
                
                # calculate top_left and bottom_right of bbox
                top_left_x = int((bbox_center_x - bbox_width/2) * img.shape[1])
                top_left_y = int((bbox_center_y - bbox_height/2) * img.shape[0])
                bottom_right_x = int((bbox_center_x + bbox_width/2) * img.shape[1])
                bottom_right_y = int((bbox_center_y + bbox_height/2) * img.shape[0])
                
                
                # arcade for yellow, non-arcade for orange
                bbox_color = (0, 255, 255) if prediction == 1 else (102, 178, 255)
                
                # draw bbox
                cv2.rectangle(img_cp, 
                            (top_left_x, top_left_y), 
                            (bottom_right_x, bottom_right_y), 
                            bbox_color, 2)
                
            # save  
            cv2.imwrite(str(output_path), img_cp)
            return str(output_path)
        
        except Exception as e:
            print(f"Visualization failed: {str(e)}")
            return None

    def process_results(self, result_file):
        """process GCN results"""
        if result_file and result_file.exists():
            try:
                with open(result_file, 'r') as f:
                    results = pd.read_csv(f)
                return {
                    'status': 'success',
                    'predictions': results.to_dict('records')
                }
            except Exception as e:
                return {
                    'status': 'error',
                    'message': f"Failed to process results: {str(e)}"
                }
        return {
            'status': 'error',
            'message': 'No result file generated'
        }

    def process(self, json_path):
        """main process"""
        try:
            self.prepare_directories()
            
            # Step 1: preprocessing
            if not self.step1_preprocessing(json_path):
                return {'status': 'error', 'message': 'Preprocessing failed'}
            
            # Step 2: JSON to TXT
            if not self.step2_json2txt():
                return {'status': 'error', 'message': 'JSON to TXT conversion failed'}
            
            # Step 3: GCN prediction
            result_file = self.step3_gcn_prediction()
            if not result_file:
                return {'status': 'error', 'message': 'GCN prediction failed'}
            
            # Step 4: visualization
            results = self.process_results(result_file)
            visualization_path = self.visualize_gcn_results(json_path)
            
            if visualization_path:
                return {
                    'status': 'success',
                    'predictions': results.get('predictions', []),
                    'visualization': {
                        'image_path': visualization_path,
                        'format': 'png'
                    }
                }
            else:
                return {
                    'status': 'error',
                    'message': 'Failed to generate visualization'
                }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }