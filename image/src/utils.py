# utils.py
import base64
import re
import os
import json
import subprocess
from PIL import Image
from io import BytesIO
from werkzeug.utils import secure_filename


import requests
from urllib3.util import Retry
from requests.adapters import HTTPAdapter

UPLOAD_FOLDER = '/workspace/Mask2Former/upload'
OUTPUT_FOLDER = '/workspace/Mask2Former/output'
DATA_FOLDER = '/workspace/Mask2Former/data'
MODELS_FOLDER = '/workspace/Mask2Former/models'



def base64_to_image(img_base64, max_dimension=1024):
    """
    convert base64 image to PIL image
    """
    try:
        # decode base64
        img_data = re.sub('^data:image/.+;base64,', '', img_base64)
        img = Image.open(BytesIO(base64.b64decode(img_data)))
        
        # if the image is RGBA, convert it to RGB
        if img.mode == 'RGBA':
            img = img.convert('RGB')
            
        # check if the image is too large
        img = resize_if_needed(img, max_dimension)
        
        return img
    except Exception as e:
        raise Exception(f"處理圖片時發生錯誤: {str(e)}")

# 2024/12/5 upload with file name
# def process_image_file(image_path):
#     """Process image using Mask2Former and return paths to results"""
#     try:
        
#         # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S') # use timestamp for file name
#         full_filename = os.path.basename(image_path)
#         base_name = os.path.splitext(os.path.basename(image_path))[0]
        
#         # define output paths
#         # result_image_path = os.path.join(OUTPUT_FOLDER, f'image_output_{timestamp}.png')
#         result_image_path = os.path.join(OUTPUT_FOLDER, f'{base_name}_seg.png')
#         result_json_path = os.path.join(OUTPUT_FOLDER, f'{base_name}.json')
#         result_pt_path = os.path.join(OUTPUT_FOLDER, f'{base_name}.pt')


#         # use the model's config and weights
#         config_path = os.path.join(MODELS_FOLDER, 'config.yaml')
#         model_path = os.path.join(MODELS_FOLDER, 'model_final.pth')
        
#         # check if model's files exist
#         if not os.path.exists(config_path):
#             raise FileNotFoundError(f"Config file not found: {config_path}")
#         if not os.path.exists(model_path):
#             raise FileNotFoundError(f"Model weights not found: {model_path}")
        
#         # Run Mask2Former prediction
#         command = f"python demo/demo_mo_test.py --config-file {config_path} --input {image_path} --output {result_image_path} --opts MODEL.WEIGHTS {model_path}"
        
#         subprocess.run(command, shell=True, check=True)
        
#         # 2024/12/4 edited
#         # if the default output files exist, rename them
#         default_pt = os.path.join(OUTPUT_FOLDER, f'{base_name.split(".")[0]}.pt')
#         default_json = os.path.join(OUTPUT_FOLDER, f'{base_name.split(".")[0]}.json')
        
#         if os.path.exists(default_pt):
#             os.rename(default_pt, result_pt_path)
#         if os.path.exists(default_json):
#             os.rename(default_json, result_json_path)

#         return result_image_path, result_json_path
        
#     except subprocess.CalledProcessError as e:
#         raise RuntimeError(f"Error running Mask2Former: {str(e)}")
#     except Exception as e:
#         raise RuntimeError(f"Error processing image: {str(e)}")


def process_image_file(image_path, max_dimension=1024):
    """
    處理圖片文件，如果需要則先調整大小
    """
    try:
        
        with Image.open(image_path) as img:
            # 如果是 RGBA 模式，轉換為 RGB
            if img.mode == 'RGBA':
                img = img.convert('RGB')
                
            # 檢查並調整大小
            img = resize_if_needed(img, max_dimension)
            
            # 保存調整後的圖片
            img.save(image_path, 'png', quality=95)
        
        
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        
        result_image_path = os.path.join(OUTPUT_FOLDER, f'{base_name}_seg.png')
        result_json_path = os.path.join(OUTPUT_FOLDER, f'{base_name}.json')
        result_pt_path = os.path.join(OUTPUT_FOLDER, f'{base_name}.pt')

        
        config_path = os.path.join(MODELS_FOLDER, 'config.yaml')
        model_path = os.path.join(MODELS_FOLDER, 'model_final.pth')
        
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"config file not found: {config_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"model weights not found: {model_path}")
        

        command = f"python demo/demo_mo_test.py --config-file {config_path} --input {image_path} --output {result_image_path} --opts MODEL.WEIGHTS {model_path}"
        
        subprocess.run(command, shell=True, check=True)
        
        
        default_pt = os.path.join(OUTPUT_FOLDER, f'{base_name.split(".")[0]}.pt')
        default_json = os.path.join(OUTPUT_FOLDER, f'{base_name.split(".")[0]}.json')
        
        if os.path.exists(default_pt):
            os.rename(default_pt, result_pt_path)
        if os.path.exists(default_json):
            os.rename(default_json, result_json_path)

        return result_image_path, result_json_path
        
    except Exception as e:
        raise RuntimeError(f"error processing image: {str(e)}")

# 2024/12/6 add
def create_response_data(result_image_path, result_json_path):
    """Create standardized response data from result paths"""
    with open(result_json_path, 'r') as json_file:
        uploaded_data = json.load(json_file)

    with open(result_image_path, 'rb') as img_file:
        encoded_image = base64.b64encode(img_file.read()).decode('utf-8')
    
    image_data_url = f"data:image/png;base64,{encoded_image}"

    return {
        'image': image_data_url,
        'seg_info': uploaded_data,
        'files': {
            'image_path': result_image_path,
            'json_path': result_json_path
        }
    }

def get_result_paths(base_filename, output_folder):
    """Generate standardized result paths"""
    return {
        'image': os.path.join(output_folder, f'{base_filename}.png'),
        'json': os.path.join(output_folder, f'{base_filename}.json'),
        'pt': os.path.join(output_folder, f'{base_filename}.pt')
    }

# 2024/12/9
class ArcadeDetector:
    def __init__(self, auth_file='/workspace/Mask2Former/auth/auth.json'):
        self.auth_file = auth_file
        self.session = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504, 10054],
            allowed_methods=["POST"]
        )
        self.session.mount('https://', HTTPAdapter(max_retries=retries))

    def get_auth(self, auth_file):
        with open(auth_file) as j:
            auth = json.load(j)
            return auth['username'], auth['password']

    def detect_arcade(self, image_path):
        url = 'https://arcade.sgis.tw/'
        auth = self.get_auth(self.auth_file)
        
        with open(image_path, "rb") as img:
            form_data = {"file": img}
            response = self.session.post(url=url, auth=auth, files=form_data, timeout=60)
            return response.json()
    def save_detection_results(self, arcade_json, output_base_path):
        """Save detection results to output folder"""
        try:
            # decode base64
            query_img = arcade_json['image'].split('data:image/png;base64,')[1]
            result_img = arcade_json['image1'].split('data:image/png;base64,')[1]
            labels = arcade_json['labels']

            if not labels:
                print(f"No labels detected for {output_base_path}")
                return None

            # save gsv
            query_img_data = base64.b64decode(query_img)
            with open(f"{output_base_path}_gsv.png", 'wb') as f:
                f.write(query_img_data)

            # save result
            result_img_data = base64.b64decode(result_img)
            with open(f"{output_base_path}_result.png", 'wb') as f:
                f.write(result_img_data)

            # save labels to txt
            with open(f"{output_base_path}.txt", 'w') as f:
                f.write(labels)

            return labels

        except KeyError as e:
            print(f"Missing key in arcade_json: {str(e)}")
            raise
        except Exception as e:
            print(f"Error saving detection results: {str(e)}")
            raise


# 2024/12/18 add
def cleanup_old_files():
    """cleanup old files"""
    try:
        # clear upload directory
        for file in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                app.logger.error(f'Error deleting {file_path}: {str(e)}')

        # clear output directory
        for file in os.listdir(OUTPUT_FOLDER):
            file_path = os.path.join(OUTPUT_FOLDER, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                app.logger.error(f'Error deleting {file_path}: {str(e)}')
    except Exception as e:
        app.logger.error(f'Error during cleanup: {str(e)}')

def cleanup_resources():
    """cleanup resources"""
    import gc
    try:
        # force garbage collection
        gc.collect()
        
        # if cuda is available, empty the cuda cache
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    except Exception as e:
        app.logger.error(f'Error during resource cleanup: {str(e)}')



# resize the image
def resize_if_needed(image, max_dimension=1024):
    """
    check if the image needs to be resized
    """
    width, height = image.size
    
    if max(width, height) > max_dimension:
        # calculate the scaling factor
        scale = max_dimension / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # use PIL to resize
        image = image.resize((new_width, new_height), Image.LANCZOS)
        
    return image

def normalize_image_extension(filepath):
    """
    lowercase the image extension
    """
    directory = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    name, ext = os.path.splitext(filename)
    
    # convert extension to lowercase
    new_filename = name + ext.lower()
    new_filepath = os.path.join(directory, new_filename)
    
    if filepath != new_filepath:
        os.rename(filepath, new_filepath)
        
    return new_filepath