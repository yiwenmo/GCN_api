# Flask
from flask import Flask, request, render_template, jsonify, send_file
from gevent.pywsgi import WSGIServer

# Some utilites
import os
import base64
import subprocess
import logging
import json
from werkzeug.utils import secure_filename
from utils import UPLOAD_FOLDER, OUTPUT_FOLDER, DATA_FOLDER, base64_to_image, process_image_file, create_response_data, get_result_paths, ArcadeDetector, cleanup_old_files, cleanup_resources, normalize_image_extension
from image.src.gcn_processor import GCNProcessor

from datetime import datetime
from werkzeug.utils import secure_filename

from PIL import Image # 確保檔案頂部有這個匯入

# Declare a flask app
app = Flask(__name__, template_folder='templates', static_folder='static')

# add log file
logging.basicConfig(level=logging.INFO)

# Ensure upload and output folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/', methods=['GET'])
def index():
    # Main page
    app.logger.info("Main page done")
    return render_template('index.html')

# add annotate route (annotate.html)
@app.route('/annotate.html', methods=['GET'])
def annotate():
    """load and pass existing bounding boxes"""
    app.logger.info("收到標註頁面請求")
    
    # get filename from query parameters
    filename = request.args.get('filename')
    if filename:
        # try to read existing bounding box file txt
        base_name = os.path.splitext(secure_filename(filename))[0]
        bbox_path = os.path.join(OUTPUT_FOLDER, f'{base_name}.txt')
        
        existing_bboxes = []
        if os.path.exists(bbox_path):
            try:
                with open(bbox_path, 'r') as f:
                    # existing_bboxes = f.read()
                    lines = f.readlines()

                    # exclude fake bbox
                    for line in lines:
                        values = line.strip().split()
                        if len(values) >= 6:
                            # 檢查是否為假邊界框（根據其固定的座標和信心分數）
                            center_x = float(values[1])
                            center_y = float(values[2])
                            width = float(values[3])
                            height = float(values[4])
                            score = float(values[5])
                            if not (center_x == 0.5000 and center_y == 0.5000 and width == 0.2000 and height == 0.2000 and score == 0.3000):
                                existing_bboxes.append(line.strip())
                    
                    # to string
                    existing_bboxes = '\n'.join(existing_bboxes)        

                app.logger.info(f"找到 {filename} 的現有邊界框")
            except Exception as e:
                app.logger.error(f"讀取邊界框檔案時發生錯誤: {str(e)}")
        
        # let the bounding boxes pass as a template variable
        return render_template('annotate.html', existing_bboxes=existing_bboxes)
    
    return render_template('annotate.html')


# panoptic
@app.route('/api/process/panoptic', methods=['POST'])
def process_panoptic():
    """Handle panoptic segmentation stage"""
    try:
        # cleanup upload and output directories
        cleanup_old_files()

        if request.is_json:
            data = request.get_json()
            image_data = data.get('image', '')
            filename = data.get('filename', 'upload.png')

            # decode base64
            img = base64_to_image(image_data)
            filename = secure_filename(filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            img.save(filepath)
        
            # standardize file extension
            filepath = normalize_image_extension(filepath)
            filename = os.path.basename(filepath)
        else:
            if 'file' not in request.files:
                return jsonify({'error': 'No file part'}), 400
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No selected file'}), 400

            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
        
        # Process panoptic segmentation
        app.logger.info("Starting panoptic segmentation...")
        panoptic_image_path, panoptic_json_path = process_image_file(filepath)

        with open(panoptic_json_path, 'r') as f:
            panoptic_info = json.load(f)
        with open(panoptic_image_path, 'rb') as f:
            panoptic_image = base64.b64encode(f.read()).decode('utf-8')
        
        return jsonify({
            'status': 'success',
            'filepath': filepath,
            'filename': filename,
            'panoptic_result': {
                'image': f"data:image/png;base64,{panoptic_image}",
                'seg_info': panoptic_info,
                'files': {
                    'image_path': panoptic_image_path,
                    'json_path': panoptic_json_path
                }
            }
        }), 200

    except Exception as e:
        cleanup_resources()
        app.logger.error(f'Panoptic API Error: {str(e)}')
        return jsonify({'error': str(e)}), 500

# yolo
@app.route('/api/process/yolo', methods=['POST'])
def process_yolo():
    """Handle YOLO detection stage"""
    try:
        data = request.get_json()
        filepath = data.get('filepath')
        filename = data.get('filename')
        
        if not filepath or not filename:
            return jsonify({'error': 'Missing filepath or filename'}), 400

        # YOLOv5 detection
        app.logger.info("Starting YOLOv5 detection...")
        detector = ArcadeDetector()
        arcade_json = detector.detect_arcade(filepath)
        output_base_path = os.path.join(OUTPUT_FOLDER, os.path.splitext(filename)[0])
        yolo_labels = detector.save_detection_results(arcade_json, output_base_path)
        
        # 添加調試日誌
        app.logger.info(f"YOLO labels type: {type(yolo_labels)}")
        app.logger.info(f"YOLO labels content: {yolo_labels}")

        if yolo_labels is None or not yolo_labels:
            return jsonify({
                'status': 'no_detection',
                'message': 'No detection results found'
            }), 200

        # read YOLOv5 result
        yolo_result_path = f"{output_base_path}_result.png"
        yolo_image = None
        if os.path.exists(yolo_result_path):
            with open(yolo_result_path, 'rb') as f:
                yolo_image = base64.b64encode(f.read()).decode('utf-8')

        return jsonify({
            'status': 'success',
            'yolo_result': {
                'base_path': output_base_path,
                'labels': yolo_labels,
                'image': f"data:image/png;base64,{yolo_image}" if yolo_image else None,
                'result_path': f"{output_base_path}_result.png",
                'labels_path': f"{output_base_path}.txt"
            }
        }), 200

    except Exception as e:
        app.logger.error(f'YOLO API Error: {str(e)}')
        return jsonify({'error': str(e)}), 500

# gcn
@app.route('/api/process/gcn', methods=['POST'])
def process_gcn():
    """Handle GCN prediction stage"""
    try:
        data = request.get_json()
        panoptic_json_path = data.get('panoptic_json_path')
        yolo_status = data.get('yolo_status', 'success')  # 預設為 success
        filename = data.get('filename')  # Add parameter for filename

        if not panoptic_json_path:
            return jsonify({'error': 'Missing panoptic_json_path'}), 400

        # if no_detection, skip gcn process
        if yolo_status == 'no_detection' and yolo_status != 'manual_annotation':
            return jsonify({
                'status': 'skipped',
                'message': 'GCN processing skipped due to no YOLO detection'
            }), 200

        # GCN processing
        app.logger.info("Starting GCN prediction...")
        gcn_processor = GCNProcessor()
        gcn_results = gcn_processor.process(panoptic_json_path)

        # Extract base filename from panoptic_json_path if not provided
        if not filename:
            base_name = os.path.splitext(os.path.basename(panoptic_json_path))[0]
        else:
            base_name = os.path.splitext(filename)[0]

        # read GCN result
        gcn_image = None
        gcn_image_path = None
        
        if gcn_results and 'visualization' in gcn_results and 'image_path' in gcn_results['visualization']:
            try:
                gcn_image_path = gcn_results['visualization']['image_path']
                app.logger.info(f"Reading GCN result image from: {gcn_image_path}")
                
                # Ensure the file is saved with the correct filename pattern
                target_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_gcn_result.png")
                
                # If the file exists but with a different name, make a copy with the correct name
                if os.path.exists(gcn_image_path) and gcn_image_path != target_path:
                    import shutil
                    shutil.copy2(gcn_image_path, target_path)
                    app.logger.info(f"Copied GCN result to standardized path: {target_path}")
                    gcn_image_path = target_path
                
                with open(gcn_image_path, 'rb') as f:
                    gcn_image = base64.b64encode(f.read()).decode('utf-8')
            except Exception as e:
                app.logger.error(f"Error reading GCN image: {str(e)}")
                
                # If there was an error but we have prediction results, try to generate a visualization
                if gcn_results.get('predictions'):
                    try:
                        # Create a basic visualization from the predictions
                        app.logger.info("Attempting to create visualization from prediction results")
                        
                        # Get the original image
                        original_image_path = os.path.join(UPLOAD_FOLDER, f"{base_name}.png")
                        if not os.path.exists(original_image_path):
                            # Try with other extensions
                            for ext in ['.jpg', '.jpeg', '.PNG', '.JPEG', '.JPG']:
                                test_path = os.path.join(UPLOAD_FOLDER, f"{base_name}{ext}")
                                if os.path.exists(test_path):
                                    original_image_path = test_path
                                    break
                        
                        if os.path.exists(original_image_path):
                            # Create a basic visualization with the predictions
                            from PIL import Image, ImageDraw, ImageFont
                            img = Image.open(original_image_path)
                            draw = ImageDraw.Draw(img)
                            
                            # Add prediction text to the image
                            y_position = 10
                            for pred in gcn_results.get('predictions', []):
                                prediction_text = f"Prediction: {pred.get('label', 'Unknown')}, Score: {pred.get('score', 0):.2f}"
                                draw.text((10, y_position), prediction_text, fill=(255, 0, 0))
                                y_position += 20
                            
                            # Save the visualization
                            target_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_gcn_result.png")
                            img.save(target_path)
                            app.logger.info(f"Created fallback visualization: {target_path}")
                            
                            # Update the gcn_image_path and read the image data
                            gcn_image_path = target_path
                            with open(gcn_image_path, 'rb') as f:
                                gcn_image = base64.b64encode(f.read()).decode('utf-8')
                    except Exception as viz_error:
                        app.logger.error(f"Error creating fallback visualization: {str(viz_error)}")

        return jsonify({
            'status': 'success',
            'gcn_result': {
                'image': f"data:image/png;base64,{gcn_image}" if gcn_image else None,
                'predictions': gcn_results.get('predictions', []),
                'image_path': gcn_image_path
            }
        }), 200

    except Exception as e:
        app.logger.error(f'GCN API Error: {str(e)}')
        return jsonify({'error': str(e)}), 500

# Add check-results route
@app.route('/api/check-results', methods=['GET'])
def check_results():
    """Check whether the GCN result exists"""
    try:
        filename = request.args.get('filename')
        app.logger.info(f"檢查 {filename} 的 GCN 結果")
        
        if not filename:
            app.logger.error("缺少 filename 變數")
            return jsonify({'error': 'Missing filename'}), 400
            
        # Sanitize and extract base name
        base_name = os.path.splitext(secure_filename(filename))[0]
        app.logger.info(f"檔案名: {base_name}")
        
        # Define expected GCN result path
        gcn_image_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_gcn_result.png")
        app.logger.info(f"找尋 GCN 結果圖片: {gcn_image_path}")
        
        # Attempt to locate file if not immediately found
        if not os.path.exists(gcn_image_path):
            app.logger.warning(f"GCN 結果圖片不存在: {gcn_image_path}")
            
            all_files = os.listdir(OUTPUT_FOLDER)
            possible_patterns = [
                f"{base_name}_gcn_result.png",
                f"{base_name}_result.png"
            ]
            
            # Try exact filename matches
            for pattern in possible_patterns:
                if pattern in all_files:
                    gcn_image_path = os.path.join(OUTPUT_FOLDER, pattern)
                    app.logger.info(f"找到匹配的檔名: {gcn_image_path}")
                    break
            
            # If still not found, try partial matches
            if not os.path.exists(gcn_image_path):
                matching_files = [f for f in all_files if '_gcn_result.png' in f or base_name in f]
                app.logger.info(f"OUTPUT_FOLDER 中找到匹配的文件: {matching_files}")
                
                # Attempt fallback creation if no match
                if not matching_files:
                    result_files = [f for f in all_files if 'result.png' in f or 'gcn' in f]
                    if result_files:
                        result_files.sort(key=lambda x: os.path.getmtime(os.path.join(OUTPUT_FOLDER, x)), reverse=True)
                        gcn_image_path = os.path.join(OUTPUT_FOLDER, result_files[0])
                        app.logger.info(f"使用最新的結果: {gcn_image_path}")
                    else:
                        app.logger.warning("找不到任何結果文件，嘗試建立一個")
                        try:
                            original_image = None
                            for folder in [UPLOAD_FOLDER, 'host_upload']:
                                for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPEG']:
                                    test_path = os.path.join(folder, f"{base_name}{ext}")
                                    if os.path.exists(test_path):
                                        original_image = test_path
                                        break
                                if original_image:
                                    break
                            
                            if original_image:
                                # Create a placeholder result image
                                from PIL import Image, ImageDraw
                                img = Image.open(original_image)
                                draw = ImageDraw.Draw(img)
                                draw.text((10, 10), "GCN 分析已完成", fill=(255, 0, 0))
                                draw.text((10, 30), "結果尚未視覺化", fill=(255, 0, 0))
                                
                                gcn_image_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_gcn_result.png")
                                img.save(gcn_image_path)
                                app.logger.info(f"建立了新的結果圖片: {gcn_image_path}")
                            else:
                                app.logger.error("找不到原始圖像")
                                return jsonify({
                                    'status': 'pending',
                                    'message': 'GCN 結果尚未產生，且無法建立替代圖像'
                                }), 200
                        except Exception as create_error:
                            app.logger.error(f"建立結果圖像失敗: {str(create_error)}")
                            return jsonify({
                                'status': 'error',
                                'message': f'建立 GCN 結果圖像失敗: {str(create_error)}'
                            }), 500
                else:
                    # Use best match if available
                    best_matches = [f for f in matching_files if base_name in f and 'gcn_result' in f]
                    if best_matches:
                        gcn_image_path = os.path.join(OUTPUT_FOLDER, best_matches[0])
                    else:
                        gcn_image_path = os.path.join(OUTPUT_FOLDER, matching_files[0])
                    
                    app.logger.info(f"使用最匹配的 GCN 結果檔案: {gcn_image_path}")
        
        # Final check if file exists
        if not os.path.exists(gcn_image_path):
            app.logger.error(f"最終找不到 GCN 圖像檔案: {gcn_image_path}")
            return jsonify({
                'status': 'pending',
                'message': 'GCN 結果尚未產生'
            }), 200
        
        # Attempt to read and encode image
        app.logger.info(f"正在讀取 GCN 圖像: {gcn_image_path}")
        try:
            if not os.path.getsize(gcn_image_path) > 0:
                app.logger.error(f"GCN 圖像檔案為空: {gcn_image_path}")
                return jsonify({
                    'status': 'error',
                    'message': 'GCN 圖像檔案為空'
                }), 500
                
            with open(gcn_image_path, 'rb') as f:
                image_data = f.read()
                gcn_image = base64.b64encode(image_data).decode('utf-8')
                app.logger.info(f"成功讀取圖像，大小: {len(image_data)}，Base64: {len(gcn_image)}")
                
                try:
                    test_decode = base64.b64decode(gcn_image)
                    app.logger.info(f"Base64 解碼驗證成功: {len(test_decode)} bytes")
                except Exception as decode_error:
                    app.logger.error(f"Base64 解碼失敗: {str(decode_error)}")
        except Exception as e:
            app.logger.error(f"讀取圖像失敗: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': f'讀取 GCN 圖像失敗: {str(e)}'
            }), 500
        
        # Prepare data URI
        image_url = f"data:image/png;base64,{gcn_image}"
        app.logger.info(f"圖片資料 URL 前綴: {image_url[:50]}...")
        
        # Optional: Load prediction data if available
        predictions = []
        bbox_info_updated_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_bbox_info_updated.json")
        if os.path.exists(bbox_info_updated_path):
            try:
                with open(bbox_info_updated_path, 'r') as f:
                    bbox_info = json.load(f)
                    if 'bbox_info' in bbox_info:
                        for bbox in bbox_info['bbox_info']:
                            if 'seg' in bbox:
                                for seg_item in bbox['seg']:
                                    prediction = {
                                        'label': seg_item.get('category', 'Unknown'),
                                        'score': seg_item.get('bbox_percent', 0.0),
                                        'category_id': seg_item.get('category_id', 0)
                                    }
                                    predictions.append(prediction)
            except Exception as json_error:
                app.logger.error(f"讀取預測結果失敗: {str(json_error)}")
        
        # Final response
        response_data = {
            'status': 'success',
            'gcn_result': {
                'image': image_url,
                'image_path': gcn_image_path,
                'image_size': len(image_data),
                'predictions': predictions
            }
        }
        
        app.logger.info("返回 GCN 結果資料")
        return jsonify(response_data), 200
        
    except Exception as e:
        app.logger.error(f'檢查結果 API 發生錯誤: {str(e)}')
        import traceback
        app.logger.error(f'錯誤堆疊: {traceback.format_exc()}')
        return jsonify({'error': str(e)}), 500


# update_bbox
# Route to update user-edited bounding boxes and generate GCN-compatible JSON
@app.route('/api/update-bbox', methods=['POST'])
def update_bbox():
    """Update manually annotated bounding boxes and trigger GCN processing"""
    try:
        import sys
        import os
        
        # Attempt relative or absolute import for image_processing
        try:
            from image.src.preprocessing.image_processing import (
                get_image_dimensions, read_seg_json, read_tensor_pt, pt2img,
                read_arcade_result, bbox2mask, extract_by_id, bboxinfo2json,
                process_jsons
            )
        except ImportError:
            project_root = os.path.dirname(os.path.abspath(__file__))
            sys.path.append(project_root)
            app.logger.info(f"Added {project_root} to Python path")
            from image.src.preprocessing.image_processing import (
                get_image_dimensions, read_seg_json, read_tensor_pt, pt2img,
                read_arcade_result, bbox2mask, extract_by_id, bboxinfo2json,
                process_jsons
            )
            app.logger.info("Successfully imported image_processing functions")

        data = request.get_json()
        filename = data.get('filename')
        bbox_data = data.get('bboxes')  # YOLO format bounding boxes

        app.logger.info(f"Processing bbox update for {filename}")

        if not filename or bbox_data is None:
            return jsonify({'error': '缺少必要資料'}), 400

        filename = secure_filename(filename)
        base_name = os.path.splitext(filename)[0]
        bbox_path = os.path.join(OUTPUT_FOLDER, f'{base_name}.txt')

        # Save YOLO bounding boxes to .txt file
        with open(bbox_path, 'w') as f:
            f.write(bbox_data)
        app.logger.info(f"Saved bbox data to {bbox_path}")

        # Prepare for segmentation + bbox intersection
        pt_path = os.path.join(OUTPUT_FOLDER, f'{base_name}.pt')
        json_path = os.path.join(OUTPUT_FOLDER, f'{base_name}.json')

        if not os.path.exists(pt_path):
            app.logger.error(f"PT 檔案不存在: {pt_path}")
            return jsonify({'error': f'找不到PT文件: {pt_path}'}), 404
        if not os.path.exists(json_path):
            app.logger.error(f"JSON 檔案不存在: {json_path}")
            return jsonify({'error': f'找不到JSON文件: {json_path}'}), 404

        # Get image dimensions from .pt
        image_width, image_height = get_image_dimensions(pt_path)
        if image_width is None or image_height is None:
            return jsonify({'error': '無法取得影像尺寸'}), 500

        app.logger.info(f"Image dimensions: {image_width}x{image_height}")
        
        # Read segmentation and image
        seg = read_seg_json(json_path)
        pano_id = read_tensor_pt(pt_path)
        image = pt2img(pt_path)

        # Read manual bounding boxes
        records = read_arcade_result(bbox_path, image_width, image_height)

        # Compute overlap between bbox and segmentation
        intersected_region, bbox_info = bbox2mask(seg, pano_id, image, records, image_width, image_height)
        
        # Save initial bbox_info.json
        bboxinfo2json(base_name, bbox_info, save=True, save_path=OUTPUT_FOLDER)

        # Ensure _gsv.png exists for visualization
        gsv_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_gsv.png")
        if not os.path.exists(gsv_path):
            original_image = None
            for folder in [UPLOAD_FOLDER, 'host_upload', app.static_folder]:
                for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPEG']:
                    test_path = os.path.join(folder, f"{base_name}{ext}")
                    if os.path.exists(test_path):
                        original_image = test_path
                        break
                if original_image:
                    import shutil
                    shutil.copy2(original_image, gsv_path)
                    app.logger.info(f"建立 _gsv.png: {gsv_path}")
                else:
                    app.logger.warning("找不到原始圖像，無法建立 _gsv.png")

        # Generate updated bbox JSON with centroid distance
        image_dimensions = { base_name: (image_width, image_height) }
        process_jsons(OUTPUT_FOLDER, OUTPUT_FOLDER, image_dimensions)

        # Run GCN processing
        try:
            from image.src.gcn_processor import GCNProcessor
            gcn_processor = GCNProcessor()
            gcn_results = gcn_processor.process(json_path)

            # Handle visualization
            gcn_image_path = None
            if gcn_results and 'visualization' in gcn_results and 'image_path' in gcn_results['visualization']:
                gcn_image_path = gcn_results['visualization']['image_path']
                target_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_gcn_result.png")
                if os.path.exists(gcn_image_path) and gcn_image_path != target_path:
                    import shutil
                    shutil.copy2(gcn_image_path, target_path)
                    gcn_image_path = target_path
            else:
                app.logger.warning("GCN 沒有產生可視化，建立備用圖像")
                original_image_path = None
                for folder in [UPLOAD_FOLDER, 'host_upload', app.static_folder]:
                    for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPEG']:
                        test_path = os.path.join(folder, f"{base_name}{ext}")
                        if os.path.exists(test_path):
                            original_image_path = test_path
                            break
                    if original_image_path:
                        from PIL import Image, ImageDraw, ImageFont
                        img = Image.open(original_image_path)
                        draw = ImageDraw.Draw(img)
                        y_position = 10
                        if gcn_results and 'predictions' in gcn_results:
                            for pred in gcn_results['predictions']:
                                prediction_text = f"Prediction: {pred.get('label', 'Unknown')}, Score: {pred.get('score', 0):.2f}"
                                draw.text((10, y_position), prediction_text, fill=(255, 0, 0))
                                y_position += 20
                        else:
                            draw.text((10, y_position), "GCN 分析已完成", fill=(255, 0, 0))
                        target_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_gcn_result.png")
                        img.save(target_path)
                        gcn_image_path = target_path
        except Exception as gcn_error:
            app.logger.error(f"GCN 處理錯誤: {str(gcn_error)}")
            try:
                original_image_path = None
                for folder in [UPLOAD_FOLDER, 'host_upload', app.static_folder]:
                    for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPEG']:
                        test_path = os.path.join(folder, f"{base_name}{ext}")
                        if os.path.exists(test_path):
                            original_image_path = test_path
                            break
                    if original_image_path:
                        from PIL import Image, ImageDraw
                        img = Image.open(original_image_path)
                        draw = ImageDraw.Draw(img)
                        draw.text((10, 10), f"GCN 錯誤: {str(gcn_error)[:50]}", fill=(255, 0, 0))
                        draw.text((10, 30), "手動標註已儲存", fill=(255, 0, 0))
                        target_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_gcn_result.png")
                        img.save(target_path)
            except Exception as img_error:
                app.logger.error(f"建立備用圖像失敗: {str(img_error)}")

        return jsonify({
            'status': 'success',
            'message': '邊界框已更新並處理成功',
            'bbox_path': bbox_path,
            'bbox_info_path': os.path.join(OUTPUT_FOLDER, f'{base_name}_bbox_info.json'),
            'bbox_info_updated_path': os.path.join(OUTPUT_FOLDER, f'{base_name}_bbox_info_updated.json')
        }), 200

    except Exception as e:
        import traceback
        stack_trace = traceback.format_exc()
        app.logger.error(f'更新邊界框時發生錯誤: {str(e)}')
        app.logger.error(f'堆疊追蹤: {stack_trace}')
        return jsonify({'error': str(e), 'stack_trace': stack_trace}), 500


# 添加新的路由來獲取圖片檔案
@app.route('/host_upload/<filename>', methods=['GET'])
def get_upload_image(filename):
    """提供上傳的原始圖片"""
    try:
        # 安全處理檔名
        filename = secure_filename(filename)
        filepath = os.path.join('host_upload', filename)
        
        if not os.path.exists(filepath):
            app.logger.error(f"找不到圖片檔案: {filepath}")
            return jsonify({'error': '找不到圖片檔案'}), 404
            
        return send_file(filepath, mimetype='image/jpeg')
    except Exception as e:
        app.logger.error(f'獲取圖片時發生錯誤: {str(e)}')
        return jsonify({'error': str(e)}), 500

# 添加新的路由來獲取標註用的圖片
@app.route('/api/get-image/<filename>', methods=['GET'])
def get_image_by_name(filename):
    """提供標註用的原始圖片 (路徑參數版本)"""
    try:
        # 安全處理檔名
        filename = secure_filename(filename)
        
        # 嘗試從多個可能的位置尋找圖片
        possible_paths = [
            os.path.join('host_upload', filename),
            os.path.join(UPLOAD_FOLDER, filename),
            os.path.join(OUTPUT_FOLDER, filename),
            os.path.join(app.static_folder, 'uploads', filename),
            os.path.join(app.static_folder, 'output', filename)
        ]
        
        # 尋找存在的圖片路徑
        image_path = None
        for path in possible_paths:
            if os.path.exists(path):
                image_path = path
                app.logger.info(f"找到圖片路徑: {path}")
                break
                
        if not image_path:
            app.logger.error(f"找不到圖片檔案: {filename}")
            return jsonify({'error': '找不到圖片檔案'}), 404
            
        return send_file(image_path)
        
    except Exception as e:
        app.logger.error(f'獲取圖片時發生錯誤: {str(e)}')
        return jsonify({'error': str(e)}), 500

@app.route('/api/annotation-image', methods=['GET'])
def get_annotation_image_by_query():
    """直接提供標註用的原始圖片 (查詢參數版本)"""
    try:
        filename = request.args.get('filename')
        if not filename:
            return jsonify({'error': '缺少filename參數'}), 400
            
        # 安全處理檔名
        filename = secure_filename(filename)
        app.logger.info(f"請求標註圖片: {filename}")
        
        # 嘗試從多個可能的位置尋找圖片
        possible_paths = [
            os.path.join('host_upload', filename),
            os.path.join(UPLOAD_FOLDER, filename),
            os.path.join(OUTPUT_FOLDER, filename),
            os.path.join(app.static_folder, 'uploads', filename),
            os.path.join(app.static_folder, 'output', filename)
        ]
        
        # 尋找存在的圖片路徑
        for path in possible_paths:
            app.logger.info(f"嘗試圖片路徑: {path}")
            if os.path.exists(path):
                app.logger.info(f"找到圖片: {path}")
                return send_file(path)
        
        app.logger.error(f"找不到圖片: {filename}")
        return jsonify({'error': f'找不到圖片: {filename}'}), 404
        
    except Exception as e:
        app.logger.error(f'取得標註圖片時發生錯誤: {str(e)}')
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 7676), app)
    print('Serving on http://127.0.0.1:7676')
    http_server.serve_forever()