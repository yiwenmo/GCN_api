// Maintain original drag and drop upload handling
var fileDrag = document.getElementById("file-drag");
var fileSelect = document.getElementById("file-upload");

fileDrag.addEventListener("dragover", fileDragHover, false);
fileDrag.addEventListener("dragleave", fileDragHover, false);
fileDrag.addEventListener("drop", fileSelectHandler, false);
fileSelect.addEventListener("change", fileSelectHandler, false);

function fileDragHover(e) {
    e.preventDefault();
    e.stopPropagation();
    
    if (e.type === "dragover") {
        fileDrag.classList.add("dragover");
    } else {
        fileDrag.classList.remove("dragover");
    }
}

function fileSelectHandler(e) {
    var files = e.target.files || e.dataTransfer.files;
    fileDragHover(e);
    for (var i = 0, f; (f = files[i]); i++) {
        previewFile(f);
    }
}

// Global variables
let currentFileName = '';
let currentFilePath = '';
let currentImageData = '';
let panopticJsonPath = '';
window.panopticJsonPath = '';

const elements = {
    imagePreview: document.getElementById("image-preview"),
    uploadCaption: document.getElementById("upload-caption"),
    panopticDisplay: document.getElementById("panoptic-display"),
    yoloDisplay: document.getElementById("yolo-display"),
    gcnDisplay: document.getElementById("gcn-display"),
    panopticMessage: document.getElementById("panoptic-message"),
    yoloMessage: document.getElementById("yolo-message"),
    gcnMessage: document.getElementById("gcn-message"),
    panopticLoader: document.getElementById("panoptic-loader"),
    yoloLoader: document.getElementById("yolo-loader"),
    gcnLoader: document.getElementById("gcn-loader"),
    manualAnnotateBtn: document.getElementById("manual-annotate-btn")
};

// File preview
function previewFile(file) {
    // ensure file name is normalized
    let fileName = file.name;
    let fileNameParts = fileName.split('.');
    if (fileNameParts.length > 1) {
        let extension = fileNameParts.pop().toLowerCase();
        currentFileName = fileNameParts.join('.') + '.' + extension;
    } else {
        currentFileName = fileName;
    }
    
    console.log("Normalized file name:", currentFileName);

    var reader = new FileReader();
    reader.onloadend = () => {
        currentImageData = reader.result;
        console.log("Read file data length:", currentImageData.length);
        
        // create object URL
        const objectUrl = URL.createObjectURL(file);
        console.log("Created object URL:", objectUrl);
        window.originalImageUrl = objectUrl;

        elements.imagePreview.src = objectUrl;
        console.log("Set preview image src");
        
        show(elements.imagePreview);
        hide(elements.uploadCaption);
        
        // Reset all results
        resetResults();

        // hide manual annotate button
        if (elements.manualAnnotateBtn) {
            hide(elements.manualAnnotateBtn);
        }
    };
    reader.readAsDataURL(file);
    console.log("Started reading file");
}

// Modified submitImage function
async function submitImage() {
    if (!currentImageData) {
        window.alert("請選擇一張圖片");
        return;
    }

    // Reset all results and show loaders
    resetResults();
    showAllLoaders();

    try {
        // Step 1: Panoptic Segmentation
        const panopticResponse = await fetch("/api/process/panoptic", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                image: currentImageData,
                filename: currentFileName || 'upload.png'
            })
        });

        if (!panopticResponse.ok) {
            throw new Error('Panoptic segmentation failed');
        }

        const panopticData = await panopticResponse.json();
        // Display panoptic results immediately
        displayPanopticResults(panopticData);

        // 保存重要的路徑和文件名
        currentFilePath = panopticData.filepath;
        panopticJsonPath = panopticData.panoptic_result.files.json_path;
        window.panopticJsonPath = panopticJsonPath;

        // Step 2: YOLO Detection
        const yoloResponse = await fetch("/api/process/yolo", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                filepath: panopticData.filepath,
                filename: panopticData.filename
            })
        });

        if (!yoloResponse.ok) {
            throw new Error('YOLO detection failed');
        }

        const yoloData = await yoloResponse.json();
        // Display YOLO results as soon as they're available
        displayYoloResults(yoloData);

        // show manual annotate button, if it doesn't exist yet
        if (elements.manualAnnotateBtn) {
            show(elements.manualAnnotateBtn);
        }

        // Step 3: GCN Processing
        if (yoloData.status === 'success') {
            const gcnResponse = await fetch("/api/process/gcn", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    panoptic_json_path: panopticData.panoptic_result.files.json_path,
                    yolo_status: yoloData.status
                })
            });

            if (!gcnResponse.ok) {
                throw new Error('GCN processing failed');
            }

            const gcnData = await gcnResponse.json();
            // Display GCN results
            displayGcnResults(gcnData);
        } else {
            // if YOLO fails
            hide(elements.gcnLoader);
            elements.gcnMessage.textContent = "需要手動標註邊界框後才能進行 GCN 處理";
            show(elements.gcnMessage);
        }

    } catch (err) {
        console.error("Error occurred:", err);
        window.alert("處理圖片時發生錯誤，請重試。");
        hideAllLoaders();
    }
}

// Separate display functions for each result type
function displayPanopticResults(data) {
    hide(elements.panopticLoader);
    if (data.panoptic_result && data.panoptic_result.image) {
        elements.panopticDisplay.src = data.panoptic_result.image;
        show(elements.panopticDisplay);
        hide(elements.panopticMessage);
    } else {
        hide(elements.panopticDisplay);
        elements.panopticMessage.textContent = "No panoptic segmentation result";
        show(elements.panopticMessage);
    }
}

function displayYoloResults(data) {
    hide(elements.yoloLoader);
    
    if (data.status === 'success' && data.yolo_result && data.yolo_result.image) {
        elements.yoloDisplay.src = data.yolo_result.image;
        show(elements.yoloDisplay);
        hide(elements.yoloMessage);
    } else {
        hide(elements.yoloDisplay);
        elements.yoloMessage.textContent = data.status === 'no_detection' ? 
            "無法自動偵測邊界框，請使用手動標註功能" : "YOLOv5 無法處理該圖片";
        show(elements.yoloMessage);
    }
}

function displayGcnResults(data) {
    hide(elements.gcnLoader);
    if (data.gcn_result && data.gcn_result.image) {
        elements.gcnDisplay.src = data.gcn_result.image;
        show(elements.gcnDisplay);
        hide(elements.gcnMessage);
    } else {
        hide(elements.gcnDisplay);
        elements.gcnMessage.textContent = "No GCN results";
        show(elements.gcnMessage);
    }
}

// check for new gcn results
async function checkForNewGcnResults() {
    try {
        console.log("開始檢查新的 GCN 結果...");
        console.log("目前的檔案名稱:", currentFileName);
       
        show(elements.gcnLoader);
        hide(elements.gcnDisplay);
        hide(elements.gcnMessage);
       
        // add timestamp
        const timestamp = new Date().getTime();
        const url = `/api/check-results?filename=${encodeURIComponent(currentFileName)}&t=${timestamp}`;
        console.log("請求 URL:", url);
       
        const response = await fetch(url);
        console.log("回應狀態:", response.status);
       
        if (response.ok) {
            const data = await response.json();
            console.log("回應資料:", data);
           
            if (data.status === 'success' && data.gcn_result && data.gcn_result.image) {
                console.log("發現 GCN 結果圖像，設定圖像來源");
                
                // check if the image URL is valid
                if (!data.gcn_result.image.startsWith('data:image/')) {
                    console.error("圖像 URL 格式不正確:", data.gcn_result.image.substring(0, 30) + "...");
                    elements.gcnMessage.textContent = "GCN 結果圖像格式不正確";
                    show(elements.gcnMessage);
                    hide(elements.gcnLoader);
                    return;
                }
               
                // setup onload and onerror event handlers
                elements.gcnDisplay.onload = function() {
                    console.log("GCN 圖像載入成功");
                    show(elements.gcnDisplay);
                    hide(elements.gcnMessage);
                    hide(elements.gcnLoader);
                };
                
                elements.gcnDisplay.onerror = function(e) {
                    console.error("GCN 圖像載入失敗", e);
                    console.log("嘗試直接顯示 data URL");
                    
                    // if the image URL is not valid, try to display the data URL
                    elements.gcnDisplay.src = data.gcn_result.image;
                    show(elements.gcnDisplay);
                    hide(elements.gcnLoader);
                };
                
                elements.gcnDisplay.src = data.gcn_result.image;
                
                
                setTimeout(() => {
                    if (elements.gcnLoader.classList.contains('hidden') === false) {
                        console.log("圖像載入逾時，強制顯示");
                        hide(elements.gcnLoader);
                        show(elements.gcnDisplay);
                    }
                }, 5000);
               
            } else if (data.status === 'pending') {
                console.log("GCN 結果尚未產生，繼續等待");
                elements.gcnMessage.textContent = "GCN 處理中，請稍候...";
                show(elements.gcnMessage);
                hide(elements.gcnLoader);
                
                // 5 seconds later
                setTimeout(checkForNewGcnResults, 5000);
            } else {
                console.log("沒有找到有效的 GCN 結果");
                elements.gcnMessage.textContent = data.message || "無法取得 GCN 結果";
                show(elements.gcnMessage);
                hide(elements.gcnLoader);
            }
        } else {
            console.error("API 回應失敗:", response.statusText);
            elements.gcnMessage.textContent = "檢查 GCN 結果失敗";
            show(elements.gcnMessage);
            hide(elements.gcnLoader);
        }
    } catch (error) {
        console.error("檢查 GCN 結果時發生錯誤:", error);
        elements.gcnMessage.textContent = "取得 GCN 結果時發生錯誤：" + error.message;
        show(elements.gcnMessage);
        hide(elements.gcnLoader);
    }
}


// if manual annotate button exists, add event listener
if (elements.manualAnnotateBtn) {
    elements.manualAnnotateBtn.addEventListener('click', openAnnotationWindow);
}

// show all loaders
function showAllLoaders() {
    show(elements.panopticLoader);
    show(elements.yoloLoader);
    show(elements.gcnLoader);
}

// hide all loaders
function hideAllLoaders() {
    hide(elements.panopticLoader);
    hide(elements.yoloLoader);
    hide(elements.gcnLoader);
}

// Modify clear function
function clearImage() {
    if (window.originalImageUrl) {
        URL.revokeObjectURL(window.originalImageUrl);
        window.originalImageUrl = null;
    }
    
    currentFileName = '';
    currentImageData = '';
    currentFilePath = '';
    panopticJsonPath = '';
    window.panopticJsonPath = '';
    
    fileSelect.value = "";
    elements.imagePreview.src = "";
    
    hide(elements.imagePreview);
    show(elements.uploadCaption);
    
    resetResults();
    
    // hide manual annotate button
    if (elements.manualAnnotateBtn) {
        hide(elements.manualAnnotateBtn);
    }
}

function resetResults() {
    // reset all results
    hide(elements.panopticDisplay);
    hide(elements.yoloDisplay);
    hide(elements.gcnDisplay);
    hide(elements.panopticMessage);
    hide(elements.yoloMessage);
    hide(elements.gcnMessage);
}

// Helper functions
function show(element) {
    if (element) element.classList.remove('hidden');
}

function hide(element) {
    if (element) element.classList.add('hidden');
}

// add message event listener
window.addEventListener('message', function(event) {
    console.log('收到訊息:', event.data);
    
    // check for refresh_gcn_results event
    if (event.data && event.data.type === 'refresh_gcn_results') {
        console.log('正在重新整理 GCN 結果...');
        // refresh gcn results
        resetGcnSection();
        
        // setup retry logic
        let retryCount = 0;
        const maxRetries = 3;
        
        function checkWithRetry() {
            console.log(`執行 GCN 結果檢查（嘗試第 ${retryCount + 1}/${maxRetries} 次）`);
            checkForNewGcnResults().then(() => {
                // check if GCN result is displayed
                if (elements.gcnDisplay.classList.contains('hidden') && 
                    !elements.gcnLoader.classList.contains('hidden') && 
                    retryCount < maxRetries) {
                    retryCount++;
                    console.log(`GCN 結果尚未顯示，將於 3 秒後重試（第 ${retryCount}/${maxRetries} 次）`);
                    setTimeout(checkWithRetry, 3000);
                }
            });
        }
        
        setTimeout(checkWithRetry, 2000);
    }
});


// add reset function
function resetGcnSection() {
    hide(elements.gcnDisplay);
    hide(elements.gcnMessage);
    show(elements.gcnLoader);
}

function openAnnotationWindow() {
    if (!currentFileName || !panopticJsonPath) {
        window.alert("請先上傳並處理圖片");
        return;
    }

    // Log important parameters to console
    console.log("開啟標註視窗，參數:", {
        filename: currentFileName,
        filepath: currentFilePath,
        jsonPath: panopticJsonPath,
        originalImageUrl: window.originalImageUrl
    });

    // Build the annotation page URL with query parameters
    const annotateUrl = `/annotate.html?filename=${encodeURIComponent(currentFileName)}&filepath=${encodeURIComponent(currentFilePath)}&json_path=${encodeURIComponent(panopticJsonPath)}`;
    
    // Open a new popup window for annotation
    const annotateWindow = window.open(annotateUrl, "手動標註", "width=1200,height=900,resizable=yes,scrollbars=yes");
    
    // If window opened successfully
    if (annotateWindow) {
        console.log("標註視窗已成功開啟");

        // Listen for messages sent from the annotation window
        window.addEventListener('message', function messageHandler(event) {
            if (event.data && event.data.type === 'refresh_gcn_results') {
                console.log('收到標註視窗的刷新請求');
                // Remove this temporary listener to avoid duplicate handling
                window.removeEventListener('message', messageHandler);
            }
        });

        // When the annotation window is about to close
        annotateWindow.addEventListener('beforeunload', () => {
            console.log("標註視窗正在關閉，即將檢查 GCN 結果");
            // Delay the check slightly to allow backend processing to complete
            setTimeout(function() {
                console.log("標註視窗關閉後檢查 GCN 結果");
                checkForNewGcnResults();
            }, 2000);
        });
    } else {
        // If the popup was blocked, notify the user
        window.alert("彈出窗口被阻擋，請允許網站開啟彈出窗口");
    }
}
