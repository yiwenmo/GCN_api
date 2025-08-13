// image-scaling.js
function calculateImageDimensions(originalWidth, originalHeight, maxWidth, maxHeight) {
    const aspectRatio = originalWidth / originalHeight;
    
    // Calculate initial scale factor
    let scale = Math.min(
        maxWidth / originalWidth,
        maxHeight / originalHeight,
        1  // Limit scale to a maximum of 1 (no upscaling)
    );
    
    // Calculate display size
    let displayWidth = Math.floor(originalWidth * scale);
    let displayHeight = Math.floor(originalHeight * scale);
    
    // Ensure scaled size does not exceed max constraints
    if (displayWidth > maxWidth) {
        displayWidth = maxWidth;
        displayHeight = Math.floor(displayWidth / aspectRatio);
    }
    if (displayHeight > maxHeight) {
        displayHeight = maxHeight;
        displayWidth = Math.floor(displayHeight * aspectRatio);
    }
    
    // Compute final scale ratios
    const scaleX = originalWidth / displayWidth;
    const scaleY = originalHeight / displayHeight;
    
    return {
        width: displayWidth,
        height: displayHeight,
        scaleX,
        scaleY
    };
}

// Convert display coordinates to original image coordinates
function displayToOriginalCoords(bbox, scaleX, scaleY) {
    return {
        x: Math.round(bbox.x * scaleX),
        y: Math.round(bbox.y * scaleY),
        width: Math.round(bbox.width * scaleX),
        height: Math.round(bbox.height * scaleY)
    };
}

// Convert original image coordinates to display coordinates
function originalToDisplayCoords(bbox, scaleX, scaleY) {
    return {
        x: Math.round(bbox.x / scaleX),
        y: Math.round(bbox.y / scaleY),
        width: Math.round(bbox.width / scaleX),
        height: Math.round(bbox.height / scaleY)
    };
}

// Initialize image display with proper scaling
function initializeImageDisplay(imageUrl, container, maxWidth, maxHeight) {
    return new Promise((resolve) => {
        const img = new Image();
        
        img.onload = function() {
            const originalWidth = this.width;
            const originalHeight = this.height;
            
            // Calculate display dimensions and scale ratios
            const dimensions = calculateImageDimensions(
                originalWidth, 
                originalHeight,
                maxWidth,
                maxHeight
            );

            // Apply container styling
            container.style.width = `${dimensions.width}px`;
            container.style.height = `${dimensions.height}px`;
            container.style.backgroundImage = `url(${imageUrl})`;
            container.style.backgroundSize = 'contain'; // Keep aspect ratio
            container.style.backgroundPosition = 'center';
            container.style.backgroundRepeat = 'no-repeat';
            container.style.position = 'relative';
            container.style.margin = '0 auto';

            // Store scale info globally
            window.imageScaleInfo = {
                displayWidth: dimensions.width,
                displayHeight: dimensions.height,
                originalWidth,
                originalHeight,
                scaleX: dimensions.scaleX,
                scaleY: dimensions.scaleY
            };

            // Return calculated scale info
            resolve(window.imageScaleInfo);
        };

        img.src = imageUrl;
    });
}

// Convert bounding box to YOLO format
function convertToYoloBBox(bbox, imageWidth, imageHeight) {
    // Compute normalized coordinates
    const x_center = (bbox.x + bbox.width / 2) / imageWidth;
    const y_center = (bbox.y + bbox.height / 2) / imageHeight;
    const norm_width = bbox.width / imageWidth;
    const norm_height = bbox.height / imageHeight;

    // Return YOLO format string: <class> <x_center> <y_center> <width> <height> <score>
    return `0 ${x_center} ${y_center} ${norm_width} ${norm_height} 0.6`;
}

export {
    calculateImageDimensions,
    displayToOriginalCoords,
    originalToDisplayCoords,
    initializeImageDisplay,
    convertToYoloBBox
};
