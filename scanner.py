import numpy as np
import argparse
import cv2
import imutils
import os
from skimage.filters import threshold_local
from datetime import datetime
from tqdm import tqdm

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    """Modified perspective transform with margin adjustment"""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Calculate width and height
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Add small margins (1%)
    margin_w = int(maxWidth * 0.01)
    margin_h = int(maxHeight * 0.01)
    maxWidth += 2 * margin_w
    maxHeight += 2 * margin_h

    # Adjust destination points to include margins
    dst = np.array([
        [margin_w, margin_h],
        [maxWidth - margin_w, margin_h],
        [maxWidth - margin_w, maxHeight - margin_h],
        [margin_w, maxHeight - margin_h]], dtype="float32")

    # Get transform matrix and apply perspective transform
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

def auto_brightness_contrast(image, clip_hist_percent=1):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate grayscale histogram
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)
    
    # Calculate cumulative distribution
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))
    
    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0
    
    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1
    
    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1
    
    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha
    
    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return auto_result

def enhance_image(image):
    """Enhance image quality for better detection"""
    # Normalize image
    normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    # Sharpen image
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(normalized, -1, kernel)
    return sharpened

def get_document_contour(image):
    """Improved document contour detection with better margin handling"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Bilateral filter preserves edges better
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)
    # Multi-stage edge detection
    edges = cv2.Canny(blurred, 30, 200)
    edges = cv2.dilate(edges, None, iterations=2)
    edges = cv2.erode(edges, None, iterations=1)
    
    # Find contours
    cnts = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    if not cnts:
        return None
        
    # Sort by area and keep top 5
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    
    max_area = 0
    best_cnt = None
    img_area = image.shape[0] * image.shape[1]
    
    for c in cnts:
        peri = cv2.arcLength(c, True)
        # Adjust epsilon values for better corner detection
        for eps in [0.02, 0.015, 0.03, 0.01]:
            approx = cv2.approxPolyDP(c, eps * peri, True)
            area = cv2.contourArea(approx)
            
            # Adjust area threshold to allow more of the document to be visible
            if len(approx) == 4 and area > max_area:
                if area > img_area * 0.05:  # Reduced from 0.1 to 0.05
                    max_area = area
                    best_cnt = approx
    
    return best_cnt

def normalize_size(image, max_dimension=1200):  # Reduced from 2000
    """Normalize image size while maintaining aspect ratio and minimum size"""
    height, width = image.shape[:2]
    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(image, (new_width, new_height))
    return image

class DocumentScanner:
    def __init__(self, config=None):
        self.config = config or self.load_config()
        self.processing_config = self.config['scanner']['processing']

    def load_config(self):
        # Placeholder for actual configuration loading logic
        return {
            'scanner': {
                'processing': {
                    'denoise': True,
                    'auto_contrast': True,
                    'auto_rotate': True
                }
            }
        }

    def auto_rotate(self, image):
        # Placeholder for actual auto-rotate logic
        return image

    def preprocess(self, image):
        # Enhanced preprocessing pipeline
        processed = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(processed)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        processed = cv2.merge((l,a,b))
        processed = cv2.cvtColor(processed, cv2.COLOR_LAB2BGR)
        
        if self.processing_config['denoise']:
            processed = cv2.fastNlMeansDenoisingColored(processed)
            
        if self.processing_config['auto_contrast']:
            processed = auto_brightness_contrast(processed)
            
        if self.processing_config['auto_rotate']:
            processed = self.auto_rotate(processed)
            
        return processed

    def process_batch(self, images, callback=None):
        results = []
        for i, image in enumerate(tqdm(images)):
            try:
                result = self.process_document(image)
                results.append(result)
                if callback:
                    callback(i + 1, len(images), result)
            except Exception as e:
                print(f"Error processing image {i}: {e}")
        return results

    def process_document(self, image):
        # Enhanced document processing
        preprocessed = self.preprocess(image)
        # ...rest of existing process_document implementation...

def process_document(image_input, debug=False):
    try:
        # Load and validate image
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            if image is None:
                raise ValueError("Could not load image from path")
        else:
            image = image_input.copy()

        if image is None:
            raise ValueError("Invalid image input")
            
        # Normalize input image to smaller size
        image = normalize_size(image_input.copy() if isinstance(image_input, np.ndarray) else cv2.imread(image_input), 
                             max_dimension=1200)  # Reduced from 1600
        
        # Enhance image quality
        enhanced = enhance_image(image)
        
        # Process at a smaller size while maintaining aspect ratio
        height = 600  # Reduced from 1000
        ratio = image.shape[0] / height
        proc_image = imutils.resize(enhanced, height=height)
        
        # Get document contour
        screenCnt = get_document_contour(proc_image)
        
        if screenCnt is None:
            # Fallback method
            gray = cv2.cvtColor(proc_image, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(gray, 75, 200)
            cnts = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            if not cnts:
                raise ValueError("No document found in image")
            largest_cnt = max(cnts, key=cv2.contourArea)
            rect = cv2.minAreaRect(largest_cnt)
            screenCnt = np.int32(cv2.boxPoints(rect))
        
        # Perspective transform
        warped = four_point_transform(image, screenCnt.reshape(4, 2) * ratio)
        
        # Limit output size with reduced maximum
        warped = normalize_size(warped, max_dimension=1200)  # Reduced from 2400
        
        # Enhanced thresholding with appropriate window size
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        window_size = min(25, min(warped_gray.shape) // 30 * 2 + 1)  # Adaptive window size
        T = threshold_local(warped_gray, window_size, offset=10, method="gaussian")
        warped_bw = (warped_gray > T).astype("uint8") * 255
        
        # Moderate sharpening
        kernel = np.array([[-0.5,-0.5,-0.5], [-0.5,5,-0.5], [-0.5,-0.5,-0.5]])
        warped_bw = cv2.filter2D(warped_bw, -1, kernel)
        
        # Subtle noise reduction
        warped_bw = cv2.medianBlur(warped_bw, 3)
        
        return warped_bw
        
    except Exception as e:
        raise Exception(f"Error processing document: {str(e)}")

def save_document(image, output_dir="scanned_docs"):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"scan_{timestamp}.png"
    output_path = os.path.join(output_dir, filename)
    
    # Save image
    cv2.imwrite(output_path, image)
    return output_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the image to be scanned")
    ap.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")
    ap.add_argument("-o", "--output", default="scanned_docs", help="Output directory")
    args = vars(ap.parse_args())

    try:
        # Process document
        print("Processing document...")
        scanned = process_document(args["image"], args["debug"])
        
        # Save result
        output_path = save_document(scanned, args["output"])
        print(f"Document saved successfully to: {output_path}")
        
        # Show result if in debug mode
        if args["debug"]:
            cv2.imshow("Original", imutils.resize(cv2.imread(args["image"]), height=650))
            cv2.imshow("Scanned", imutils.resize(scanned, height=650))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()