
import yaml
import os
from PyQt5.QtGui import QImage, QPixmap
import cv2
import numpy as np
import imutils

def load_config():
    """
    The above functions include loading a configuration file, converting OpenCV images to QPixmap,
    resizing images while maintaining aspect ratio, creating a PDF from images, extracting text from an
    image using Tesseract OCR, and automatically rotating an image based on its content.
    :return: The code provided contains several functions for image processing and PDF generation using
    OpenCV, ReportLab, and other libraries. The functions include loading a configuration file,
    converting OpenCV images to QPixmap, resizing images while maintaining aspect ratio, creating a PDF
    from a list of images, extracting text from an image using Tesseract OCR, and auto-rotating an image
    based on its content.
    """
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def cv2_to_qpixmap(cv_img):
    height, width = cv_img.shape[:2]
    if len(cv_img.shape) == 2:  # Grayscale
        bytes_per_line = width
        q_img = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
    else:  # Color
        bytes_per_line = 3 * width
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        q_img = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
    
    return QPixmap.fromImage(q_img)

def resize_image_aspect(image, target_width, target_height):
    height, width = image.shape[:2]
    aspect = width / height

    if width > height:
        new_width = target_width
        new_height = int(target_width / aspect)
    else:
        new_height = target_height
        new_width = int(target_height * aspect)

    return cv2.resize(image, (new_width, new_height))

def create_pdf(images, output_path, metadata=None):
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    import io
    from PIL import Image
    from reportlab.lib.utils import ImageReader

    c = canvas.Canvas(output_path, pagesize=letter)
    if metadata:
        c.setAuthor(metadata.get('author', ''))
        c.setTitle(metadata.get('title', ''))
        c.setSubject(metadata.get('subject', ''))

    for img in images:
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img_width, img_height = img_pil.size
        aspect = img_height / float(img_width)
        
        # Scale to fit on the page
        if aspect > 1:
            width = letter[0] - 40
            height = width * aspect
        else:
            height = letter[1] - 40
            width = height / aspect

        c.drawImage(ImageReader(img_pil), 20, letter[1] - height - 20, 
                   width=width, height=height)
        c.showPage()
    
    c.save()

def extract_text(image):
    import pytesseract
    try:
        return pytesseract.image_to_string(image)
    except Exception as e:
        print(f"OCR error: {e}")
        return ""

def auto_rotate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    coords = np.column_stack(np.where(gray > 0))
    angle = cv2.minAreaRect(coords)[-1]
    
    if angle < -45:
        angle = 90 + angle
    
    return imutils.rotate_bound(image, -angle)
