# The above Python code is importing the `sys` module. This module provides access to some variables
# used or maintained by the Python interpreter and to functions that interact with the interpreter.
# However, the code snippet you provided is incomplete and does not contain any specific functionality
# or operations.
import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                            QMessageBox, QProgressBar, QStyleFactory, QFrame, QSizePolicy, 
                            QSpacerItem, QLabel, QToolBar, QActionGroup, QAction, QDockWidget, 
                            QListWidget, QMenuBar, QMenu, QGraphicsView, 
                            QGraphicsScene, QGraphicsPixmapItem)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QIcon, QFont, QPalette, QColor
from utils import load_config, cv2_to_qpixmap, resize_image_aspect
from scanner import process_document, save_document
import cv2

class ScannerThread(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, image):
        super().__init__()
        self.image = image

    def run(self):
        try:
            self.progress.emit(10)
            # Enhanced progress feedback
            self.progress.emit(25)
            scanned = process_document(self.image)
            self.progress.emit(60)
            
            # Additional image cleanup if needed
            if scanned is not None and scanned.size > 0:
                self.progress.emit(80)
                # Ensure good contrast
                scanned = cv2.normalize(scanned, None, 0, 255, cv2.NORM_MINMAX)
            else:
                raise ValueError("Failed to process document")
                
            self.progress.emit(90)
            saved_path = save_document(scanned)
            self.progress.emit(100)
            self.finished.emit(scanned)
        except Exception as e:
            self.error.emit(str(e))

class ZoomableImageView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene())
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setBackgroundBrush(QColor("#f0f0f0"))
        self.setFrameShape(QFrame.NoFrame)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self._empty = True
        self.setScene(QGraphicsScene(self))

    def set_image(self, pixmap):
        """Set the image to display"""
        self.scene().clear()
        self.scene().addItem(QGraphicsPixmapItem(pixmap))
        self.setSceneRect(self.scene().itemsBoundingRect())
        self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)
        self._empty = False

    def fitInView(self, *args, **kwargs):
        """Custom fit to handle empty scenes"""
        if not self._empty:
            super().fitInView(*args, **kwargs)

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            factor = 1.25
        else:
            factor = 0.8
        self.scale(factor, factor)
        if not self._empty:
            super().wheelEvent(event)

class DocumentScannerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = load_config()
        self.setup_styling()
        self.init_ui()

        # Add toolbar
        self.create_toolbar()
        
        # Add menu bar
        self.create_menu()
        
        # Add recent files dock
        self.create_recent_files_dock()

        self.recent_files = []
        self.max_recent_files = self.config['gui']['features']['recent_files_count']

    def setup_styling(self):
        # Set application style
        self.app_style = QStyleFactory.create(self.config['gui']['theme'])
        QApplication.setStyle(self.app_style)
        
        # Create and set palette
        palette = QPalette()
        colors = self.config['gui']['colors']
        palette.setColor(QPalette.Window, QColor(colors['background']))
        palette.setColor(QPalette.WindowText, QColor(colors['text']))
        palette.setColor(QPalette.Button, QColor(colors['primary']))
        palette.setColor(QPalette.ButtonText, QColor(colors['background']))
        palette.setColor(QPalette.Highlight, QColor(colors['accent']))
        QApplication.setPalette(palette)

        # Set default font
        font = QFont(self.config['gui']['fonts']['main'], 
                    self.config['gui']['fonts']['size'])
        QApplication.setFont(font)

    def init_ui(self):
        self.setWindowTitle(f"{self.config['app']['name']} v{self.config['app']['version']}")
        self.setGeometry(100, 100, self.config['gui']['window_width'], 
                        self.config['gui']['window_height'])

        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)

        # Add header
        header = QLabel(self.config['app']['name'])
        header.setFont(QFont(self.config['gui']['fonts']['main'], 24, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)

        # Create preview container
        preview_container = QFrame()
        preview_container.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        preview_container.setStyleSheet(f"""
            QFrame {{
                background-color: {self.config['gui']['colors']['secondary']};
                border-radius: 10px;
                padding: 10px;
            }}
        """)
        preview_layout = QHBoxLayout(preview_container)
        preview_layout.setSpacing(15)

        # Style preview labels
        preview_style = f"""
            QLabel {{
                background-color: {self.config['gui']['colors']['background']};
                border-radius: 5px;
                padding: 10px;
                color: {self.config['gui']['colors']['text']};
            }}
        """

        # Original image preview
        self.original_preview = ZoomableImageView()
        self.original_preview.setStyleSheet(preview_style)
        self.original_preview.setAlignment(Qt.AlignCenter)
        self.original_preview.setMinimumSize(400, 400)
        preview_layout.addWidget(self.original_preview)

        # Scanned image preview
        self.scanned_preview = ZoomableImageView()
        self.scanned_preview.setStyleSheet(preview_style)
        self.scanned_preview.setAlignment(Qt.AlignCenter)
        self.scanned_preview.setMinimumSize(400, 400)
        preview_layout.addWidget(self.scanned_preview)

        layout.addWidget(preview_container)

        # Create controls
        controls = QFrame()
        controls.setStyleSheet(f"""
            QFrame {{
                background-color: {self.config['gui']['colors']['secondary']};
                border-radius: 10px;
                padding: 10px;
            }}
        """)
        controls_layout = QHBoxLayout(controls)
        controls_layout.setSpacing(10)

        # Button style
        button_style = f"""
            QPushButton {{
                background-color: {self.config['gui']['colors']['primary']};
                color: {self.config['gui']['colors']['background']};
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {self.config['gui']['colors']['accent']};
            }}
            QPushButton:disabled {{
                background-color: {self.config['gui']['colors']['secondary']};
                color: #7f8c8d;
            }}
        """

        self.select_btn = QPushButton("Select Image")
        self.select_btn.setStyleSheet(button_style)
        self.select_btn.clicked.connect(self.select_image)
        controls_layout.addWidget(self.select_btn)

        self.scan_btn = QPushButton("Scan Document")
        self.scan_btn.setStyleSheet(button_style)
        self.scan_btn.clicked.connect(self.scan_document)
        self.scan_btn.setEnabled(False)
        controls_layout.addWidget(self.scan_btn)

        self.save_btn = QPushButton("Save Result")
        self.save_btn.setStyleSheet(button_style)
        self.save_btn.clicked.connect(self.save_result)
        self.save_btn.setEnabled(False)
        controls_layout.addWidget(self.save_btn)

        layout.addWidget(controls)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                border: none;
                border-radius: 5px;
                text-align: center;
                background-color: {self.config['gui']['colors']['background']};
            }}
            QProgressBar::chunk {{
                background-color: {self.config['gui']['colors']['accent']};
                border-radius: 5px;
            }}
        """)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Add status bar
        self.statusBar().showMessage(f"{self.config['app']['company']} | Ready")

        self.current_image = None
        self.scanned_image = None
        self.show()

    def create_toolbar(self):
        toolbar = QToolBar()
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(32, 32))
        
        # Add toolbar actions
        scan_action = QAction(QIcon("icons/scan.png"), "Scan", self)
        scan_action.triggered.connect(self.scan_document)
        toolbar.addAction(scan_action)
        
        # Add more toolbar actions...
        
        self.addToolBar(toolbar)

    def create_menu(self):
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        open_action = QAction("Open", self)
        open_action.triggered.connect(self.select_image)
        file_menu.addAction(open_action)
        
        # Add more menus and actions...

    def create_recent_files_dock(self):
        dock = QDockWidget("Recent Files", self)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        
        self.recent_files_list = QListWidget()
        self.recent_files_list.itemClicked.connect(self.load_recent_file)
        
        dock.setWidget(self.recent_files_list)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)

    def load_recent_file(self, item):
        """Load a file from the recent files list"""
        file_path = item.text()
        if os.path.exists(file_path):
            try:
                self.current_image = cv2.imread(file_path)
                if self.current_image is None:
                    raise ValueError("Could not load image")
                
                preview = resize_image_aspect(
                    self.current_image,
                    self.width() // 2 - 30,
                    self.height() - 100
                )
                self.original_preview.set_image(cv2_to_qpixmap(preview))  # Changed from setPixmap
                self.scan_btn.setEnabled(True)
                self.save_btn.setEnabled(False)
                self.add_to_recent_files(file_path)
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not load image: {str(e)}")
                self.remove_from_recent_files(file_path)
        else:
            QMessageBox.warning(self, "Warning", "File no longer exists")
            self.remove_from_recent_files(file_path)

    def add_to_recent_files(self, file_path):
        """Add a file to the recent files list"""
        if file_path in self.recent_files:
            self.recent_files.remove(file_path)
        self.recent_files.insert(0, file_path)
        
        # Limit the number of recent files
        if len(self.recent_files) > self.max_recent_files:
            self.recent_files = self.recent_files[:self.max_recent_files]
            
        self.update_recent_files_list()

    def remove_from_recent_files(self, file_path):
        """Remove a file from the recent files list"""
        if file_path in self.recent_files:
            self.recent_files.remove(file_path)
        self.update_recent_files_list()

    def update_recent_files_list(self):
        """Update the recent files list widget"""
        self.recent_files_list.clear()
        for file_path in self.recent_files:
            self.recent_files_list.addItem(file_path)

    def select_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if file_name:
            try:
                self.current_image = cv2.imread(file_name)
                if self.current_image is None:
                    raise ValueError("Could not load image")
                
                preview = resize_image_aspect(
                    self.current_image,
                    self.width() // 2 - 30,
                    self.height() - 100
                )
                self.original_preview.set_image(cv2_to_qpixmap(preview))  # Changed from setPixmap
                self.scan_btn.setEnabled(True)
                self.save_btn.setEnabled(False)
                self.add_to_recent_files(file_name)
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not load image: {str(e)}")

    def scan_document(self):
        if self.current_image is None:
            QMessageBox.warning(self, "Warning", "Please select an image first")
            return

        # Check image size and quality
        if self.current_image.size == 0:
            QMessageBox.warning(self, "Warning", "Invalid image")
            return

        height, width = self.current_image.shape[:2]
        if height < 100 or width < 100:
            QMessageBox.warning(self, "Warning", "Image too small")
            return

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.scan_btn.setEnabled(False)
        self.select_btn.setEnabled(False)

        self.scanner_thread = ScannerThread(self.current_image)
        self.scanner_thread.finished.connect(self.on_scan_complete)
        self.scanner_thread.error.connect(self.on_scan_error)
        self.scanner_thread.progress.connect(self.progress_bar.setValue)
        self.scanner_thread.start()

    def on_scan_complete(self, scanned_image):
        self.scanned_image = scanned_image
        preview = resize_image_aspect(
            scanned_image,
            self.width() // 2 - 30,
            self.height() - 100
        )
        self.scanned_preview.set_image(cv2_to_qpixmap(preview))  # Changed from setPixmap
        self.save_btn.setEnabled(True)
        self.scan_btn.setEnabled(True)
        self.select_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

    def on_scan_error(self, error_message):
        QMessageBox.critical(self, "Error", f"Scanning failed: {error_message}")
        self.scan_btn.setEnabled(True)
        self.select_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

    def save_result(self):
        if self.scanned_image is None:
            return

        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Save Scanned Document",
            "",
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;All Files (*.*)"
        )

        if file_name:
            try:
                cv2.imwrite(file_name, self.scanned_image)
                QMessageBox.information(self, "Success", "Document saved successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not save document: {str(e)}")

def main():
    app = QApplication(sys.argv)
    app.setStyle(load_config()['gui']['theme'])
    ex = DocumentScannerGUI()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

