from setuptools import setup, find_packages

setup(
    name="TilinScanner",
    version="2.0.0",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.19.0',
        'opencv-python>=4.5.0',
        'imutils>=0.5.4',
        'scikit-image>=0.18.0',
        'PyQt5>=5.15.0',
        'PyQt5-sip>=12.8.0',
        'pillow>=8.0.0',
        'pyyaml>=5.4.1',
        'pdf2image>=1.16.0',
        'pytesseract>=0.3.8',
        'reportlab>=3.6.2',
        'tqdm>=4.65.0',
        'pyinstaller>=5.0.0'
    ],
    author="xuyaxaki",
    description="A document scanner application",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    keywords="document scanner ocr pdf",
)
