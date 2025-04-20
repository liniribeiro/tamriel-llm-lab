# Update package lists
sudo apt update

# Install tesseract-ocr and libtesseract-dev
sudo apt install tesseract-ocr libtesseract-dev

# Install more dependencies
sudo apt-get install \
    libleptonica-dev \
    tesseract-ocr-dev \
    python3-pil \
    tesseract-ocr-eng

# Install the python depedencies

pip install pytesseract