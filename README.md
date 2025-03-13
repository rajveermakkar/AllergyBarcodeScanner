# Product Scanner with Allergy Analysis

[![Run on Replit](https://replit.com/badge/github/rajveermakkar/AllergyBarcodeScanner)](https://replit.com/new/github/rajveermakkar/AllergyBarcodeScanner)

A Flask web application that scans product barcodes and analyzes ingredients for potential allergens. The application supports barcode scanning via camera, manual entry, and image upload.

## Features

- Live barcode scanning using device camera
- Manual barcode entry
- Image upload for barcode scanning
- Image upload for ingredients text extraction
- Allergy analysis using Google's Gemini AI
- PDF report generation
- Support for multiple image formats (JPG, PNG, GIF, HEIC)
- One-click deployment to Replit

## Quick Start with Replit

1. Click the "Run on Replit" button above
2. Sign in to your Replit account (or create one if you don't have it)
3. Wait for the project to import
4. Create a `.env` file in the Replit project and add your Gemini API key:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```
5. Click the "Run" button to start the application
6. The application will be available at the URL shown in the Replit console

## Prerequisites

### System Dependencies

#### macOS
```bash
brew install tesseract
brew install zbar
```

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr
sudo apt-get install -y libzbar0
```

#### Windows
1. Install Tesseract OCR from: https://github.com/UB-Mannheim/tesseract/wiki
2. Add Tesseract to your system PATH
3. Install Visual C++ Redistributable if not already installed

### Python Dependencies
```bash
pip install -r requirements.txt
```

## Environment Setup

1. Create a `.env` file in the project root
2. Add your Gemini API key:
```
GEMINI_API_KEY=your_api_key_here
```

## Running the Application

1. Start the Flask server:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5004
```

## Usage

1. **Camera Scanner**: Click "Start Scanner" and position the barcode in front of your camera
2. **Manual Entry**: Enter the barcode number manually
3. **Upload Images**: Upload either a barcode image or an ingredients list image

Enter your allergies (comma-separated) in the provided input field before scanning or uploading.

## Notes

- The application requires a working camera for the barcode scanner feature
- For best results with image uploads, ensure good lighting and clear images
- HEIC image support requires additional system libraries on some platforms
- The application uses the Open Food Facts API for product information
- Allergy analysis is performed using Google's Gemini AI model
- When running on Replit, the application will be accessible via the provided Replit URL

## License

MIT License

Copyright (c) 2024 Jordan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Author

Jordan