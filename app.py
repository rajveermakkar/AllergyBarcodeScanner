import os
import requests
from flask import Flask, render_template, request, send_file, jsonify
import json
from fpdf import FPDF
import tempfile
import logging
from urllib.parse import urlparse
import os.path
import re
import google.generativeai as genai
from dotenv import load_dotenv
import pytesseract
from PIL import Image, ImageEnhance
import io
import base64
from pyzbar.pyzbar import decode
import traceback
import cv2
import numpy as np
import pillow_heif

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Open Food Facts API URL with English language preference
API_URL = "https://world.openfoodfacts.org/api/v0/product"

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found in environment variables. Please check your .env file."
    )

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.0-pro-exp-02-05')


def check_allergies(ingredients, allergies):
    try:
        if not ingredients or ingredients.lower() == 'not available':
            return "INGREDIENTS NOT AVAILABLE: Unable to perform safety analysis as ingredients information is not available."

        prompt = f"""Analyze these food ingredients for someone with the following allergies/conditions: {allergies}

Ingredients: {ingredients}

First, if the ingredients are not in English, translate them to English.
Then, provide a detailed analysis in this format:

SAFETY RATING: [1-10]
1-3: Extremely Dangerous (RED) - Do not consume
4-5: High Risk (ORANGE) - Avoid unless necessary, consult healthcare provider
6-7: Moderate Risk (YELLOW) - Use with caution, limit consumption
8-9: Safe (LIGHT GREEN) - Can be consumed occasionally
10: Very Safe (GREEN) - Can be consumed regularly

SAFETY STATUS: [SAFE/UNSAFE/CAUTION]
[Color-coded status based on rating]

Explanation of rating:
- What makes this product safe/unsafe
- How frequently/in what quantity it might be safe to consume (if applicable)
- Any specific risks or concerns

ANALYSIS:
1. SAFE INGREDIENTS: List ingredients that are definitely safe
2. UNSAFE INGREDIENTS: List ingredients that are definitely problematic
3. UNCERTAIN INGREDIENTS: List any ingredients where safety cannot be determined with certainty
4. CROSS-CONTAMINATION RISKS: List any potential risks

IMPORTANT:
- If you're uncertain about any ingredient's safety, explicitly state "Safety of [ingredient] cannot be determined"
- DO NOT make assumptions about ingredient safety
- If ingredients are in a different language, provide both original and translated versions

CONCLUSION:
[SAFE/UNSAFE/CAUTION] - Brief explanation why

Remember, someone's health depends on this analysis. Be thorough and explicit about any uncertainties."""

        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Error checking allergies with Gemini API: {str(e)}")
        return "Error analyzing ingredients for allergies. Please consult with a healthcare professional."


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/scan_barcode', methods=['POST'])
def scan_barcode():
    try:
        data = request.json
        barcode = data.get('barcode')
        allergies = data.get('allergies', '')
        logger.debug(f"Received barcode: {barcode}")

        if not barcode:
            return jsonify({'error': 'No barcode provided'}), 400

        # Call the Open Food Facts API with English language preference
        logger.debug(f"Making API request to {API_URL}/{barcode}.json")
        response = requests.get(f"{API_URL}/{barcode}.json",
                                params={'lc': 'en'})

        logger.debug(f"API Response Status: {response.status_code}")
        logger.debug(f"API Response: {response.text}")

        if response.status_code != 200:
            error_msg = f'API error: {response.status_code} - {response.text}'
            logger.error(error_msg)
            return jsonify({'error': error_msg}), 500

        product_data = response.json()

        if product_data.get('status') == 0:
            error_msg = 'No product found for this barcode'
            logger.warning(error_msg)
            return jsonify({'error': error_msg}), 404

        # Get product information
        product = product_data.get('product', {})
        ingredients = product.get('ingredients_text', 'Not available')
        if ingredients:
            ingredients = ingredients.replace('_', ' ').replace('en:', '')

        # Check allergies if allergies are provided
        allergy_analysis = None
        if allergies:
            allergy_analysis = check_allergies(ingredients, allergies)

        # Create a temporary file for the PDF
        temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        temp_pdf.close()

        # Generate PDF with the product information
        generate_pdf(product_data, temp_pdf.name, allergy_analysis)

        return jsonify({
            'success': True,
            'message': 'Product found',
            'product': product_data,
            'allergy_analysis': allergy_analysis,
            'pdf_url': f'/download_pdf?barcode={barcode}'
        })

    except requests.exceptions.RequestException as e:
        error_msg = f'Network error while calling API: {str(e)}'
        logger.error(error_msg, exc_info=True)
        return jsonify({'error': error_msg}), 500
    except Exception as e:
        error_msg = f'Error processing barcode: {str(e)}'
        logger.error(error_msg, exc_info=True)
        return jsonify({'error': error_msg}), 500


@app.route('/download_pdf')
def download_pdf():
    try:
        barcode = request.args.get('barcode')
        is_custom = request.args.get('custom', 'false').lower() == 'true'

        if is_custom:
            # Handle custom analysis PDF download
            temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
            temp_pdf.close()

            return send_file(temp_pdf.name,
                             as_attachment=True,
                             download_name="ingredients_analysis.pdf",
                             mimetype='application/pdf')
        else:
            # Handle barcode product PDF download
            if not barcode:
                return "No barcode provided", 400

            response = requests.get(f"{API_URL}/{barcode}.json",
                                    params={'lc': 'en'})

            if response.status_code != 200:
                return f"API error: {response.status_code} - {response.text}", 500

            product_data = response.json()

            if product_data.get('status') == 0:
                return "No product found for this barcode", 404

            temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
            temp_pdf.close()

            generate_pdf(product_data, temp_pdf.name)

            return send_file(temp_pdf.name,
                             as_attachment=True,
                             download_name=f"product_{barcode}.pdf",
                             mimetype='application/pdf')

    except Exception as e:
        logger.error(f"Error in download_pdf: {str(e)}")
        return f"Error generating PDF: {str(e)}", 500


def download_image(url, output_path):
    try:
        # Add a timeout to the request
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()

        # Check if the response is actually an image
        content_type = response.headers.get('content-type', '')
        if not content_type.startswith('image/'):
            logger.warning(
                f"URL {url} is not an image (content-type: {content_type})")
            return False

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        # Verify the file was created and has content
        if not os.path.exists(output_path) or os.path.getsize(
                output_path) == 0:
            logger.warning(
                f"Downloaded image file is empty or was not created: {output_path}"
            )
            return False

        return True
    except requests.exceptions.Timeout:
        logger.error(f"Timeout while downloading image from {url}")
        return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading image from {url}: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error downloading image: {str(e)}")
        return False


def generate_pdf(product_data, output_path, allergy_analysis=None):
    try:
        pdf = FPDF()
        pdf.add_page()

        # Get product information
        product = product_data.get('product', {})

        # Title first
        pdf.set_font("Arial", "B", 16)
        title = product.get('product_name', 'Unknown Product')
        if not title:
            title = product.get('generic_name', 'Unknown Product')
        pdf.cell(0, 10, title, 0, 1, 'C')
        pdf.ln(5)

        # Download and add product image if available
        image_url = product.get('image_url')
        if image_url:
            try:
                # Create a temporary file for the image
                temp_img = tempfile.NamedTemporaryFile(delete=False,
                                                       suffix='.jpg')
                temp_img.close()

                # Download the image
                if download_image(image_url, temp_img.name):
                    # Add image to PDF with proper sizing and positioning
                    pdf.image(temp_img.name, x=80, y=25, w=50, h=50)
                    # Clean up temporary image file
                    os.unlink(temp_img.name)
            except Exception as e:
                logger.error(f"Error adding image to PDF: {str(e)}")

        # Move to position after image for content
        pdf.set_xy(10, 85)  # Start content below the image

        # Ingredients Section
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Ingredients:", 0, 1)

        # Get ingredients
        ingredients = product.get('ingredients_text', 'Not available')
        if ingredients:
            ingredients = ingredients.replace('_', ' ').replace('en:', '')
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 10, ingredients)

        # Nutrition Facts Section
        pdf.ln(5)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Nutrition Facts:", 0, 1)

        # Get nutrition facts
        nutriments = product.get('nutriments', {})
        if nutriments:
            pdf.set_font("Arial", "", 12)
            for key, value in nutriments.items():
                if isinstance(value, (int, float)):
                    unit = nutriments.get(f'{key}_unit', '')
                    pdf.multi_cell(
                        0, 10,
                        f"{key.replace('_', ' ').title()}: {value}{unit}")

        # Allergy Analysis Section (if available)
        if allergy_analysis:
            pdf.ln(5)
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, "Allergy Analysis:", 0, 1)
            pdf.set_font("Arial", "", 12)

            # Replace star symbols with text
            allergy_analysis = allergy_analysis.replace('★', '*').replace('☆', '*')

            # Extract safety status and conclusion
            safety_status_match = re.search(
                r'SAFETY STATUS:\s*(SAFE|UNSAFE|WARNING)', allergy_analysis,
                re.IGNORECASE)
            conclusion_match = re.search(
                r'CONCLUSION:\s*(SAFE|UNSAFE|WARNING)\s*-\s*(.*)',
                allergy_analysis, re.IGNORECASE)

            if safety_status_match:
                status = safety_status_match.group(1).upper()
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 10, f"Safety Status: {status}", 0, 1)
                pdf.set_font("Arial", "", 12)

            # Add the analysis text
            pdf.multi_cell(0, 10, allergy_analysis)

            if conclusion_match:
                status = conclusion_match.group(1).upper()
                explanation = conclusion_match.group(2)
                pdf.ln(5)
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 10, f"Final Decision: {status}", 0, 1)
                pdf.set_font("Arial", "", 12)
                pdf.multi_cell(0, 10, explanation)

        # Save the PDF
        pdf.output(output_path)
        logger.debug(f"PDF generated successfully at {output_path}")

    except Exception as e:
        logger.error(f"Error generating PDF: {str(e)}", exc_info=True)
        raise


# Create templates directory if it doesn't exist
if not os.path.exists('templates'):
    os.makedirs('templates')

# Create the HTML template file
with open('templates/index.html', 'w') as f:
    f.write('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Barcode Scanner</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/quagga@0.12.1/dist/quagga.min.js"></script>
    <style>
        body {
            background-color: #f8f9fa;
        }
        .navbar {
            background-color: #2c3e50;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .navbar-brand {
            color: white !important;
            font-weight: bold;
        }
        #interactive.viewport {
            position: relative;
            width: 100%;
            height: 400px;  /* Increased height */
            overflow: hidden;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            background-color: #000;  /* Dark background */
        }
        #interactive.viewport video {
            width: 100%;
            height: 100%;
            object-fit: cover;  /* Ensure video fills container */
            position: absolute;
            top: 0;
            left: 0;
            border-radius: 10px;
        }
        #interactive.viewport canvas {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 1;
        }
        .scanning-feedback {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 280px;
            height: 180px;
            border: 2px solid #3498db;
            border-radius: 10px;
            z-index: 2;
            box-shadow: 0 0 0 9999px rgba(0, 0, 0, 0.5);
        }
        .scanning-line {
            position: absolute;
            width: 100%;
            height: 2px;
            background-color: #3498db;
            animation: scan 2s linear infinite;
        }
        @keyframes scan {
            0% { top: 0; }
            100% { top: 100%; }
        }
        .loading-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.7);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 9999;
            color: white;
            font-size: 1.5rem;
            display: none;
        }
        .product-info {
            margin-top: 20px;
            display: none;
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .card {
            margin-bottom: 20px;
            border: none;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .card-header {
            background-color: #2c3e50;
            color: white;
            border-radius: 10px 10px 0 0 !important;
        }
        .allergy-analysis {
            margin-top: 20px;
            padding: 15px;
            border-radius: 10px;
            background-color: #f8f9fa;
        }
        .product-image {
            max-width: 200px;
            max-height: 200px;
            object-fit: contain;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .btn-primary {
            background-color: #3498db;
            border-color: #3498db;
        }
        .btn-primary:hover {
            background-color: #2980b9;
            border-color: #2980b9;
        }
        .btn-success {
            background-color: #2ecc71;
            border-color: #2ecc71;
        }
        .btn-success:hover {
            background-color: #27ae60;
            border-color: #27ae60;
        }
        .nav-tabs .nav-link {
            color: #2c3e50;
            border: none;
            padding: 10px 20px;
            margin-right: 5px;
            border-radius: 5px 5px 0 0;
            background-color: #f8f9fa;
        }
        .nav-tabs .nav-link.active {
            background-color: #3498db;
            color: white;
        }
        .nav-tabs .nav-link:not(.active):hover {
            background-color: #e9ecef;
        }
        .form-control {
            border-radius: 5px;
            border: 1px solid #ddd;
        }
        .form-control:focus {
            border-color: #3498db;
            box-shadow: 0 0 0 0.2rem rgba(52, 152, 219, 0.25);
        }
        .alert {
            border-radius: 10px;
        }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark mb-4">
        <div class="container">
            <a class="navbar-brand" href="#">
                <i class="fas fa-barcode me-2"></i>
                Product Scanner
            </a>
        </div>
    </nav>

    <div class="loading-overlay" id="loadingOverlay">
        <div class="spinner-border text-light" role="status">
            <span class="visually-hidden">Loading...</span>
        </div>
        <span class="ms-2">Processing...</span>
    </div>

    <div class="container">
        <div class="row">
            <div class="col-md-8 mx-auto">
                <div class="card">
                    <div class="card-header">
                        <ul class="nav nav-tabs card-header-tabs" id="scannerTabs" role="tablist">
                            <li class="nav-item" role="presentation">
                                <button class="nav-link active" id="camera-tab" data-bs-toggle="tab" data-bs-target="#camera" type="button" role="tab" aria-controls="camera" aria-selected="true">
                                    <i class="fas fa-camera me-2"></i>Camera Scanner
                                </button>
                            </li>
                            <li class="nav-item" role="presentation">
                                <button class="nav-link" id="manual-tab" data-bs-toggle="tab" data-bs-target="#manual" type="button" role="tab" aria-controls="manual" aria-selected="false">
                                    <i class="fas fa-keyboard me-2"></i>Manual Entry
                                </button>
                            </li>
                            <li class="nav-item" role="presentation">
                                <button class="nav-link" id="upload-tab" data-bs-toggle="tab" data-bs-target="#upload" type="button">
                                    <i class="fas fa-upload me-2"></i>Upload Images
                                </button>
                            </li>
                        </ul>
                    </div>
                    <div class="card-body">
                        <div class="tab-content" id="scannerTabsContent">
                            <div class="tab-pane fade show active" id="camera" role="tabpanel" aria-labelledby="camera-tab">
                                <div class="mb-3">
                                    <label for="cameraAllergiesInput" class="form-label">Your Allergies (comma-separated)</label>
                                    <input type="text" class="form-control" id="cameraAllergiesInput" placeholder="e.g., peanuts, dairy, shellfish">
                                </div>
                                <div class="text-center mb-3">
                                    <button class="btn btn-primary" id="startButton">
                                        <i class="fas fa-play me-2"></i>Start Scanner
                                    </button>
                                    <button class="btn btn-danger d-none" id="stopButton">
                                        <i class="fas fa-stop me-2"></i>Stop Scanner
                                    </button>
                                </div>
                                <div id="interactive" class="viewport"></div>
                                <div class="alert alert-info mt-3">
                                    <i class="fas fa-info-circle me-2"></i>
                                    Position the barcode in front of the camera to scan it.
                                </div>
                                <div class="alert alert-warning camera-permission-alert" id="cameraPermissionAlert">
                                    <i class="fas fa-exclamation-triangle me-2"></i>
                                    <strong>Camera Permission Required!</strong><br>
                                    Please follow these steps to enable camera access:<br>
                                    1. Click the camera icon in your browser's address bar<br>
                                    2. Select "Allow" for camera access<br>
                                    3. Refresh the page and try again
                                </div>
                                <div class="camera-instructions">
                                    <strong>Mobile Device Instructions:</strong><br>
                                    1. Make sure you're using a modern browser (Chrome, Safari, or Firefox)<br>
                                    2. When prompted, allow camera access<br>
                                    3. If no prompt appears, look for a camera icon in your browser's address bar<br>
                                    4. If still having issues, try using the Manual Entry tab instead
                                </div>
                            </div>
                            <div class="tab-pane fade" id="manual" role="tabpanel" aria-labelledby="manual-tab">
                                <div class="mb-3">
                                    <label for="allergiesInput" class="form-label">Your Allergies (comma-separated)</label>
                                    <input type="text" class="form-control" id="allergiesInput" placeholder="e.g., peanuts, dairy, shellfish">
                                </div>
                                <div class="mb-3">
                                    <label for="barcodeInput" class="form-label">Enter Barcode Number</label>
                                    <input type="text" class="form-control" id="barcodeInput" placeholder="e.g., 0123456789012">
                                </div>
                                <button class="btn btn-primary" id="manualSubmit">
                                    <i class="fas fa-search me-2"></i>Submit
                                </button>
                            </div>
                            <div class="tab-pane fade" id="upload" role="tabpanel" aria-labelledby="upload-tab">
                                <div class="mb-3">
                                    <label for="uploadAllergiesInput" class="form-label">Your Allergies (comma-separated)</label>
                                    <input type="text" class="form-control" id="uploadAllergiesInput" placeholder="e.g., peanuts, dairy, shellfish">
                                </div>

                                <div class="mb-4">
                                    <h5>Upload Barcode Image</h5>
                                    <div class="input-group">
                                        <input type="file" class="form-control" id="barcodeImageInput" accept="image/jpeg,image/png,image/gif,image/heic">
                                        <button class="btn btn-primary" id="uploadBarcodeBtn">
                                            <i class="fas fa-barcode me-2"></i>Process Barcode
                                        </button>
                                    </div>
                                    <small class="text-muted">Supported formats: JPG, PNG, GIF, HEIC</small>
                                </div>

                                <div class="mb-4">
                                    <h5>Upload Ingredients Image</h5>
                                    <div class="input-group">
                                        <input type="file" class="form-control" id="ingredientsImageInput" accept="image/jpeg,image/png,image/gif,image/heic">
                                        <button class="btn btn-primary" id="uploadIngredientsBtn">
                                            <i class="fas fa-list me-2"></i>Process Ingredients
                                        </button>
                                    </div>
                                    <small class="text-muted">Supported formats: JPG, PNG, GIF, HEIC</small>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="card product-info" id="productInfo">
                    <div class="card-header">
                        <h5 class="card-title mb-0" id="productTitle">Product Information</h5>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-4 text-center mb-3">
                                <img id="productImage" class="product-image" src="" alt="Product Image" style="display: none;">
                            </div>
                            <div class="col-md-8">
                                <div id="productDetails">
                                    <!-- Product details will be displayed here -->
                                </div>
                            </div>
                        </div>
                        <div id="allergyAnalysis" class="allergy-analysis" style="display: none;">
                            <!-- Allergy analysis will be displayed here -->
                        </div>
                        <div class="text-center mt-3">
                            <a href="#" class="btn btn-success" id="downloadButton">
                                <i class="fas fa-download me-2"></i>Download PDF
                            </a>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const startButton = document.getElementById('startButton');
            const stopButton = document.getElementById('stopButton');
            const manualSubmit = document.getElementById('manualSubmit');
            const barcodeInput = document.getElementById('barcodeInput');
            const allergiesInput = document.getElementById('allergiesInput');
            const productInfo = document.getElementById('productInfo');
            const productTitle = document.getElementById('productTitle');
            const productDetails = document.getElementById('productDetails');
            const productImage = document.getElementById('productImage');
            const allergyAnalysis = document.getElementById('allergyAnalysis');
            const downloadButton = document.getElementById('downloadButton');
            const loadingOverlay = document.getElementById('loadingOverlay');
            const cameraPermissionAlert = document.getElementById('cameraPermissionAlert');

            let downloadUrl = '';

            // Scanner configuration
            function startScanner() {
                const viewport = document.getElementById('interactive');
                viewport.innerHTML = '';

                // Create video element
                const video = document.createElement('video');
                video.setAttribute('playsinline', true);
                video.setAttribute('autoplay', true);
                video.style.width = '100%';
                video.style.height = '100%';
                viewport.appendChild(video);

                // Add scanning feedback overlay
                const feedback = document.createElement('div');
                feedback.className = 'scanning-feedback';
                const scanLine = document.createElement('div');
                scanLine.className = 'scanning-line';
                feedback.appendChild(scanLine);
                viewport.appendChild(feedback);

                let isProcessing = false;
                let lastDetectedCode = null;
                let lastDetectionTime = 0;
                const detectionThreshold = 50; // Reduced to 50ms for faster response

                // Initialize Quagga
                function initQuagga(stream) {
                    Quagga.init({
                        inputStream: {
                            name: "Live",
                            type: "LiveStream",
                            target: video,
                            constraints: {
                                width: 1280,
                                height: 720,
                                facingMode: "environment",
                                aspectRatio: 1.777778,
                                frameRate: 60
                            },
                            area: { // Tighter scan area for faster processing
                                top: "35%",
                                right: "15%",
                                left: "15%",
                                bottom: "35%",
                            },
                            singleChannel: true // Faster processing
                        },
                        decoder: {
                            readers: ["ean_reader", "ean_8_reader", "upc_reader", "upc_e_reader"],
                            multiple: false,
                            debug: false,
                            locate: true
                        },
                        locate: true,
                        frequency: 60, // Process every frame
                        numOfWorkers: 8, // Use more workers
                        locator: {
                            patchSize: "medium",
                            halfSample: false
                        }
                    }, function(err) {
                        if (err) {
                            console.error('Quagga initialization failed:', err);
                            return;
                        }
                        console.log('Quagga initialization succeeded');
                        Quagga.start();
                    });

                    // Handle successful scans
                    Quagga.onDetected(function(result) {
                        const currentTime = Date.now();
                        const code = result.codeResult.code;

                        // Only check time threshold for same code
                        if (isProcessing ||
                            (code === lastDetectedCode &&
                             currentTime - lastDetectionTime < detectionThreshold)) {
                            return;
                        }

                        lastDetectedCode = code;
                        lastDetectionTime = currentTime;

                        // Immediately stop scanning and show processing state
                        isProcessing = true;
                        Quagga.stop();
                        stopScanner();

                        // Show processing overlay
                        loadingOverlay.style.display = 'flex';
                        loadingOverlay.innerHTML = `
                            <div class="text-center">
                                <div class="spinner-border text-light mb-3" role="status">
                                    <span class="visually-hidden">Loading...</span>
                                </div>
                                <div class="text-light">
                                    <h4>Barcode Detected!</h4>
                                    <p>Processing product information...</p>
                                </div>
                            </div>
                        `;

                        console.log('Barcode detected:', code);

                        // Make API call
                        fetch('/scan_barcode', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify({
                                barcode: code,
                                allergies: document.getElementById('cameraAllergiesInput').value.trim()
                            })
                        })
                        .then(response => response.json())
                        .then(data => {
                            if (data.success) {
                                handleBarcodeSuccess(data);
                            } else {
                                loadingOverlay.style.display = 'none';
                                alert('Error: ' + (data.error || 'Failed to process barcode'));
                            }
                        })
                        .catch(error => {
                            console.error('Error with API call:', error);
                            loadingOverlay.style.display = 'none';
                            alert('Error processing barcode. Please try again.');
                        });
                    });

                    // Minimal overlay drawing for performance
                    Quagga.onProcessed(function(result) {
                        if (result && result.codeResult && result.codeResult.code) {
                            const drawingCtx = Quagga.canvas.ctx.overlay;
                            const drawingCanvas = Quagga.canvas.dom.overlay;
                            drawingCtx.clearRect(0, 0, parseInt(drawingCanvas.getAttribute("width")), parseInt(drawingCanvas.getAttribute("height")));
                            if (result.box) {
                                Quagga.ImageDebug.drawPath(result.box, { x: 0, y: 1 }, drawingCtx, { color: "#00F", lineWidth: 2 });
                            }
                        }
                    });
                }

                // Try to start the camera
                navigator.mediaDevices.getUserMedia({
                    video: {
                        facingMode: { ideal: 'environment' },
                        width: { ideal: 1280 },
                        height: { ideal: 720 }
                    }
                })
                .then(function(stream) {
                    video.srcObject = stream;
                    return video.play();
                })
                .then(() => {
                    initQuagga(video.srcObject);
                    startButton.classList.add('d-none');
                    stopButton.classList.remove('d-none');
                    cameraPermissionAlert.style.display = 'none';
                })
                .catch(function(err) {
                    console.error('Camera error:', err);
                    displayCameraError(err);
                });
            }

            // Improve stop scanner function
            function stopScanner() {
                Quagga.stop();
                const video = document.querySelector('#interactive video');
                if (video && video.srcObject) {
                    const tracks = video.srcObject.getTracks();
                    tracks.forEach(track => track.stop());
                    video.srcObject = null;
                }
                const viewport = document.getElementById('interactive');
                viewport.innerHTML = '';
                startButton.classList.remove('d-none');
                stopButton.classList.add('d-none');
            }

            function displayCameraError(err) {
                console.error('Camera error:', err);
                let errorMessage = 'Camera access error: ';

                if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
                    errorMessage = 'Please allow camera access in your browser settings.';
                } else if (err.name === 'NotFoundError') {
                    errorMessage = 'No camera found. Please check your camera connection.';
                } else if (err.name === 'NotReadableError') {
                    errorMessage = 'Cannot access camera. Please make sure no other app is using it.';
                } else {
                    errorMessage = 'Error accessing camera. Please try again.';
                }

                cameraPermissionAlert.style.display = 'block';
                cameraPermissionAlert.innerHTML = `
                    <i class="fas fa-exclamation-triangle me-2"></i>
                    <strong>Camera Access Error</strong><br>
                    ${errorMessage}<br><br>
                    <strong>Tips:</strong><br>
                    1. Make sure your camera is working<br>
                    2. Allow camera access if prompted<br>
                    3. Try refreshing the page<br>
                    4. You can also use the Manual Entry or Upload Images options
                `;
            }

            function handleBarcodeSuccess(data) {
                loadingOverlay.style.display = 'none';

                // Show product info section
                productInfo.style.display = 'block';

                // Store the download URL
                downloadUrl = data.pdf_url;
                downloadButton.href = downloadUrl;

                // Display product details
                const product = data.product.product;
                productTitle.textContent = product.product_name || 'Product Information';

                // Display product image if available
                if (product.image_url) {
                    productImage.src = product.image_url;
                    productImage.style.display = 'block';
                } else {
                    productImage.style.display = 'none';
                }

                let detailsHTML = '<p><strong>Ingredients:</strong> ' + (product.ingredients_text || 'Not available') + '</p>';

                // Display allergy analysis if available
                if (data.allergy_analysis) {
                    displayAllergyAnalysis(data.allergy_analysis);
                } else {
                    allergyAnalysis.style.display = 'none';
                }

                productDetails.innerHTML = detailsHTML;

                // Scroll to the product info
                productInfo.scrollIntoView({ behavior: 'smooth' });
            }

            // Start scanner button click handler
            startButton.addEventListener('click', startScanner);

            // Stop scanner button click handler
            stopButton.addEventListener('click', stopScanner);

            // Manual barcode submission
            manualSubmit.addEventListener('click', function() {
                const barcode = barcodeInput.value.trim();
                const allergies = allergiesInput.value.trim();

                if (!barcode) {
                    alert('Please enter a valid barcode');
                    return;
                }

                loadingOverlay.style.display = 'flex';

                // Make direct API call
                fetch('/scan_barcode', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        barcode: barcode,
                        allergies: allergies
                    })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        throw new Error(data.error);
                    }
                    handleBarcodeSuccess(data);
                })
                .catch(error => {
                    console.error('Error:', error);
                    loadingOverlay.style.display = 'none';
                    alert('Failed to process barcode: ' + error.message);
                });
            });

            // Add new JavaScript for handling uploads
            const uploadBarcodeBtn = document.getElementById('uploadBarcodeBtn');
            const uploadIngredientsBtn = document.getElementById('uploadIngredientsBtn');
            const barcodeImageInput = document.getElementById('barcodeImageInput');
            const ingredientsImageInput = document.getElementById('ingredientsImageInput');
            const uploadAllergiesInput = document.getElementById('uploadAllergiesInput');

            uploadBarcodeBtn.addEventListener('click', function() {
                const file = barcodeImageInput.files[0];
                const allergies = uploadAllergiesInput.value.trim();

                if (!file) {
                    alert('Please select a barcode image');
                    return;
                }

                const formData = new FormData();
                formData.append('barcode_image', file);
                formData.append('allergies', allergies);

                loadingOverlay.style.display = 'flex';

                fetch('/upload_barcode', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => handleResponse(data))
                .catch(handleError);
            });

            uploadIngredientsBtn.addEventListener('click', function() {
                const file = ingredientsImageInput.files[0];
                const allergies = uploadAllergiesInput.value.trim();

                if (!file) {
                    alert('Please select an ingredients image');
                    return;
                }

                const formData = new FormData();
                formData.append('ingredients_image', file);
                formData.append('allergies', allergies);

                loadingOverlay.style.display = 'flex';

                fetch('/upload_ingredients', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => handleResponse(data))
                .catch(handleError);
            });

            function handleResponse(data) {
                loadingOverlay.style.display = 'none';

                if (data.error) {
                    alert('Error: ' + data.error);
                    return;
                }

                // Show product info section
                productInfo.style.display = 'block';

                if (data.product) {
                    const product = data.product.product;
                    productTitle.textContent = product.product_name || 'Product Information';

                    if (product.image_url) {
                        productImage.src = product.image_url;
                        productImage.style.display = 'block';
                    } else {
                        productImage.style.display = 'none';
                    }

                    let detailsHTML = '<p><strong>Ingredients:</strong> ' + (product.ingredients_text || 'Not available') + '</p>';
                    productDetails.innerHTML = detailsHTML;
                } else {
                    productTitle.textContent = 'Ingredients Analysis';
                    productImage.style.display = 'none';
                    let detailsHTML = '<p><strong>Extracted Ingredients:</strong> ' + data.ingredients + '</p>';
                    productDetails.innerHTML = detailsHTML;
                }

                // Display allergy analysis
                if (data.allergy_analysis) {
                    displayAllergyAnalysis(data.allergy_analysis);
                }

                // Update download button
                downloadButton.href = data.pdf_url;

                // Scroll to results
                productInfo.scrollIntoView({ behavior: 'smooth' });
            }

            function handleError(error) {
                console.error('Error:', error);
                loadingOverlay.style.display = 'none';
                alert('An error occurred. Please try again.');
            }

            function displayAllergyAnalysis(analysis) {
                allergyAnalysis.style.display = 'block';

                // Extract safety rating and status
                const ratingMatch = analysis.match(/SAFETY RATING:\s*(\d+)/i);
                const statusMatch = analysis.match(/SAFETY STATUS:\s*(SAFE|UNSAFE|CAUTION)/i);
                const conclusionMatch = analysis.match(/CONCLUSION:\s*(SAFE|UNSAFE|CAUTION)\s*-\s*(.*)/i);

                let formattedAnalysis = analysis;

                // Add prominent safety rating and status at the top
                if (ratingMatch) {
                    const rating = parseInt(ratingMatch[1]);
                    let statusColor, statusText, statusBoxColor;

                    if (rating <= 3) {
                        statusColor = 'danger';
                        statusText = 'Extremely Dangerous';
                        statusBoxColor = '#dc3545'; // Red
                    } else if (rating <= 5) {
                        statusColor = 'warning';
                        statusText = 'High Risk';
                        statusBoxColor = '#fd7e14'; // Orange
                    } else if (rating <= 7) {
                        statusColor = 'info';
                        statusText = 'Moderate Risk';
                        statusBoxColor = '#ffc107'; // Yellow
                    } else if (rating <= 9) {
                        statusColor = 'success';
                        statusText = 'Safe';
                        statusBoxColor = '#28a745'; // Light Green
                    } else {
                        statusColor = 'success';
                        statusText = 'Very Safe';
                        statusBoxColor = '#20c997'; // Green
                    }

                    formattedAnalysis = `
                        <div class="safety-rating-box" style="
                            background-color: ${statusBoxColor};
                            color: white;
                            padding: 20px;
                            border-radius: 10px;
                            margin-bottom: 20px;
                            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                        ">
                            <div class="d-flex justify-content-between align-items-center">
                                <div>
                                    <h3 class="mb-2" style="margin: 0;">Safety Rating: ${rating}/10</h3>
                                    <h4 class="mb-0" style="margin: 0;">${statusText}</h4>
                                </div>
                                <div class="text-end">
                                    <div class="progress" style="width: 120px; height: 25px; background-color: rgba(255,255,255,0.3);">
                                        <div class="progress-bar bg-white" role="progressbar"
                                             style="width: ${rating * 10}%"
                                             aria-valuenow="${rating}"
                                             aria-valuemin="0"
                                             aria-valuemax="10">
                                            ${rating}/10
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        ${formattedAnalysis}
                    `;
                }

                // Format the analysis text
                formattedAnalysis = formattedAnalysis.replace(/\\n/g, '<br>');

                // Add highlighted conclusion if available
                if (conclusionMatch) {
                    const status = conclusionMatch[1].toUpperCase();
                    const explanation = conclusionMatch[2];
                    const statusColor = status === 'SAFE' ? 'success' :
                                      status === 'UNSAFE' ? 'danger' : 'warning';
                    formattedAnalysis += `
                        <div class="alert alert-${statusColor} mt-3">
                            <strong>Final Decision:</strong> ${status}<br>
                            ${explanation}
                        </div>
                    `;
                }

                allergyAnalysis.innerHTML = formattedAnalysis;
            }
        });
    </script>
</body>
</html>
    ''')


# Add new route for image upload
@app.route('/upload_barcode', methods=['POST'])
def upload_barcode():
    try:
        logger.debug("Starting barcode upload process")
        file = request.files.get('barcode_image')
        allergies = request.form.get('allergies', '')

        if not file:
            logger.warning("No file uploaded")
            return jsonify({'error': 'No image uploaded'}), 400

        # Check file format
        allowed_extensions = {'jpg', 'jpeg', 'png', 'gif', 'heic'}
        file_extension = file.filename.rsplit(
            '.', 1)[1].lower() if '.' in file.filename else ''
        if file_extension not in allowed_extensions:
            logger.warning(f"Unsupported file format: {file_extension}")
            return jsonify({
                'error':
                'Unsupported file format. Please upload JPG, PNG, GIF, or HEIC images only.'
            }), 400

        # Save the uploaded image temporarily
        temp_img = tempfile.NamedTemporaryFile(delete=False,
                                               suffix=f'.{file_extension}')
        file.save(temp_img.name)
        logger.debug(f"Saved temporary image to {temp_img.name}")

        try:
            # Handle HEIC format
            if file_extension == 'heic':
                try:
                    # Read HEIC file
                    heif_file = pillow_heif.read_heif(temp_img.name)
                    # Convert to PIL Image
                    image = Image.frombytes(
                        heif_file.mode,
                        heif_file.size,
                        heif_file.data,
                        "raw",
                    )
                    # Save as temporary JPEG
                    jpeg_temp = tempfile.NamedTemporaryFile(delete=False,
                                                            suffix='.jpg')
                    image.save(jpeg_temp.name, format='JPEG')
                    # Clean up original HEIC temp file
                    os.unlink(temp_img.name)
                    # Update temp_img to point to the JPEG file
                    temp_img = jpeg_temp
                    logger.debug("Successfully converted HEIC to JPEG")
                except Exception as heic_error:
                    logger.error(
                        f"Error converting HEIC image: {str(heic_error)}")
                    return jsonify({
                        'error':
                        'Failed to process HEIC image. Please try converting to JPEG first.'
                    }), 500

            # Try multiple image processing techniques to improve barcode detection
            image = Image.open(temp_img.name)
            logger.debug("Successfully opened image")

            # List to store all processing attempts
            processing_attempts = []

            # Original image attempt
            processing_attempts.append(("Original", image))

            # Convert to grayscale
            if image.mode != 'L':
                gray_image = image.convert('L')
                processing_attempts.append(("Grayscale", gray_image))

            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            enhanced_image = enhancer.enhance(2.0)  # Increase contrast
            processing_attempts.append(("Enhanced Contrast", enhanced_image))

            # Convert to OpenCV format for additional processing
            cv_image = cv2.imread(temp_img.name)
            if cv_image is not None:
                # Apply adaptive thresholding
                gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
                thresh = cv2.adaptiveThreshold(gray, 255,
                                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY, 11, 2)
                thresh_pil = Image.fromarray(thresh)
                processing_attempts.append(("Adaptive Threshold", thresh_pil))

            decoded_objects = None
            successful_method = None

            # Try decoding with each processed image
            for method, processed_image in processing_attempts:
                try:
                    current_decoded = decode(processed_image)
                    if current_decoded:
                        decoded_objects = current_decoded
                        successful_method = method
                        logger.debug(
                            f"Successfully decoded barcode using {method}")
                        break
                except Exception as e:
                    logger.debug(f"Failed to decode with {method}: {str(e)}")
                    continue

            if not decoded_objects:
                logger.warning(
                    "No barcode found in image after all processing attempts")
                return jsonify({
                    'error':
                    'No barcode found in image. Please ensure the barcode is clear, well-lit, and properly focused. Try holding the camera steady and closer to the barcode.'
                }), 400

            barcode = decoded_objects[0].data.decode('utf-8')
            logger.debug(
                f"Successfully decoded barcode: {barcode} using {successful_method}"
            )

            # Get product information from API
            logger.debug(f"Fetching product info for barcode: {barcode}")
            response = requests.get(f"{API_URL}/{barcode}.json",
                                    params={'lc': 'en'})

            if response.status_code != 200:
                logger.error(
                    f"API request failed with status {response.status_code}: {response.text}"
                )
                return jsonify({
                    'error':
                    f'Failed to fetch product information. Status code: {response.status_code}'
                }), 500

            product_data = response.json()

            if product_data.get('status') == 0:
                logger.warning(f"Product not found for barcode: {barcode}")
                return jsonify({'error': 'Product not found in database'}), 404

            # Check allergies if provided
            allergy_analysis = None
            if allergies:
                logger.debug(f"Checking allergies: {allergies}")
                ingredients = product_data.get('product',
                                               {}).get('ingredients_text', '')
                if ingredients:
                    allergy_analysis = check_allergies(ingredients, allergies)
                    logger.debug("Allergy analysis completed")

            # Generate PDF
            temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
            temp_pdf.close()
            logger.debug(f"Created temporary PDF file: {temp_pdf.name}")

            generate_pdf(product_data, temp_pdf.name, allergy_analysis)
            logger.debug("PDF generated successfully")

            return jsonify({
                'success': True,
                'product': product_data,
                'allergy_analysis': allergy_analysis,
                'pdf_url': f'/download_pdf?barcode={barcode}'
            })

        except Exception as process_error:
            logger.error(
                f"Error processing barcode image: {str(process_error)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return jsonify({
                'error': 'Failed to process barcode image',
                'details': str(process_error)
            }), 500
        finally:
            # Clean up temp file
            try:
                if os.path.exists(temp_img.name):
                    os.unlink(temp_img.name)
                    logger.debug("Temporary image file cleaned up")
            except Exception as cleanup_error:
                logger.error(
                    f"Error cleaning up temporary file: {str(cleanup_error)}")

    except Exception as e:
        logger.error(f"General Error in upload_barcode: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'error': 'An unexpected error occurred',
            'details': str(e)
        }), 500


@app.route('/upload_ingredients', methods=['POST'])
def upload_ingredients():
    try:
        file = request.files.get('ingredients_image')
        allergies = request.form.get('allergies', '')

        if not file:
            return jsonify({'error': 'No image uploaded'}), 400

        # Check file format
        allowed_extensions = {'jpg', 'jpeg', 'png', 'gif', 'heic'}
        file_extension = file.filename.rsplit(
            '.', 1)[1].lower() if '.' in file.filename else ''
        if file_extension not in allowed_extensions:
            logger.warning(f"Unsupported file format: {file_extension}")
            return jsonify({
                'error':
                'Unsupported file format. Please upload JPG, PNG, GIF, or HEIC images only.'
            }), 400

        # Save the uploaded image temporarily
        temp_img = tempfile.NamedTemporaryFile(delete=False,
                                               suffix=f'.{file_extension}')
        file.save(temp_img.name)

        try:
            # Handle HEIC format
            if file_extension == 'heic':
                try:
                    # Read HEIC file
                    heif_file = pillow_heif.read_heif(temp_img.name)
                    # Convert to PIL Image
                    image = Image.frombytes(
                        heif_file.mode,
                        heif_file.size,
                        heif_file.data,
                        "raw",
                    )
                    # Save as temporary JPEG
                    jpeg_temp = tempfile.NamedTemporaryFile(delete=False,
                                                            suffix='.jpg')
                    image.save(jpeg_temp.name, format='JPEG')
                    # Clean up original HEIC temp file
                    os.unlink(temp_img.name)
                    # Update temp_img to point to the JPEG file
                    temp_img = jpeg_temp
                    logger.debug("Successfully converted HEIC to JPEG")
                except Exception as heic_error:
                    logger.error(
                        f"Error converting HEIC image: {str(heic_error)}")
                    return jsonify({
                        'error':
                        'Failed to process HEIC image. Please try converting to JPEG first.'
                    }), 500

            # Extract text from image using OCR with proper encoding
            image = Image.open(temp_img.name)
            ingredients_text = pytesseract.image_to_string(image, lang='eng')
            # Clean and normalize the text
            ingredients_text = ingredients_text.strip()
            ingredients_text = ' '.join(
                ingredients_text.split())  # Normalize whitespace
            ingredients_text = ingredients_text.encode(
                'ascii', 'ignore').decode('ascii')  # Remove non-ASCII chars

            if not ingredients_text:
                return jsonify({'error':
                                'No readable text found in image'}), 400

            # Log the extracted text for debugging
            logger.debug(f"Extracted ingredients text: {ingredients_text}")

        except Exception as ocr_error:
            logger.error(f"OCR Error: {str(ocr_error)}")
            return jsonify({'error': 'Failed to process image text'}), 500
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_img.name)
            except:
                pass

        # Check allergies against extracted ingredients
        try:
            allergy_analysis = check_allergies(ingredients_text, allergies)
        except Exception as analysis_error:
            logger.error(f"Allergy Analysis Error: {str(analysis_error)}")
            return jsonify({'error': 'Failed to analyze ingredients'}), 500

        # Generate PDF
        try:
            temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
            temp_pdf.close()

            # Create a simple product data structure for the PDF
            product_data = {
                'product': {
                    'product_name': 'Custom Product Analysis',
                    'ingredients_text': ingredients_text
                }
            }

            generate_pdf(product_data, temp_pdf.name, allergy_analysis)

            return jsonify({
                'success': True,
                'ingredients': ingredients_text,
                'allergy_analysis': allergy_analysis,
                'pdf_url': f'/download_pdf?custom=true'
            })

        except Exception as pdf_error:
            logger.error(f"PDF Generation Error: {str(pdf_error)}")
            return jsonify({'error': 'Failed to generate PDF report'}), 500

    except Exception as e:
        logger.error(f"General Error in upload_ingredients: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5004, debug=True)
