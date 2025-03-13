{ pkgs }: {
    deps = [
        pkgs.python39
        pkgs.python39Packages.pip
        pkgs.python39Packages.flask
        pkgs.tesseract
        pkgs.zbar
        pkgs.opencv
        pkgs.python39Packages.pillow
        pkgs.python39Packages.requests
        pkgs.python39Packages.python-dotenv
        pkgs.python39Packages.fpdf
        pkgs.python39Packages.google-generativeai
        pkgs.python39Packages.pytesseract
        pkgs.python39Packages.pyzbar
        pkgs.python39Packages.opencv-python
        pkgs.python39Packages.pillow-heif
    ];
} 