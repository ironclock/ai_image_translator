# AI Image Text Translator

This project is an AI-powered image text translator that uses OCR to detect text in images, translates it, and then overlays the translated text back onto the image while preserving the original style and layout.

## Features

- Upload images and select text areas for translation
- Detect text using OCR (Optical Character Recognition)
- Translate text using OpenAI's GPT-3.5 model
- Preserve original text color and alignment
- Estimate and match original font size
- Inpaint the background for seamless text replacement

## Requirements

- Python 3.8+
- Django 5.1.2
- OpenCV
- Tesseract OCR
- OpenAI API key

For a full list of Python package requirements, see `requirements.txt`.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/ai-image-text-translator.git
   cd ai-image-text-translator
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Install Tesseract OCR:
   - On Ubuntu: `sudo apt-get install tesseract-ocr`
   - On macOS: `brew install tesseract`
   - On Windows: Download and install from [GitHub Tesseract releases](https://github.com/UB-Mannheim/tesseract/wiki)

5. Set up your environment variables:
   Create a `.env` file in the project root and add your OpenAI API key:
   ```
   OPENAI_KEY=your_openai_api_key_here
   ```

6. Run migrations:
   ```
   python manage.py migrate
   ```

7. Start the development server:
   ```
   python manage.py runserver
   ```

## Usage

1. Navigate to `http://localhost:8000` in your web browser.
2. Upload an image and select the area containing the text you want to translate.
3. Enter the source and target languages.
4. Click "Translate" and wait for the process to complete.
5. View and download the translated image.

## Areas for Improvement

- **Text Alignment**: Improve the precision of text placement to better match the original layout.
- **Text Color Matching**: Enhance color detection and application to ensure translated text matches the original color, including handling gradients and transparency.
- **Font Matching**: Implement more accurate font detection and matching to closely replicate the original style.
- **Font Size Matching**: Refine the font size estimation for a more consistent match with the original text.
- **Text Detection**: Utilize more advanced OCR models to improve accuracy, especially for complex or non-standard text layouts.
- **Inpainting**: Improve the background inpainting process for smoother and more seamless text replacement.


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenAI for their GPT-3.5 model
- Tesseract OCR project
- OpenCV community
