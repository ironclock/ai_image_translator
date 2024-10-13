from django.shortcuts import render, redirect
from django.views import View
from django.http import JsonResponse
from django.urls import reverse
from .models import TranslationJob
import pytesseract
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from openai import OpenAI
from django.core.files.base import ContentFile
import io
import base64
import logging
import cv2
import numpy as np
from decouple import config
from sklearn.cluster import KMeans
import logging

logger = logging.getLogger(__name__)

OPENAI_KEY = config('OPENAI_KEY')


class TranslationJobView(View):
    def get(self, request):
        return render(request, 'translator/upload.html')

    def post(self, request):
        try:
            # Extract data from the request
            source_language = request.POST.get('source_language')
            target_language = request.POST.get('target_language')
            image_data = request.POST.get('image')
            x = int(float(request.POST.get('x')))
            y = int(float(request.POST.get('y')))
            width = int(float(request.POST.get('width')))
            height = int(float(request.POST.get('height')))
            scale_factor = float(request.POST.get('scale_factor', 1))

            # Process the image
            format, imgstr = image_data.split(';base64,')
            image = Image.open(io.BytesIO(base64.b64decode(imgstr)))

            # Convert PIL Image to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # Create a mask for the text area
            mask = np.zeros(cv_image.shape[:2], np.uint8)
            mask[y:y+height, x:x+width] = 255

            # Perform inpainting
            inpainted_image = cv2.inpaint(
                cv_image, mask, 10, cv2.INPAINT_TELEA)

            # Convert back to PIL Image
            inpainted_pil = Image.fromarray(
                cv2.cvtColor(inpainted_image, cv2.COLOR_BGR2RGB))

            # Perform OCR on the original cropped area
            cropped_image = image.crop((x, y, x + width, y + height))
            text = pytesseract.image_to_string(cropped_image)

            text_color = self.detect_text_color(image, x, y, width, height)
            print(text)
            print(text_color)
            # Translate the text
            client = OpenAI(api_key=OPENAI_KEY)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"You are a translator. Translate the following text from {
                        source_language} to {target_language}."},
                    {"role": "user", "content": text}
                ]
            )
            translated_text = response.choices[0].message.content

            # Estimate font size
            estimated_font_size = self.estimate_font_size(cropped_image, text)

            # Load font
            try:
                font = ImageFont.truetype(
                    '/Users/joncaceres/NotoSansSC-Regular.ttf', estimated_font_size)
            except IOError:
                try:
                    font = ImageFont.truetype(
                        '/Users/joncaceres/NotoSans-Regular.ttf', estimated_font_size)
                except IOError:
                    font = ImageFont.load_default()
                    print(
                        "Warning: Using default font. Chinese characters may not display correctly.")

            alignment = self.detect_alignment(image, x, y, width, height)

            # Wrap the translated text
            wrapped_text = self.wrap_text(translated_text, font, width)

            # Draw the new text on the inpainted image
            self.draw_text_with_outline_and_alignment(
                inpainted_pil, (x, y, width, height), wrapped_text, font, alignment, text_color=text_color, outline_color=text_color)

            # Save the new image
            buffer = io.BytesIO()
            inpainted_pil.save(buffer, format='PNG')

            # Create and save the TranslationJob
            job = TranslationJob(
                source_language=source_language,
                target_language=target_language
            )
            job.original_image.save('original.png', ContentFile(
                base64.b64decode(imgstr.encode())))
            job.translated_image.save(
                'translated.png', ContentFile(buffer.getvalue()))
            job.save()

            result_url = reverse('translation_result',
                                 kwargs={'job_id': job.id})
            logger.info(
                f"Translation completed. Returning result URL: {result_url}")
            return JsonResponse({'result_url': result_url})

        except Exception as e:
            logger.error(f"Error during translation: {str(e)}")
            return JsonResponse({'error': str(e)}, status=500)

    def estimate_font_size(self, image, text):
        # Perform OCR with Tesseract to get bounding box information
        ocr_data = pytesseract.image_to_data(
            image, output_type=pytesseract.Output.DICT)

        # Filter out empty results
        valid_indices = [i for i, t in enumerate(
            ocr_data['text']) if t.strip()]

        if not valid_indices:
            print("No text found in the image.")
            return 12  # Default font size if no text found

        # Calculate average height of text boxes
        heights = [ocr_data['height'][i] for i in valid_indices]
        avg_height = np.mean(heights)

        # Estimate font size based on average height
        estimated_size = int(avg_height * 0.8)  # Adjust this factor as needed

        print(f"Estimated font size: {estimated_size}")

        # Ensure the font size is within a reasonable range
        return max(min(estimated_size, 72), 8)  # Min 8, Max 72

    def wrap_text(self, text, font, max_width):
        words = text.split()
        lines = []
        current_line = []
        for word in words:
            if font.getlength(' '.join(current_line + [word])) <= max_width:
                current_line.append(word)
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
        lines.append(' '.join(current_line))
        return lines

    def draw_text_with_outline_and_alignment(self, image, pos, text_lines, font, alignment, text_color=(0, 0, 0), outline_color=(255, 255, 255)):
        draw = ImageDraw.Draw(image)
        x, y, width, height = pos
        for line in text_lines:
            # Calculate text width
            text_width, text_height = font.getbbox(
                line)[2], font.getbbox(line)[3]

            # Adjust x position based on alignment
            if alignment == 'center':
                line_x = x + (width - text_width) // 2
            elif alignment == 'right':
                line_x = x + width - text_width
            else:  # left alignment
                line_x = x

            # Draw the outline
            for adj in range(-1, 2):
                for adj_y in range(-1, 2):
                    draw.text((line_x+adj, y+adj_y), line,
                              font=font, fill=outline_color)
            # Draw the text
            draw.text((line_x, y), line, font=font, fill=text_color)
            y += text_height + 2  # Add a small vertical padding

    def detect_alignment(self, image, x, y, width, height):
        # Crop the region of interest
        roi = image.crop((x, y, x + width, y + height))

        # Perform OCR with Tesseract
        ocr_data = pytesseract.image_to_data(
            roi, output_type=pytesseract.Output.DICT)

        # Filter out empty results
        valid_indices = [i for i, text in enumerate(
            ocr_data['text']) if text.strip()]

        if not valid_indices:
            print("No text found in the selected area.")
            return 'left'  # Default to left if no text found

        # Calculate left and right margins for each text block
        left_margins = [ocr_data['left'][i] for i in valid_indices]
        right_margins = [
            width - (ocr_data['left'][i] + ocr_data['width'][i]) for i in valid_indices]

        # Calculate average margins
        avg_left_margin = np.mean(left_margins)
        avg_right_margin = np.mean(right_margins)

        # Determine alignment
        margin_difference = abs(avg_left_margin - avg_right_margin)
        center_threshold = width * 0.05  # Reduced from 10% to 5% of width
        alignment_ratio = avg_left_margin / \
            (avg_left_margin + avg_right_margin)

        print(f"Average left margin: {avg_left_margin}")
        print(f"Average right margin: {avg_right_margin}")
        print(f"Margin difference: {margin_difference}")
        print(f"Center threshold: {center_threshold}")
        print(f"Alignment ratio: {alignment_ratio}")

        if margin_difference < center_threshold and 0.45 < alignment_ratio < 0.55:
            alignment = 'center'
        elif alignment_ratio < 0.45:
            alignment = 'right'
        else:
            alignment = 'left'

        print(f"Detected alignment: {alignment}")
        return alignment

    def detect_text_color(self, image, x, y, width, height):
        # Convert PIL Image to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Crop the region of interest
        roi = cv_image[y:y+height, x:x+width]

        # Reshape the image to be a list of pixels
        pixels = roi.reshape((-1, 3))

        # Convert to float
        pixels = np.float32(pixels)

        # Define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        k = 2
        _, labels, centers = cv2.kmeans(
            pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Convert back to 8 bit values
        centers = np.uint8(centers)

        # Flatten the labels array
        labels = labels.flatten()

        # Count occurrences of each label
        label_counts = np.bincount(labels)

        # Determine which cluster is the background (assumed to be the larger cluster)
        background_label = np.argmax(label_counts)
        foreground_label = 1 - background_label  # Since we only have 2 clusters

        # Get the color of the foreground (text) cluster
        text_color = centers[foreground_label]

        # Convert BGR to RGB
        return tuple(reversed(text_color))


class TranslationResultView(View):
    def get(self, request, job_id):
        job = TranslationJob.objects.get(id=job_id)
        return render(request, 'translator/result.html', {'job': job})
