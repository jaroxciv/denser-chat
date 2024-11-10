import fitz
import json
from typing import List, Tuple, Dict
import argparse
import os
import requests
import tempfile
from urllib.parse import urlparse


class PDFPassageProcessor:
    def __init__(self, input_path: str, chars_per_passage: int = 1000):
        """
        Initialize the PDF processor.

        Args:
            input_path: Path or URL to the input PDF file
            chars_per_passage: Number of characters per passage
        """
        self.input_path = input_path
        self.chars_per_passage = chars_per_passage
        self.temp_file = None

        # If input is URL, download it first
        if self._is_url(input_path):
            self.temp_file = self._download_pdf(input_path)
            self.doc = fitz.open(self.temp_file.name)
        else:
            self.doc = fitz.open(input_path)

    def _is_url(self, path: str) -> bool:
        """Check if the input path is a URL."""
        try:
            result = urlparse(path)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False

    def _download_pdf(self, url: str) -> tempfile.NamedTemporaryFile:
        """Download PDF from URL to a temporary file."""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()

        # Create temporary file
        temp = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                temp.write(chunk)
        temp.close()
        return temp

    def extract_text_with_positions(self) -> List[Tuple[str, int, fitz.Rect, int]]:
        """
        Extract text and position information from the PDF.

        Returns:
            List of tuples containing (text, page_number, text_rectangle, char_count)
        """
        text_positions = []

        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            blocks = page.get_text("dict")["blocks"]

            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span["text"]
                            if text.strip():  # Skip empty text
                                bbox = fitz.Rect(span["bbox"])
                                char_count = len(text)
                                text_positions.append((text, page_num, bbox, char_count))

        return text_positions

    def create_passages(self, text_positions: List[Tuple[str, int, fitz.Rect, int]]) -> List[Dict]:
        """
        Create passages with PDF.js compatible annotations.
        """
        passages = []
        current_passage = ""
        current_positions = []
        char_count = 0

        for text, page_num, bbox, text_char_count in text_positions:
            new_char_count = char_count + text_char_count + 1

            # Get page dimensions for coordinate conversion
            page = self.doc[page_num]
            page_height = page.rect.height

            if new_char_count > self.chars_per_passage and current_passage:
                passage_dict = {
                    "page_content": current_passage.strip(),
                    "metadata": {
                        "source": self.input_path,
                        "title": "",
                        "pid": str(len(passages)),
                        "annotations": json.dumps(current_positions)
                    },
                    "type": "Document"
                }
                passages.append(passage_dict)

                # Convert coordinates for PDF.js
                current_passage = text + " "
                current_positions = [{
                    "page": page_num,
                    "x": bbox.x0,
                    "y": page_height - bbox.y1,  # Convert y-coordinate
                    "width": bbox.width,
                    "height": bbox.height,
                    "color": "#FFFF00"
                }]
                char_count = text_char_count + 1
            else:
                current_passage += text + " "
                current_positions.append({
                    "page": page_num,
                    "x": bbox.x0,
                    "y": page_height - bbox.y1,  # Convert y-coordinate
                    "width": bbox.width,
                    "height": bbox.height,
                    "color": "#FFFF00"
                })
                char_count = new_char_count

        if current_passage.strip():
            passage_dict = {
                "page_content": current_passage.strip(),
                "metadata": {
                    "source": self.input_path,
                    "title": "",
                    "pid": str(len(passages)),
                    "annotations": json.dumps(current_positions)
                },
                "type": "Document"
            }
            passages.append(passage_dict)

        return passages

    def highlight_passages(self, passages: List[Dict], output_path: str):
        """
        Create a new PDF with highlighted passages.

        Args:
            passages: List of passage dictionaries
            output_path: Path where to save the output PDF
        """
        # Create a copy of the document for highlighting
        output_doc = self.doc

        # Highlight each passage
        for passage in passages:
            annotations = json.loads(passage["metadata"]["annotations"])
            for annotation in annotations:
                page = output_doc[annotation["page"]]
                bbox = fitz.Rect(
                    annotation["x"],
                    annotation["y"],
                    annotation["x"] + annotation["width"],
                    annotation["y"] + annotation["height"]
                )
                highlight = page.add_highlight_annot(bbox)
                highlight.set_colors(stroke=(1, 1, 0))  # Yellow color
                highlight.update()

        # Save the highlighted PDF
        output_doc.save(output_path)
        output_doc.close()

    def process_pdf(self, output_pdf_path: str, output_jsonl_path: str):
        """
        Process the PDF file and create highlighted output and JSONL file.

        Args:
            output_pdf_path: Path where to save the highlighted PDF
            output_jsonl_path: Path where to save the JSONL file
        """
        # Extract text with positions
        text_positions = self.extract_text_with_positions()

        # Create passages
        passages = self.create_passages(text_positions)

        # Ensure the directory for output_jsonl_path exists
        output_dir = os.path.dirname(output_jsonl_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Save passages to JSONL file
        with open(output_jsonl_path, 'w', encoding='utf-8') as f:
            for passage in passages:
                f.write(json.dumps(passage, ensure_ascii=False) + '\n')

        # Highlight passages in the PDF
        self.highlight_passages(passages, output_pdf_path)

        return passages

    def close(self):
        if self.temp_file:
            try:
                os.unlink(self.temp_file.name)
            except:
                pass


def main():
    parser = argparse.ArgumentParser(description='Process PDF file into passages and highlight them.')
    parser.add_argument('input_pdf', help='Path or URL to the input PDF file')
    parser.add_argument('output_pdf', help='Path to save the highlighted PDF file')
    parser.add_argument('output_jsonl', help='Path to save the passages JSONL file')
    parser.add_argument('--chars', type=int, default=2000, help='Characters per passage (default: 1000)')

    args = parser.parse_args()

    processor = PDFPassageProcessor(args.input_pdf, args.chars)
    try:
        passages = processor.process_pdf(args.output_pdf, args.output_jsonl)
        print(f"Processed {len(passages)} passages.")
        print(f"Created highlighted PDF: {args.output_pdf}")
        print(f"Created passages JSONL: {args.output_jsonl}")
    finally:
        processor.close()


if __name__ == "__main__":
    main()
