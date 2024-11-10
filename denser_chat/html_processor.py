import json
from typing import List, Dict
import argparse
import os
import requests
from unstructured.partition.html import partition_html
from bs4 import BeautifulSoup


class HTMLPassageProcessor:
    def __init__(self, url: str, chars_per_passage: int = 1000):
        """
        Initialize the HTML processor.

        Args:
            url: URL of the HTML page
            chars_per_passage: Number of characters per passage
        """
        self.url = url
        self.chars_per_passage = chars_per_passage
        self.content = self._load_page()
        self.title = self._get_title()

    def _load_page(self) -> str:
        """Load HTML content."""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(self.url, headers=headers)
        response.raise_for_status()
        return response.text

    def _get_title(self) -> str:
        """Extract page title from HTML content."""
        soup = BeautifulSoup(self.content, 'html.parser')
        if soup.title:
            return soup.title.string.strip()
        return ""

    def create_passages(self) -> List[Dict]:
        """
        Create passages from the HTML content.

        Returns:
            List of dictionaries containing passage information
        """
        # Parse HTML content using unstructured
        elements = partition_html(text=self.content)

        passages = []
        current_passage = ""
        char_count = 0

        for element in elements:
            text = str(element)
            if not text.strip():
                continue

            new_char_count = char_count + len(text) + 1  # +1 for space

            if new_char_count > self.chars_per_passage and current_passage:
                # Create passage dictionary with current content
                passage_dict = {
                    "page_content": current_passage.strip(),
                    "metadata": {
                        "source": self.url,
                        "title": self.title,
                        "pid": str(len(passages))
                    },
                    "type": "Document"
                }
                passages.append(passage_dict)

                # Reset for next passage
                current_passage = text + " "
                char_count = len(text) + 1
            else:
                current_passage += text + " "
                char_count = new_char_count

        # Add the remaining text as the last passage
        if current_passage.strip():
            passage_dict = {
                "page_content": current_passage.strip(),
                "metadata": {
                    "source": self.url,
                    "title": self.title,
                    "pid": str(len(passages))
                },
                "type": "Document"
            }
            passages.append(passage_dict)

        return passages

    def process_html(self, output_jsonl_path: str) -> List[Dict]:
        """
        Process the HTML file and create JSONL file.

        Args:
            output_jsonl_path: Path where to save the JSONL file
        """
        # Create passages
        passages = self.create_passages()

        # Ensure the directory for output_jsonl_path exists
        output_dir = os.path.dirname(output_jsonl_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Save passages to JSONL file
        with open(output_jsonl_path, 'w', encoding='utf-8') as f:
            for passage in passages:
                f.write(json.dumps(passage, ensure_ascii=False) + '\n')

        return passages


def main():
    parser = argparse.ArgumentParser(description='Process HTML page into passages.')
    parser.add_argument('input_url', help='URL of the HTML page')
    parser.add_argument('output_jsonl', help='Path to save the passages JSONL file')
    parser.add_argument('--chars', type=int, default=1000,
                        help='Characters per passage (default: 1000)')

    args = parser.parse_args()

    processor = HTMLPassageProcessor(args.input_url, args.chars)
    try:
        passages = processor.process_html(args.output_jsonl)
        print(f"Processed {len(passages)} passages.")
        print(f"Title: {processor.title}")
        print(f"Created passages JSONL: {args.output_jsonl}")
    except Exception as e:
        print(f"Error processing URL: {str(e)}")
        raise


if __name__ == "__main__":
    main()