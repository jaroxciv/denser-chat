import os
import argparse
import glob
import shutil
from denser_chat.indexer import Indexer
from denser_chat.pdf_processor import PDFPassageProcessor
from denser_chat.html_processor import HTMLPassageProcessor
from urllib.parse import urlparse

def is_url(path: str) -> bool:
    """Check if the input path is a URL."""
    try:
        result = urlparse(path)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def is_html_url(url: str) -> bool:
    """Check if URL is likely an HTML page (not a PDF)."""
    return is_url(url) and not url.lower().endswith('.pdf')

def process_single_file(input_file, output_dir):
    """Process a single file (PDF or HTML) and return passage file path."""
    # Get base name for output files
    if is_url(input_file):
        base_name = os.path.splitext(os.path.basename(urlparse(input_file).path))[0]
        if not base_name:  # Handle URLs without file paths
            base_name = "webpage_" + str(hash(input_file))[:8]
    else:
        base_name = os.path.splitext(os.path.basename(input_file))[0]

    # Define output path for passages
    passage_file = os.path.join(output_dir, f"{base_name}_passages.jsonl")

    if is_html_url(input_file):
        # Process HTML page
        processor = HTMLPassageProcessor(input_file, 1000)
        passages = processor.process_html(passage_file)
        print(f"Processed {len(passages)} passages from HTML: {input_file}")
        print(f"Title: {processor.title}")
    else:
        # Process PDF file
        annotated_pdf = os.path.join(output_dir, f"{base_name}_annotated.pdf")
        processor = PDFPassageProcessor(input_file, 1000)
        try:
            passages = processor.process_pdf(annotated_pdf, passage_file)
            print(f"Processed {len(passages)} passages from PDF: {input_file}")
        finally:
            processor.close()

    return passage_file

def concatenate_passage_files(output_dir):
    """Concatenate all individual passage files into one final passages.jsonl"""
    final_passage_file = os.path.join(output_dir, "passages.jsonl")

    with open(final_passage_file, 'w') as outfile:
        passage_files = glob.glob(os.path.join(output_dir, "*_passages.jsonl"))
        for passage_file in passage_files:
            with open(passage_file, 'r') as infile:
                shutil.copyfileobj(infile, outfile)

    return final_passage_file

def read_sources_file(sources_file):
    """Read file paths from a sources file."""
    with open(sources_file, 'r') as f:
        sources = f.read().split()
        return [source.strip() for source in sources if source.strip()]

def main(sources_file, output_dir, index_name):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Read source files
    input_files = read_sources_file(sources_file)
    if not input_files:
        print(f"No files found in {sources_file}")
        return

    # Process each file
    passage_files = []
    for input_file in input_files:
        try:
            passage_file = process_single_file(input_file, output_dir)
            passage_files.append(passage_file)
        except Exception as e:
            print(f"Error processing {input_file}: {str(e)}")
            continue

    if not passage_files:
        print("No files were successfully processed.")
        return

    # Concatenate all passage files into one
    final_passage_file = concatenate_passage_files(output_dir)
    print(f"Created combined passage file: {final_passage_file}")

    # Index the combined passages
    indexer = Indexer(index_name)
    indexer.index(final_passage_file)
    print(f"Indexed passages to {index_name}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Process PDFs and HTML pages listed in sources file and create an index.")

    parser.add_argument(
        'sources_file',
        type=str,
        help="Path to the sources.txt file containing list of PDFs and URLs"
    )
    parser.add_argument(
        'output_dir',
        type=str,
        help="Directory where output files will be stored"
    )
    parser.add_argument(
        'index_name',
        type=str,
        help="Name for the index to be created"
    )

    args = parser.parse_args()

    main(args.sources_file, args.output_dir, args.index_name)