from openai import OpenAI
import streamlit as st
import time
import os
from denser_retriever.keyword import (
    ElasticKeywordSearch,
    create_elasticsearch_client,
)
from denser_retriever.retriever import DenserRetriever
import fitz
import json
import logging
import anthropic
import argparse
import requests
import base64

logger = logging.getLogger(__name__)

# Define available models
MODEL_OPTIONS = {
    "GPT-4": "gpt-4o",
    "Claude 3.5": "claude-3-5-sonnet-20241022"
}
context_window = 128000
# Get API keys from environment variables with optional default values
openai_api_key = os.getenv('OPENAI_API_KEY')
claude_api_key = os.getenv('CLAUDE_API_KEY')

# Check if API keys are set
if not openai_api_key and not claude_api_key:
    raise ValueError("Neither OPENAI_API_KEY nor CLAUDE_API_KEY environment variables is set")

openai_client = OpenAI(api_key=openai_api_key)
claude_client = anthropic.Client(api_key=claude_api_key)
history_turns = 5

prompt_default = "### Instructions:\n" \
                 "You are a professional AI assistant. The following context consists of an ordered list of sources. " \
                 "If you can find answers from the context, use the context to provide a response. " \
                 "You must cite passages in square brackets [X] where X is the passage number (the ranking order of provided passages)." \
                 "If you cannot find the answer from the sources, use your knowledge to come up a reasonable answer. " \
                 "If the query asks to summarize the file or uploaded file, provide a summarization based on the provided sources. " \
                 "If the conversation involves casual talk or greetings, rely on your knowledge for an appropriate response. "


def get_annotation_pages(annotations_str):
    """Get all unique page numbers from annotations."""
    try:
        annotations = json.loads(annotations_str)
        if annotations and isinstance(annotations, list):
            return sorted(set(ann.get('page', 0) for ann in annotations))
    except (json.JSONDecodeError, AttributeError, KeyError):
        pass
    return []


def get_pdf_display_url(pdf_path):
    """Convert PDF path or URL to a data URL."""
    if pdf_path.startswith(('http://', 'https://')):
        # For URLs, download the content
        response = requests.get(pdf_path)
        pdf_content = response.content
    else:
        # For local files, read the content
        with open(pdf_path, 'rb') as file:
            pdf_content = file.read()

    # Convert to base64
    base64_pdf = base64.b64encode(pdf_content).decode('utf-8')
    return f"data:application/pdf;base64,{base64_pdf}"


def render_pdf():
    """Render PDF using PDF.js directly."""
    try:
        if not st.session_state.current_pdf:
            return

        # Get PDF content as data URL
        pdf_url = get_pdf_display_url(st.session_state.current_pdf)

        # Create viewer using PDF.js directly
        viewer_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js"></script>
            <style>
                #pdf-container {{
                    width: 100%;
                    height: 800px;
                    overflow: auto;
                    background-color: #525659;
                    text-align: center;
                }}
                canvas {{
                    background-color: white;
                    margin: 10px auto;
                    display: block;
                }}
            </style>
        </head>
        <body>
            <div id="pdf-container"></div>
            <script>
                // Initialize PDF.js
                pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';

                // Load and render PDF
                async function renderPDF() {{
                    try {{
                        const container = document.getElementById('pdf-container');
                        const loadingTask = pdfjsLib.getDocument('{pdf_url}');
                        const pdf = await loadingTask.promise;

                        // Get current page
                        const pageNum = {st.session_state.current_page + 1};
                        const page = await pdf.getPage(pageNum);

                        // Calculate scale to fit width
                        const viewport = page.getViewport({{ scale: 1.5 }});

                        // Create canvas
                        const canvas = document.createElement('canvas');
                        container.appendChild(canvas);
                        const context = canvas.getContext('2d');

                        canvas.height = viewport.height;
                        canvas.width = viewport.width;

                        // Render PDF page
                        await page.render({{
                            canvasContext: context,
                            viewport: viewport
                        }}).promise;

                        // Add highlights if any
                        const annotations = {st.session_state.current_annotations or '[]'};
                        annotations.forEach(ann => {{
                            if (ann.page === {st.session_state.current_page}) {{
                                const rect = {{
                                    x: ann.x * 1.5,
                                    y: viewport.height - (ann.y + ann.height) * 1.5,
                                    width: ann.width * 1.5,
                                    height: ann.height * 1.5
                                }};

                                context.fillStyle = 'rgba(255, 255, 0, 0.3)';
                                context.fillRect(rect.x, rect.y, rect.width, rect.height);
                            }}
                        }});

                    }} catch (error) {{
                        console.error('Error rendering PDF:', error);
                        document.getElementById('pdf-container').textContent = 'Error loading PDF: ' + error.message;
                    }}
                }}

                renderPDF();
            </script>
        </body>
        </html>
        """

        # Display the viewer
        st.components.v1.html(viewer_html, height=800)

        # Navigation controls
        nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 1])

        with nav_col1:
            if st.button("Previous Page", key="prev_btn",
                         disabled=(st.session_state.current_page <= 0)):
                st.session_state.current_page -= 1
                st.rerun()

        with nav_col2:
            if st.session_state.current_pdf.startswith(('http://', 'https://')):
                response = requests.get(st.session_state.current_pdf)
                stream = response.content
                doc = fitz.open(stream=stream, filetype="pdf")
            else:
                doc = fitz.open(st.session_state.current_pdf)
            total_pages = doc.page_count
            st.write(f"Page {st.session_state.current_page + 1} of {total_pages}")
            doc.close()

        with nav_col3:
            if st.button("Next Page", key="next_btn",
                         disabled=(st.session_state.current_page >= total_pages - 1)):
                st.session_state.current_page += 1
                st.rerun()

    except Exception as e:
        st.error(f"Error rendering PDF: {str(e)}")
        print(f"Error details: {str(e)}")  # For debugging


def stream_response(selected_model, messages, passages):
    # Generate and display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        if selected_model == "gpt-4o":
            print("Using OpenAI GPT-4 model")
            messages.insert(0, {"role": "system", "content": "You are a helpful assistant."})
            for response in openai_client.chat.completions.create(
                    model=selected_model,
                    messages=messages,
                    stream=True,
                    top_p=0,
                    temperature=0.0
            ):
                full_response += response.choices[0].delta.content or ""
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        else:
            print("Using Claude 3.5 model")
            with claude_client.messages.stream(
                    max_tokens=1024,
                    messages=messages,
                    model="claude-3-5-sonnet-20241022",
            ) as stream:
                for text in stream.text_stream:
                    full_response += text
                    message_placeholder.markdown(full_response + "▌", unsafe_allow_html=True)

            message_placeholder.markdown(full_response, unsafe_allow_html=True)

    # Update session state
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.session_state.passages = passages

    # Rerun to show the updated UI with passages
    st.rerun()


def main(args):
    # Set page configuration to use wide mode
    st.set_page_config(layout="wide")

    global retriever
    retriever = DenserRetriever(
        index_name=args.index_name,
        keyword_search=ElasticKeywordSearch(
            top_k=100,
            es_connection=create_elasticsearch_client(url="http://localhost:9200",
                                                      username="elastic",
                                                      password="",
                                                      ),
            drop_old=False,
            analysis="default"  # default or ik
        ),
        vector_db=None,
        reranker=None,
        embeddings=None,
        gradient_boost=None,
        search_fields=["annotations:keyword"],
    )

    # Initialize session states
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = "Claude 3.5"  # Default model
    if 'clicked' not in st.session_state:
        st.session_state.clicked = False
    if 'current_pdf' not in st.session_state:
        st.session_state.current_pdf = None
    if 'current_annotations' not in st.session_state:
        st.session_state.current_annotations = None
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 0
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "passages" not in st.session_state:
        st.session_state.passages = []

    # Create two columns for main content and PDF viewer
    main_col, pdf_col = st.columns([1, 1])

    with main_col:
        # Create a header row with title and model selector
        st.title("Denser Chat Demo")
        selected_model_name = st.selectbox(
            "Select Model",
            options=list(MODEL_OPTIONS.keys()),
            key="model_selector",
            index=list(MODEL_OPTIONS.keys()).index(st.session_state.selected_model)
        )
        st.session_state.selected_model = selected_model_name

        st.caption(
            "Try question \"What is in-batch negative sampling ?\" or \"what parts have stop pins?\"")
        st.divider()

        if len(st.session_state.messages) > 1:
            with st.chat_message(st.session_state.messages[-2]["role"]):
                st.markdown(st.session_state.messages[-2]["content"], unsafe_allow_html=True)

        # Display passages and add annotation buttons
        if st.session_state.passages:  # Show passages if they exist
            num_passages = len(st.session_state.passages)
            buttons_per_row = 5

            # Calculate number of rows needed
            num_rows = (num_passages + buttons_per_row - 1) // buttons_per_row

            for row in range(num_rows):
                # Create columns for this row
                start_idx = row * buttons_per_row
                end_idx = min(start_idx + buttons_per_row, num_passages)
                num_buttons_this_row = end_idx - start_idx

                # Create columns for this row
                cols = st.columns(num_buttons_this_row)

                # Add buttons to columns
                for col_idx, passage_idx in enumerate(range(start_idx, end_idx)):
                    passage = st.session_state.passages[passage_idx]
                    annotations = passage[0].metadata.get('annotations', '[]')
                    pages = get_annotation_pages(annotations)
                    page_str = f"Source {passage_idx + 1}" if pages else "No annotations"
                    # print(f"Passage {passage_idx}: {passage[0].page_content}")

                    with cols[col_idx]:
                        if st.button(page_str, key=f"btn_page_{passage_idx}"):
                            st.session_state.current_pdf = passage[0].metadata.get('source', None)
                            st.session_state.current_annotations = annotations
                            st.session_state.clicked = True
                            st.rerun()

        if len(st.session_state.messages) > 0:
            with st.chat_message(st.session_state.messages[-1]["role"]):
                st.markdown(st.session_state.messages[-1]["content"], unsafe_allow_html=True)

        # Handle user input
        query = st.chat_input("Please input your question")
        if query:
            with st.chat_message("user"):
                st.markdown(query)

            start_time = time.time()
            passages = retriever.retrieve(query, 5, {})
            retrieve_time_sec = time.time() - start_time
            st.write(f"Retrieve time: {retrieve_time_sec:.3f} sec.")

            # Process chat completion
            prompt = prompt_default + f"### Query:\n{query}\n"
            if len(passages) > 0:
                prompt += "\n### Context:\n"
                for i, passage in enumerate(passages):
                    prompt += f"#### Passage {i + 1}:\n{passage[0].page_content}\n"

            if args.language == "en":
                context_limit = 4 * context_window
            else:
                context_limit = context_window
            prompt = prompt[:context_limit] + "### Response:"

            # Prepare messages for chat completion
            messages = st.session_state.messages[-history_turns * 2:]
            messages.append({"role": "user", "content": prompt})

            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": query})

            stream_response(MODEL_OPTIONS[selected_model_name], messages, passages)

    # Render PDF viewer in the second column
    with pdf_col:
        render_pdf()


def parse_args():
    parser = argparse.ArgumentParser(description='Denser Chat Demo')
    parser.add_argument('--index_name', type=str, default=None,
                        help='Name of the Elasticsearch index to use')
    parser.add_argument('--language', type=str, default='en',
                        help='Language setting for context window (en or ch, default: en)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
