# src/upload_to_chromadb.py

import json
import os
import logging
from dotenv import load_dotenv

from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

# Load environment variables from .env file (if using one)
load_dotenv()

# Configure logging
logging.basicConfig(
    filename='upload_to_chromadb.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

# Define file paths
MERGED_FILE = 'data/merged_music_videos.json'  # Path to your merged JSON file
# Directory to persist ChromaDB data
PERSIST_DIRECTORY = 'data/vector_db'


def load_merged_data(file_path):
    """
    Loads the merged JSON data containing transcripts and metadata.

    Args:
        file_path (str): Path to the merged JSON file.

    Returns:
        list: A list of video entries as dictionaries.
    """
    if not os.path.exists(file_path):
        logging.error(f"Merged JSON file '{file_path}' not found.")
        raise FileNotFoundError(f"Merged JSON file '{file_path}' not found.")

    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            logging.info(f"Loaded {len(data)} entries from '{file_path}'.")
            return data
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON from '{file_path}': {e}")
            raise e


def sanitize_metadata(metadata):
    """
    Converts list-type metadata fields into comma-separated strings.

    Args:
        metadata (dict): The original metadata dictionary.

    Returns:
        dict: The sanitized metadata dictionary with scalar values.
    """
    sanitized = {}
    for key, value in metadata.items():
        if isinstance(value, list):
            # Convert list to a comma-separated string
            sanitized[key] = ', '.join(map(str, value))
            logging.debug(f"Converted list to string for key '{
                          key}': {sanitized[key]}")
        elif isinstance(value, dict):
            # Optionally handle nested dictionaries if present
            sanitized[key] = json.dumps(value)
            logging.debug(f"Converted dict to JSON string for key '{
                          key}': {sanitized[key]}")
        else:
            sanitized[key] = value
    return sanitized


def create_documents(data):
    """
    Converts each video entry into a Document object with sanitized metadata.

    Args:
        data (list): List of video entries as dictionaries.

    Returns:
        list: List of Document objects.
    """
    documents = []
    for entry in data:
        lyrics = entry.get('lyrics', '')
        if not lyrics:
            logging.warning(f"No lyrics found for video ID {
                            entry.get('video_id')}. Skipping.")
            continue  # Skip entries without lyrics

        # Extract and sanitize metadata
        metadata = {
            "video_id": entry.get('video_id', ''),
            "title": entry.get('title', ''),
            "artist": entry.get('artist', ''),
            "genre": entry.get('genre', ''),
            "upload_date": entry.get('upload_date', ''),
            "duration": entry.get('duration', 0),
            "view_count": entry.get('view_count', 0),
            "like_count": entry.get('like_count', 0),
            "description": entry.get('description', ''),
            "tags": entry.get('tags', []),
            "categories": entry.get('categories', [])
        }
