
import yt_dlp
import whisper
import json
import os
import re
import logging
from dotenv import load_dotenv
from langchain_community.document_loaders import YoutubeLoader

# Load environment variables from .env file (ensure you have one if needed)
load_dotenv()

# Configure logging
logging.basicConfig(
    filename='data_extractor.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

# Define file paths
METADATA_INPUT_FILE = 'data/music_videos_metadata.json'
PROCESSED_OUTPUT_FILE = 'data/merged_music_videos.json'
AUDIO_OUTPUT_DIR = 'data/audio'

# Ensure audio directory exists
os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)


def fetch_lyrics(video_id):
    """
    Fetches lyrics for a given YouTube video ID.

    Args:
        video_id (str): The YouTube video ID.

    Returns:
        str: The lyrics/transcript of the song.
    """
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        loader = YoutubeLoader.from_youtube_url(video_url)
        transcript = loader.load()
        lyrics = " ".join([entry['text'] for entry in transcript])
        logging.info(
            f"Lyrics fetched using transcript for video ID {video_id}.")
        return lyrics
    except Exception as e:
        logging.warning(f"Transcript not available for video ID {
                        video_id}. Error: {e}")
        print(f"Transcript not available for video ID {
              video_id}. Transcribing audio...")
        try:
            audio_file = download_audio(video_url)
            lyrics = transcribe_audio(audio_file)
            logging.info(
                f"Lyrics fetched by transcribing audio for video ID {video_id}.")
            return lyrics
        except Exception as transcribe_e:
            logging.error(f"Failed to transcribe audio for video ID {
                          video_id}: {transcribe_e}")
            print(f"Failed to transcribe audio for video ID {
                  video_id}: {transcribe_e}")
            return ""


def download_audio(video_url):
    """
    Downloads the audio from a YouTube video.

    Args:
        video_url (str): The YouTube video URL.

    Returns:
        str: The path to the downloaded audio file.
    """
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(AUDIO_OUTPUT_DIR, '%(id)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True,
        'no_warnings': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])
    video_id = video_url.split("v=")[-1]
    audio_path = os.path.join(AUDIO_OUTPUT_DIR, f"{video_id}.mp3")
    logging.info(f"Audio downloaded for video ID {video_id}: {audio_path}")
    return audio_path


def transcribe_audio(audio_file):
    """
    Transcribes audio to text using Whisper.

    Args:
        audio_file (str): The path to the audio file.

    Returns:
        str: The transcribed text.
    """
    try:
        model = whisper.load_model("base")
        result = model.transcribe(audio_file)
        transcript_text = result['text']
        logging.info(f"Audio transcribed successfully for file {audio_file}.")
        return transcript_text
    except Exception as e:
        logging.error(f"Error transcribing audio file {audio_file}: {e}")
        print(f"Error transcribing audio file {audio_file}: {e}")
        return ""


def extract_metadata(video_id):
    """
    Extracts metadata for a given YouTube video ID using yt_dlp.

    Args:
        video_id (str): The YouTube video ID.

    Returns:
        dict: A dictionary containing metadata fields.
    """
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts = {
        'quiet': True,
        'skip_download': True,
        'format': 'best',
        'forcejson': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)

        # Extract relevant metadata
        metadata = {
            'video_id': video_id,
            'title': info.get('title', 'Unknown Title'),
            'uploader': info.get('uploader', 'Unknown Artist'),
            'upload_date': info.get('upload_date', 'Unknown Date'),
            'duration': info.get('duration', 0),  # Duration in seconds
            'view_count': info.get('view_count', 0),
            'like_count': info.get('like_count', 0),
            'description': info.get('description', ''),
            'tags': info.get('tags', []),
            'categories': info.get('categories', []),
        }

        # Attempt to extract artist from title using regex
        title = metadata['title']
        artist = "Unknown Artist"

        # Pattern 1: "Artist - Song Title"
        match = re.match(r"^(?P<artist>.+?)\s*-\s*(?P<title>.+)$", title)
        if match:
            artist = match.group('artist').strip()
            song_title = match.group('title').strip()
        else:
            # Pattern 2: "Song Title by Artist"
            match = re.match(
                r"^(?P<title>.+?)\s*by\s*(?P<artist>.+)$", title, re.IGNORECASE)
            if match:
                artist = match.group('artist').strip()
                song_title = match.group('title').strip()
            else:
                # Fallback to uploader as artist
                artist = metadata['uploader']
                song_title = title

        metadata['artist'] = artist
        metadata['song_title'] = song_title

        # Define genre based on keywords in description or title
        description = metadata['description'].lower()
        genre = "Unknown Genre"
        if "pop" in description:
            genre = "Pop"
        elif "rock" in description:
            genre = "Rock"
        elif "jazz" in description:
            genre = "Jazz"
        elif "classical" in description:
            genre = "Classical"
        elif "hip-hop" in description or "hip hop" in description:
            genre = "Hip-Hop"
        elif "electronic" in description or "edm" in description:
            genre = "Electronic"
        # Add more genres as needed
        metadata['genre'] = genre

        logging.info(f"Metadata extracted for video ID {video_id}: {
                     metadata['title']} by {metadata['artist']}.")
        return metadata

    except Exception as e:
        logging.error(f"Error extracting metadata for video ID {
                      video_id}: {e}")
        print(f"Error extracting metadata for video ID {video_id}: {e}")
        return None


def process_videos():
    """
    Processes each video by fetching lyrics and extracting metadata,
    then combines them into a unified JSON structure.
    """
    # Check if metadata input file exists
    if not os.path.exists(METADATA_INPUT_FILE):
        logging.error(f"Metadata input file '{
                      METADATA_INPUT_FILE}' not found.")
        print(f"Error: Metadata input file '{METADATA_INPUT_FILE}' not found.")
        return

    # Load video metadata
    with open(METADATA_INPUT_FILE, 'r') as f:
        try:
            videos = json.load(f)
            logging.info(f"Loaded {len(videos)} videos from '{
                         METADATA_INPUT_FILE}'.")
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON from '{
                          METADATA_INPUT_FILE}': {e}")
            print(f"Error: Failed to parse '{METADATA_INPUT_FILE}': {e}")
            return

    processed_videos = []
    for idx, video in enumerate(videos, start=1):
        video_id = video.get('id')
        title = video.get('title', 'Unknown Title')

        if not video_id:
            logging.warning(f"Video entry missing 'id': {video}")
            print(f"Skipping video with missing ID: {title}")
            continue

        print(f"Processing video {
              idx}/{len(videos)}: {title} (ID: {video_id})")
        logging.info(f"Processing video {
                     idx}/{len(videos)}: {title} (ID: {video_id})")

        # Fetch lyrics
        lyrics = fetch_lyrics(video_id)
        if not lyrics:
            logging.warning(f"No lyrics found for video ID {
                            video_id}. Skipping.")
            print(f"No lyrics found for video ID {video_id}. Skipping.")
            continue

        # Extract metadata
        metadata = extract_metadata(video_id)
        if not metadata:
            logging.warning(f"Failed to extract metadata for video ID {
                            video_id}. Skipping.")
            print(f"Failed to extract metadata for video ID {
                  video_id}. Skipping.")
            continue

        # Combine all data
        combined_entry = {
            "video_id": metadata['video_id'],
            "title": metadata['song_title'],
            "artist": metadata['artist'],
            "genre": metadata['genre'],
            "lyrics": lyrics,
            "upload_date": metadata['upload_date'],
            "duration": metadata['duration'],
            "view_count": metadata['view_count'],
            "like_count": metadata['like_count'],
            "description": metadata['description'],
            "tags": metadata['tags'],
            "categories": metadata['categories']
        }

        processed_videos.append(combined_entry)
        print(f"Processed: {title}")
        logging.info(f"Processed video: {title}")

    # Save the combined data to JSON
    try:
        with open(PROCESSED_OUTPUT_FILE, 'w') as f:
            json.dump(processed_videos, f, indent=4)
        print(f"Lyrics extraction and metadata extraction completed. Data saved to '{
              PROCESSED_OUTPUT_FILE}'.")
        logging.info(f"All processing completed. Data saved to '{
                     PROCESSED_OUTPUT_FILE}'.")
    except Exception as e:
        logging.error(f"Error saving processed data to '{
                      PROCESSED_OUTPUT_FILE}': {e}")
        print(f"Error: Failed to save processed data to '{
              PROCESSED_OUTPUT_FILE}': {e}")


if __name__ == "__main__":
    process_videos()
