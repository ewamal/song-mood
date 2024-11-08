import json
import os
from textblob import TextBlob
import logging

logging.basicConfig(
    filename='mood_analyzer.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

MERGED_FILE = 'data/merged_music_videos.json'


def analyze_mood(lyrics):
    """
    Classifies the mood of the lyrics based on sentiment polarity.

    Args:
        lyrics (str): The lyrics of the song.

    Returns:
        str: The classified mood ("Happy", "Sad", or "Chill").
    """
    try:
        analysis = TextBlob(lyrics)
        polarity = analysis.sentiment.polarity
        if polarity > 0.2:
            return "Happy"
        elif polarity < -0.2:
            return "Sad"
        else:
            return "Chill"
    except Exception as e:
        logging.error(f"Error analyzing mood: {e}")
        return "Unknown"


def process_videos():
    """
    Processes each video by analyzing its mood and updating the JSON file.
    """
    # Check if the merged JSON file exists
    if not os.path.exists(MERGED_FILE):
        logging.error(f"Error: '{MERGED_FILE}' not found.")
        print(f"Error: '{MERGED_FILE}' not found.")
        return

    # Load the merged JSON file
    with open(MERGED_FILE, 'r') as f:
        try:
            merged_videos = json.load(f)
            logging.info(f"Loaded {len(merged_videos)
                                   } videos from '{MERGED_FILE}'.")
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON from '{MERGED_FILE}': {e}")
            print(f"Error: Failed to parse '{MERGED_FILE}': {e}")
            return

    # Analyze mood and update each video entry
    for idx, video in enumerate(merged_videos, start=1):
        video_id = video.get('video_id')
        title = video.get('title', 'Unknown Title')
        lyrics = video.get('lyrics', '')

        if not lyrics:
            logging.warning(f"No lyrics found for video ID {
                            video_id}. Skipping mood analysis.")
            print(f"No lyrics found for video '{
                  title}' (ID: {video_id}). Skipping mood analysis.")
            video["mood"] = "Unknown"
            continue

        # Analyze mood
        mood = analyze_mood(lyrics)

        # Update the video entry with mood
        video["mood"] = mood
        print(f"Processed [{idx}/{len(merged_videos)}]: {title}, Mood: {mood}")
        logging.info(
            f"Processed [{idx}/{len(merged_videos)}]: {title}, Mood: {mood}")

    # Save the updated JSON with mood information
    try:
        with open(MERGED_FILE, 'w') as f:
            json.dump(merged_videos, f, indent=4)
        print(f"Mood analysis completed. Data updated in '{MERGED_FILE}'.")
        logging.info(f"Mood analysis completed. Data updated in '{
                     MERGED_FILE}'.")
    except Exception as e:
        logging.error(f"Error saving updated JSON file: {e}")
        print(f"Error: Failed to save updated data to '{MERGED_FILE}': {e}")


if __name__ == "__main__":
    process_videos()
