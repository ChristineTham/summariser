# yt2md.py
#
# Convert a Youtube Video transcript to markdown
# Usage: python yt2md.py <URL>
# Example: python yt2md.py https://www.youtube.com/watch?v=dQw4w9WgXcQ
#

import sys
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
import yt_dlp

def generate_slug(input_string):
    # Convert input string to lowercase and replace spaces with hyphens
    slug = input_string.lower().replace(" ", "-")

    # Remove any non-alphanumeric characters (except hyphens) using a regular expression
    import re
    slug = re.sub(r"[^a-z0-9\-]", "", slug)

    return slug

def get_video_metadata(url):
    ydl_opts = {}

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=False)
        # print(ydl.sanitize_info(info_dict))
        return info_dict

def download_transcript(video_url):
    try:
        video = get_video_metadata(video_url)
        video_id = video["id"]
        # video_id = extract_youtube_id(video_url)
        # transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
        # print(transcripts)
        transcript = YouTubeTranscriptApi.get_transcript(video_id, preserve_formatting=True)
        
        filename = generate_slug(video["title"])
        print(f"Saving {filename}.md")
        
        with open(f"{filename}.md", "w") as f:
            f.write(f"# {video["title"]}\n\n")
            f.write(f"ID: {video_id}  \n")
            f.write(f"[Original]({video_url})\n\n")
            f.write(f"## Description\n\n```text\n{video["description"]}\n```\n\n")
            f.write("## Transcript\n\n")
            for line in transcript:
                f.write(line['text'] + '\n')
    except Exception as e:
        print(f"Error processing {video_url}: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python download_transcripts.py [YouTube video URLs]")
        sys.exit(1)

    for url in sys.argv[1:]:
        download_transcript(url)
