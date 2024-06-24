from __future__ import unicode_literals
import sys
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
import yt_dlp

def get_video_metadata(url):
    ydl_opts = {}

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=False)
        # print(ydl.sanitize_info(info_dict))
        return info_dict
    
def extract_youtube_id(url):
    parsed_url = urlparse(url)
    video_id = None

    if 'youtu.be' in parsed_url.netloc:
        video_id = parsed_url.path[1:]
    elif 'youtube.com' in parsed_url.netloc and 'watch' in parsed_url.path:
        query_params = parse_qs(parsed_url.query)
        video_id = query_params['v'][0]
    else:
        raise ValueError("Invalid YouTube URL")

    return video_id

def download_transcript(video_url):
    try:
        video = get_video_metadata(video_url)
        video_id = video["id"]
        # video_id = extract_youtube_id(video_url)
        # transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
        # print(transcripts)
        transcript = YouTubeTranscriptApi.get_transcript(video_id, preserve_formatting=True)

        with open(f"{video_id}.md", "w") as f:
            f.write(f"# {video["title"]}\n\n")
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
