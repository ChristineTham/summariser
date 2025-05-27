# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "google-api-core",
#     "google-genai",
#     "tqdm",
# ]
# ///

# sump.py
#
# Summarise markdown files into a single paragraph on standard output
# Takes filenames or directories as command line arguments
# (by default all files in "_markdown" directory)
# Then summarise each file into "_output/file-summ.md"
# as Markdown with headings retained where appropriate and bullet points in content.
# Accepted file extensions: .md

import sys
import os
import re
import argparse
from google import genai
from google.genai import types

# MODEL = "gemini-2.5-pro-preview-03-25"
MODEL = "gemini-2.5-flash-preview-05-20"
client = genai.Client()

MAX_TOKENS = 8192
TEMPERATURE=0.3
INPUT_DIR = "_markdown/"

def output_summary(filename, markdown):
        response = client.models.generate_content(
            model=MODEL,
            contents=["Generate a set of keywords for the following text. Put keywords in a Markdown list with each keyword enclosed in [[ and ]]:", markdown],
            config=types.GenerateContentConfig(
                max_output_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            ),
        )
        summary = response.text
        print(summary)

def process_path(path):
    # _, file_extension = os.path.splitext(path)
    # if (file_extension == ".md" or file_extension == ".txt"):
    #     print(f"Processing file: [{path}]")
    # else:
    #     print(f"Skipping unknown file [{path}]")
    #     return

    with open(path, 'r') as file:
        output_summary(path, file.read())

def process_dir(folder):
    for foldername, _, filenames in os.walk(folder):
        for filename in filenames:
            path = os.path.join(foldername, filename)
            process_path(path)

def main():
    global MODEL, model, tokenizer

    parser = argparse.ArgumentParser(description="Summarise markdown files into a single paragraph")
    parser.add_argument("path", nargs="*", help="Path to a file or directory (default: _markdown)")
    parser.add_argument("-m", "--model", type=str, default=MODEL,
                        help=f"Gemini Model (default: {MODEL})")

    args = parser.parse_args()
    MODEL = args.model
    if not args.path:
        print("Processing _markdown directory by default")
        path = "_markdown"
        process_dir(path)
    else:
        for path in args.path:
            if os.path.isfile(path):
                process_path(path)
            elif os.path.isdir(path):
                process_dir(path)
            else:
                print(f'The file {path} does not exist')

if __name__ == "__main__":
    main()
