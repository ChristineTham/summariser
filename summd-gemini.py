# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "google-api-core",
#     "google-genai",
#     "tqdm",
#     "mdsplit",
# ]
# ///

# summd.py
#
# Generic summariser for text/markdown files
# Takes filenames or directories as command line arguments
# (by default all files in "_markdown" directory)
# Then summarise each file into "_output/file-summ.md"
# as Markdown with headings retained where appropriate and bullet points in content.
# Accepted file extensions: .md, .txt

import sys
import os
import re
import argparse
import pprint
from mdsplit import split_by_heading
from tqdm import tqdm
from google import genai
from google.genai import types

MODEL = "gemini-2.5-pro-preview-05-06"
# MODEL = "gemini-2.5-flash-preview-05-20"
LEVEL=0
INPUT_DIR = "_markdown/"
OUTPUT_DIR = "_summary/"
MAX_TOKENS = 8192
TEMPERATURE=0.3

client = genai.Client()

system_prompt = """
You are an efficient text summarizer.

## Instructions

Step 1. Read the entire text (comes after <Summarise>).
Step 2. Extract headings which begin with #.
Step 3. Include each heading in the output.
Step 4. For each heading, consider the key points in the text and create a summary in bullet points.
Step 5. Don't include preambles, postambles or explanations.
"""

def summarise_by_heading(markdown: str, verbose: bool):
    chunks = ["\n".join(s.text) + "\n" for s in split_by_heading(markdown.split("\n"), LEVEL)]

    if verbose:
        print(f"Splitting the text into {len(chunks)} chunks to be summarized.")
        print(f"Chunk lengths are {[len(x) for x in chunks]}")

    # set system message
    system_message_content = system_prompt

    accumulated_summaries = []
    for chunk in tqdm(chunks):
        user_message_content = "<Summarise>\n" + chunk
        summary = None

        while not summary:
            response = client.models.generate_content(
                model=MODEL,
                contents=[user_message_content],
                config=types.GenerateContentConfig(
                    system_instruction=system_message_content,
                    max_output_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                ),
            )
            summary = response.text

        accumulated_summaries.append(summary)

    # Compile final summary from partial summaries
    final_summary = '\n\n'.join(accumulated_summaries)

    return final_summary

def output_file(s, prefix):
    # Strip _input/ prefix if it exists
    if s.startswith(INPUT_DIR):
        s = s[len(INPUT_DIR):]

    # Add prefix
    s = os.path.join(prefix, s)

    return s

def output_summary(filename: str, summary: str):
    base = os.path.splitext(filename)[0]

    summary_file = output_file(f"{base}.md", OUTPUT_DIR)
    original = output_file(filename, "")
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    with open(summary_file, 'w') as file:
        file.write(summary)
        file.write(f"\n\n[Original]({original})\n")
    print(f'Output: [{summary_file}]\n')

def summarise(markdown: str):
    summary = None

    while not summary:
        response = client.models.generate_content(
            model=MODEL,
            contents=["<Summarise>\n", markdown],
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            ),
        )
        summary = response.text

    return summary

def output_md(filename, markdown):
    summary = ""
    if LEVEL > 0:
        summary = summarise_by_heading(markdown, verbose=True)
    else:
        summary = summarise(markdown)

    # Generate keywords
    summary = summary + "\n\n## Keywords\n\n"

    keywords = None

    while not keywords:
        response = client.models.generate_content(
            model=MODEL,
            contents=["Generate a set of keywords for the following text. Do not add preamble, postamble or explanations. Put keywords in a Markdown list with each keyword enclosed in [[ and ]]:", markdown],
            config=types.GenerateContentConfig(
                max_output_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            ),
        )
        keywords = response.text

    summary += keywords

    output_summary(filename, summary)

def process_path(path):
    _, file_extension = os.path.splitext(path)
    if (file_extension == ".md" or file_extension == ".txt"):
        print(f"Processing file: [{path}]")
    else:
        print(f"Skipping unknown file [{path}]")
        return

    base = os.path.splitext(path)[0]
    output = output_file(f"{base}.md", OUTPUT_DIR)
    if os.path.isfile(output):
        print(f"Skipping existing  [{output}]")
        return

    with open(path, 'r') as file:
        output_md(path, file.read())

def process_dir(folder):
    for foldername, _, filenames in os.walk(folder):
        for filename in filenames:
            path = os.path.join(foldername, filename)
            process_path(path)

def main():
    global MODEL, LEVEL, OUTPUT_DIR, model, tokenizer

    parser = argparse.ArgumentParser(description="Summarise markdown files")
    parser.add_argument("path", nargs="*", help="Path to a file or directory (default: _markdown)")
    parser.add_argument("-m", "--model", type=str, default=MODEL,
                        help=f"MLX Model (default: {MODEL})")
    parser.add_argument("-l", "--level", type=int, default=LEVEL,
                        help=f"Level of heading to break (default: none)")
    parser.add_argument("-o", "--output", type=str, default=OUTPUT_DIR,
                        help=f"Output directory (default: {OUTPUT_DIR})")

    args = parser.parse_args()
    MODEL = args.model
    LEVEL = args.level
    OUTPUT_DIR = args.output

    print(f"Model: {MODEL}\nOutput dir: [{OUTPUT_DIR}]")
    if LEVEL:
        print(f"Level: {LEVEL}")

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
