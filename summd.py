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
import argparse

import ollama

model="gemma2:9b-instruct-fp16"
num_ctx=8192
temperature=0.3

def output_file(s, prefix):
    input_prefix = "_markdown/"

    # Strip _input/ prefix if it exists
    if s.startswith(input_prefix):
        s = s[len(input_prefix):]

    # Add _output/ prefix
    s = prefix + s

    return s

def output_summary(filename: str, summary: str):
    base = os.path.splitext(filename)[0]
    
    summary_file = output_file(f"{base}.md", "_output/")
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    with open(summary_file, 'w') as file:
        file.write(summary)
    print(f'Converted to Markdown summary {summary_file}')
    
def output_md(filename, markdown):
    system_prompt = """
You are an efficient text summarizer.

## instructions
Step 1. Read the entire text.
Step 2. Extract headings which begin with #.
Step 3. For each heading, create a summary in bullet points.
Step 4. Don't include preambles, postambles or explanations.
"""

    response = ollama.chat(
        model=model,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': "## text\n" + markdown}
        ],
        options={
            "temperature": temperature,
            "num_ctx": num_ctx
        }
    )
    
    output_summary(filename, response['message']['content'])
    
def process_path(path):
    _, file_extension = os.path.splitext(path)
    if (file_extension == ".md" or file_extension == ".txt"):
        print("Processing file:", path)
    else:
        print("Skipping unknown file ", path)
        return

    with open(path, 'r') as file:
        output_md(path, file.read())

def process_dir(folder):
    for foldername, _, filenames in os.walk(folder):
        for filename in filenames:
            path = os.path.join(foldername, filename)
            process_path(path)            
            
def main():
    if len(sys.argv) < 2:
        print("Processing _markdown directory by default")
        path = "_markdown"
        process_dir(path)
    else:
        parser = argparse.ArgumentParser(description="summarise.py [file|dir] ...")
        parser.add_argument("path", nargs="+", help="Path to a file or directory")
        args = parser.parse_args()

        for path in args.path:
            if os.path.isfile(path):
                process_path(path)
            elif os.path.isdir(path):
                process_dir(path)
            else:
                print(f'The file {path} does not exist')

if __name__ == "__main__":
    main()
