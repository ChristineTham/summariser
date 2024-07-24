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
from typing import List, Tuple, Optional
from collections import namedtuple
from tqdm import tqdm
import ollama
from sumy.nlp.tokenizers import Tokenizer
from sumy.nlp.stemmers import Stemmer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lsa import LsaSummarizer     
from sumy.utils import get_stop_words
import yake

MODEL="gemma2:27b-instruct-fp16"
NUM_CTX=8192
TEMPERATURE=0.3
LANGUAGE = "english"
SENTENCES_COUNT = 10
BLOCKSIZE=NUM_CTX * 1.5
MIN_BLOCKSIZE=NUM_CTX / 2
FENCES = ["```", "~~~"]
MAX_HEADING_LEVEL = 6

Chapter = namedtuple("Chapter", "parent_headings, heading, text")

system_prompt = """
You are an efficient text summarizer.

## Instructions

Step 1. Read the entire text.
Step 2. Extract headings which begin with #.
Step 3. Include each heading in the output.
Step 4. For each heading, create a summary in bullet points.
Step 5. Don't include preambles, postambles or explanations.
"""

# This function chunks a text into smaller pieces based on a maximum token count and a delimiter.
def chunk_on_delimiter(input_string: str,
                       max_chunksize: int, delimiter: str) -> List[str]:
    chunks = input_string.split(delimiter)
    combined_chunks, _, dropped_chunk_count = combine_chunks_with_no_minimum(
        chunks, max_chunksize, chunk_delimiter=delimiter, add_ellipsis_for_overflow=True
    )
    if dropped_chunk_count > 0:
        print(f"warning: {dropped_chunk_count} chunks were dropped due to overflow")
    combined_chunks = [f"{chunk}{delimiter}" for chunk in combined_chunks]
    return combined_chunks

def split_block(block, blocksize=BLOCKSIZE):
    groups = []
    rem_block = block
    while len(rem_block) > 0:
        split_index = rem_block.rfind("\n", 0, blocksize)
        if split_index != -1:
            groups.append(rem_block[:split_index])
            rem_block = rem_block[split_index+1:]
        else:
            groups.append(rem_block)
            rem_block = []
    return groups

def group_sections(sections, blocksize=BLOCKSIZE):
    """
    Groups a list of sections into a new list, where each group contains
    no more than `blocksize` characters. If a group is larger than `blocksize`, it will be
    split into additional groups using the newline character ('\\n') as the delimiter.
    """
    # Initialize an empty list to store the groups
    groups = []

    # Initialize an empty string to store the current group
    curr_group = ""

    # Loop over each string in the input list
    for s in sections:
        # If adding the current string to the current group would exceed `blocksize`,
        # check if the current group is empty. If it is, add the current string as a
        # separate group and continue with the next string. Otherwise, split the current
        # group into additional groups using newline as the delimiter and continue processing
        # the remaining part of the current string.
        if len(curr_group) + len(s) > blocksize:
            if len(curr_group) > 0:
                groups.append(curr_group)
            curr_group = ""
            if (len(s) > blocksize):
                groups.extend(split_block(s))
            else:
                curr_group = s
        else:
            # Otherwise, add the current string to the current group with a space
            # separator.
            curr_group += "\n" + s

    # Add any remaining characters in the last group to the list of groups.
    if curr_group:
        if len(curr_group) < MIN_BLOCKSIZE:
            groups[-1] += "\n" + curr_group
        else:
            groups.append(curr_group)

    return groups

def split_by_heading(text, max_level=3):
    """
    Generator that returns a list of chapters from text.
    Each chapter's text includes the heading line.
    """
    curr_parent_headings = [None] * MAX_HEADING_LEVEL
    curr_heading_line = None
    curr_lines = []
    within_fence = False
    for next_line in text:
        next_line = Line(next_line)

        if next_line.is_fence():
            within_fence = not within_fence

        is_chapter_finished = (
            not within_fence and next_line.is_heading() and next_line.heading_level <= max_level
        )
        if is_chapter_finished:
            if len(curr_lines) > 0:
                parents = __get_parents(curr_parent_headings, curr_heading_line)
                yield Chapter(parents, curr_heading_line, curr_lines)

                if curr_heading_line is not None:
                    curr_level = curr_heading_line.heading_level
                    curr_parent_headings[curr_level - 1] = curr_heading_line.heading_title
                    for level in range(curr_level, MAX_HEADING_LEVEL):
                        curr_parent_headings[level] = None

            curr_heading_line = next_line
            curr_lines = []

        curr_lines.append(next_line.full_line)
    parents = __get_parents(curr_parent_headings, curr_heading_line)
    yield Chapter(parents, curr_heading_line, curr_lines)


def __get_parents(parent_headings, heading_line):
    if heading_line is None:
        return []
    max_level = heading_line.heading_level
    trunc = list(parent_headings)[: (max_level - 1)]
    return [h for h in trunc if h is not None]


class Line:
    """
    Detect code blocks and ATX headings.

    Headings are detected according to commonmark, e.g.:
    - only 6 valid levels
    - up to three spaces before the first # is ok
    - empty heading is valid
    - closing hashes are stripped
    - whitespace around title are stripped
    """

    def __init__(self, line):
        self.full_line = line
        self._detect_heading(line)

    def _detect_heading(self, line):
        self.heading_level = 0
        self.heading_title = None
        result = re.search("^[ ]{0,3}(#+)(.*)", line)
        if result is not None and (len(result[1]) <= MAX_HEADING_LEVEL):
            title = result[2]
            if len(title) > 0 and not (title.startswith(" ") or title.startswith("\t")):
                # if there is a title it must start with space or tab
                return
            self.heading_level = len(result[1])

            # strip whitespace and closing hashes
            title = title.strip().rstrip("#").rstrip()
            self.heading_title = title

    def is_fence(self):
        for fence in FENCES:
            if self.full_line.startswith(fence):
                return True
        return False

    def is_heading(self):
        return self.heading_level > 0

def summarize(text: str,
              model: str = MODEL,
              additional_instructions: Optional[str] = None,
              summarize_recursively=False,
              verbose=False):
    """
    Summarizes a given text by splitting it into chunks, each of which is summarized individually. 
    The process can optionally be made recursive.

    Parameters:
    - text (str): The text to be summarized.
    - model (str, optional): The model to use for generating summaries. Defaults to MODEL.
    - additional_instructions (Optional[str], optional): Additional instructions to provide to the model for customizing summaries.
    - summarize_recursively (bool, optional): If True, summaries are generated recursively, using previous summaries for context.
    - verbose (bool, optional): If True, prints detailed information about the chunking process.

    Returns:
    - str: The final compiled summary of the text.

    The function first determines the number of chunks by interpolating between a minimum and a maximum chunk count based on the `detail` parameter. 
    It then splits the text into chunks and summarizes each chunk. If `summarize_recursively` is True, each summary is based on the previous summaries, 
    adding more context to the summarization process. The function returns a compiled summary of all chunks.
    """

    sections = ["\n".join(s.text) for s in split_by_heading(text.split("\n"))]
    text_chunks = group_sections(sections)

    if verbose:
        print(f"Splitting the text into {len(text_chunks)} chunks to be summarized.")
        print(f"Chunk lengths are {[len(x) for x in text_chunks]}")

    # set system message
    system_message_content = system_prompt

    if additional_instructions is not None:
        system_message_content += f"\n\n{additional_instructions}"

    accumulated_summaries = []
    for chunk in tqdm(text_chunks):
        if summarize_recursively and accumulated_summaries:
            # Creating a structured prompt for recursive summarization
            accumulated_summaries_string = '\n\n'.join(accumulated_summaries)
            user_message_content = f"## Previous summaries:\n\n{accumulated_summaries_string}\n\n## Text to summarize next:\n\n{chunk}"
        else:
            # Directly passing the chunk for summarization without recursive context
            user_message_content = "## Text to summarize\n" + chunk

        # Constructing messages based on whether recursive summarization is applied
        messages = [
            {"role": "system", "content": system_message_content},
            {"role": "user", "content": user_message_content}
        ]

        # print(messages)
        # Assuming this function gets the completion and works as expected
        response = ollama.chat(
            model=model,
            messages=messages,
            options={
                "temperature": TEMPERATURE,
                "num_ctx": NUM_CTX
            },
        )
        accumulated_summaries.append(response['message']['content'])

    # Compile final summary from partial summaries
    final_summary = '\n\n'.join(accumulated_summaries)

    return final_summary

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
    original = output_file(filename, "")
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    with open(summary_file, 'w') as file:
        file.write(summary)
        file.write(f"\n\n[Original]({original})\n")
    print(f'Converted to Markdown summary [{summary_file}]\n')
    
def output_md(filename, markdown):
    if (len(markdown) > BLOCKSIZE):
        summary = summarize(markdown, verbose=True)
    else:
        response = ollama.chat(
            model=MODEL,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': "## text\n" + markdown}
            ],
            options={
                "temperature": TEMPERATURE,
                "num_ctx": NUM_CTX
            }
        )
        summary = response['message']['content']
        print(f"Total {(response["total_duration"]/1e9):.1f}s Load {(response["load_duration"]/1e9):.1f}s Prompt {(response["prompt_eval_duration"]/1e9):.1f}s Eval {(response["eval_duration"]/1e9):.1f}s", file=sys.stderr)

    # Extract keywords using YAKE
    summary += "\n## Keywords\n\n"
    kw_extractor = yake.KeywordExtractor()
    keywords = kw_extractor.extract_keywords(markdown)

    for kw in keywords:
        summary += f"* [[{kw[0]}]]\n" 
    
    # Summarize using sumy LSA       
    # summary += "\n## Abstract\n\n"  
    # parser = PlaintextParser.from_string(markdown, Tokenizer(LANGUAGE))
    # stemmer = Stemmer(LANGUAGE)
    # summarizer = LsaSummarizer(stemmer)     
    # summarizer.stop_words = get_stop_words(LANGUAGE)                  
    # for sentence in summarizer(parser.document, SENTENCES_COUNT):
    #     summary += f"* {sentence}\n"

    output_summary(filename, summary)
    
def process_path(path):
    _, file_extension = os.path.splitext(path)
    if (file_extension == ".md" or file_extension == ".txt"):
        print(f"Processing file: [{path}]")
    else:
        print(f"Skipping unknown file [{path}]")
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
