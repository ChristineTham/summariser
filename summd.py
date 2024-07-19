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
from typing import List, Tuple, Optional
from tqdm import tqdm
import ollama

model="gemma2:27b-instruct-fp16"
num_ctx=8192
temperature=0.3
chunk_delimiter="\n#"
detail=0.5

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


# This function combines text chunks into larger blocks without exceeding a specified token count. It returns the combined text blocks, their original indices, and the count of chunks dropped due to overflow.
def combine_chunks_with_no_minimum(
        chunks: List[str],
        max_chunksize: int,
        chunk_delimiter=chunk_delimiter,
        header: Optional[str] = None,
        add_ellipsis_for_overflow=False,
) -> Tuple[List[str], List[int]]:
    dropped_chunk_count = 0
    output = []  # list to hold the final combined chunks
    output_indices = []  # list to hold the indices of the final combined chunks
    candidate = (
        [] if header is None else [header]
    )  # list to hold the current combined chunk candidate
    candidate_indices = []
    for chunk_i, chunk in enumerate(chunks):
        chunk_with_header = [chunk] if header is None else [header, chunk]
        if len(chunk_delimiter.join(chunk_with_header)) > max_chunksize:
            print("warning: chunk overflow")
            if (
                    add_ellipsis_for_overflow
                    and len(chunk_delimiter.join(candidate + ["..."])) <= max_chunksize
            ):
                candidate.append("...")
                dropped_chunk_count += 1
            continue  # this case would break downstream assumptions
        # estimate token count with the current chunk added
        extended_candidate_token_count = len(chunk_delimiter.join(candidate + [chunk]))
        # If the token count exceeds max_tokens, add the current candidate to output and start a new candidate
        if extended_candidate_token_count > max_chunksize:
            output.append(chunk_delimiter.join(candidate))
            output_indices.append(candidate_indices)
            candidate = chunk_with_header  # re-initialize candidate
            candidate_indices = [chunk_i]
        # otherwise keep extending the candidate
        else:
            candidate.append(chunk)
            candidate_indices.append(chunk_i)
    # add the remaining candidate to output if it's not empty
    if (header is not None and len(candidate) > 1) or (header is None and len(candidate) > 0):
        output.append(chunk_delimiter.join(candidate))
        output_indices.append(candidate_indices)
    return output, output_indices, dropped_chunk_count

def summarize(text: str,
              detail: float = 0,
              model: str = model,
              additional_instructions: Optional[str] = None,
              minimum_chunk_size: Optional[int] = num_ctx / 2,
              chunk_delimiter: str = ".",
              summarize_recursively=False,
              verbose=False):
    """
    Summarizes a given text by splitting it into chunks, each of which is summarized individually. 
    The level of detail in the summary can be adjusted, and the process can optionally be made recursive.

    Parameters:
    - text (str): The text to be summarized.
    - detail (float, optional): A value between 0 and 1 indicating the desired level of detail in the summary.
      0 leads to a higher level summary, and 1 results in a more detailed summary. Defaults to 0.
    - model (str, optional): The model to use for generating summaries. Defaults to 'gpt-3.5-turbo'.
    - additional_instructions (Optional[str], optional): Additional instructions to provide to the model for customizing summaries.
    - minimum_chunk_size (Optional[int], optional): The minimum size for text chunks. Defaults to 500.
    - chunk_delimiter (str, optional): The delimiter used to split the text into chunks. Defaults to ".".
    - summarize_recursively (bool, optional): If True, summaries are generated recursively, using previous summaries for context.
    - verbose (bool, optional): If True, prints detailed information about the chunking process.

    Returns:
    - str: The final compiled summary of the text.

    The function first determines the number of chunks by interpolating between a minimum and a maximum chunk count based on the `detail` parameter. 
    It then splits the text into chunks and summarizes each chunk. If `summarize_recursively` is True, each summary is based on the previous summaries, 
    adding more context to the summarization process. The function returns a compiled summary of all chunks.
    """

    # check detail is set correctly
    assert 0 <= detail <= 1

    # interpolate the number of chunks based to get specified level of detail
    max_chunks = len(chunk_on_delimiter(text, minimum_chunk_size, chunk_delimiter))
    min_chunks = 1
    num_chunks = int(min_chunks + detail * (max_chunks - min_chunks))

    # adjust chunk_size based on interpolated number of chunks
    document_length = len(text)
    chunk_size = max(minimum_chunk_size, document_length // num_chunks)
    text_chunks = chunk_on_delimiter(text, chunk_size, chunk_delimiter)
    if verbose:
        print(f"Splitting the text into {len(text_chunks)} chunks to be summarized.")
        print(f"Chunk lengths are {[len(x) for x in text_chunks]}")

    # set system message
    system_message_content = """
You are an efficient text summarizer.

## instructions
Step 1. Read the entire text.
Step 2. Extract headings which begin with #.
Step 3. For each heading, create a summary in bullet points.
Step 4. Don't include preambles, postambles or explanations.
"""
    if additional_instructions is not None:
        system_message_content += f"\n\n{additional_instructions}"

    accumulated_summaries = []
    for chunk in tqdm(text_chunks):
        if summarize_recursively and accumulated_summaries:
            # Creating a structured prompt for recursive summarization
            accumulated_summaries_string = '\n\n'.join(accumulated_summaries)
            user_message_content = f"Previous summaries:\n\n{accumulated_summaries_string}\n\nText to summarize next:\n\n{chunk}"
        else:
            # Directly passing the chunk for summarization without recursive context
            user_message_content = "## text\n" + chunk

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
                "temperature": temperature,
                "num_ctx": num_ctx
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
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    with open(summary_file, 'w') as file:
        file.write(summary)
    print(f'Converted to Markdown summary {summary_file}')
    
def output_md(filename, markdown):
    if (len(markdown) > num_ctx):
        summary = summarize(markdown, detail=detail, verbose=True)
    else:
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
        summary = response['message']['content']
    
    output_summary(filename, summary)
    
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
