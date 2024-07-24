import sys
import os
import re
from typing import List, Tuple, Optional
from collections import namedtuple
from tqdm import tqdm
import ollama

FENCES = ["```", "~~~"]
MAX_HEADING_LEVEL = 6

Chapter = namedtuple("Chapter", "parent_headings, heading, text")

MODEL="gemma2:27b-instruct-fp16"
NUM_CTX=8192
TEMPERATURE=0.3
chunk_delimiter="\n#"
detail=0.5
BLOCKSIZE=NUM_CTX

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


# This function chunks a text into smaller pieces based on a maximum token count and a delimiter.
def chunk_on_delimiter(input_string: str,
                       max_chunksize: int, delimiter: str) -> List[str]:
    chunks = input_string.split(delimiter)
    combined_chunks, _, dropped_chunk_count = combine_chunks_with_no_minimum(
        chunks, max_chunksize, chunk_delimiter=delimiter, add_ellipsis_for_overflow=False
    )
    if dropped_chunk_count > 0:
        print(f"warning: {dropped_chunk_count} chunk(s) overflow")
    combined_chunks = [f"{chunk}{delimiter}" for chunk in combined_chunks]
    if len(combined_chunks[-1]) < max_chunksize / 2:
        combined_chunks[-2] += delimiter + combined_chunks[-1]
        combined_chunks.pop()
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
              model: str = MODEL,
              additional_instructions: Optional[str] = None,
              minimum_chunk_size: Optional[int] = NUM_CTX / 2,
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
    # max_chunks = len(chunk_on_delimiter(text, minimum_chunk_size, chunk_delimiter))
    # min_chunks = 1
    # num_chunks = int(min_chunks + detail * (max_chunks - min_chunks))

    # adjust chunk_size based on interpolated number of chunks
    # document_length = len(text)
    # chunk_size = max(minimum_chunk_size, document_length // num_chunks)
    # text_chunks = chunk_on_delimiter(text, chunk_size, chunk_delimiter)

    sections = ["\n".join(s.text) for s in split_by_heading(text.split("\n"))]
    text_chunks = group_sections(sections)
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
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    with open(summary_file, 'w') as file:
        file.write(summary)
    print(f'Converted to Markdown summary {summary_file}')

def main():
    # Check if the user provided a file name as an argument
    if len(sys.argv) < 2:
        print("Please provide a filename containing text or Markdown.")
        return

    filename = sys.argv[1]

    # Check if the file exists
    if not os.path.isfile(filename):
        print("The file does not exist.")
        return

    with open(filename, 'r') as file:
        markdown = file.read()

    print("Length: ", len(markdown))

    summary = summarize(markdown, detail=detail, verbose=True)
    output_summary(filename, summary)
    # sections = ["\n".join(s.text) for s in split_by_heading(markdown.split("\n"))]
    # for i,s in enumerate(sections):
    #     print(f"Section {i} length {len(s)}")
    # blocks = group_sections(sections)
    # for i,s in enumerate(blocks):
    #     print(f"Block {i} length {len(s)}")

if __name__ == "__main__":
    main()
