import sys
import os
import ollama

model="gemma2:9b-instruct-fp16"
num_ctx=8192
temperature=0.3
streaming=True

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
    
    prompt_template = """
## text
<markdown>
{context}
</markdown>

## instructions
Step 1. Read the entire text from <markdown> to </markdown>.
Step 2. Extract headings which begin with #.
Step 3. For each heading, create a summary in bullet points.
Step 4. Don't include preambles, postambles or explanations.

## summary
"""

    prompt = prompt_template.format(context=markdown)

    stream = ollama.chat(
        model=model,
        messages=[{'role': 'user', 'content': prompt}],
        options={
            "temperature": temperature,
            "num_ctx": num_ctx
        },
        stream=streaming,
    )

    if streaming:
        for chunk in stream:
            print(chunk['message']['content'], end='', flush=True)
    else:
        print(stream['message']['content'])

if __name__ == "__main__":
    main()
