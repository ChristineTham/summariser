import sys
import os
from gensim.summarization import summarize
from gensim.summarization import keywords

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

    # print("Length: ", len(markdown))
    
    print("## Keywords\n")
    kwlist = keywords(markdown, ratio=0.01).split()
    for kw in kwlist:
        print(f"* [[{kw}]]")
    print("\n## Summary\n")
    print(summarize(markdown, word_count=100))

if __name__ == "__main__":
    main()
