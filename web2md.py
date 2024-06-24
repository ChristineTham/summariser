# web2md.py
#
# Convert a URL to markdown format using readabilipy and markdownify.
# Usage: python news2md.py <URL>
# Example: python news2md.py https://www.example.com/article-url
#
import sys
import os
from readability import Document
from markdownify import markdownify
from urllib.request import Request, urlopen

def generate_slug(input_string):
    # Convert input string to lowercase and replace spaces with hyphens
    slug = input_string.lower().replace(" ", "-")

    # Remove any non-alphanumeric characters (except hyphens) using a regular expression
    import re
    slug = re.sub(r"[^a-z0-9\-]", "", slug)

    return slug

# Check if URL is provided as a command-line argument
if len(sys.argv) != 2:
    print("Usage: python script.py <URL>")
    sys.exit(1)

url = sys.argv[1]

try:
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})

    web_byte = urlopen(req).read()

    webpage = web_byte.decode('utf-8')

    doc = Document(webpage)

    # Extract the main content using readability
    content = doc.summary()

    # Convert to markdown using markdownify
    markdown_content = markdownify(content,  heading_style="ATX")
    
    filename = generate_slug(doc.title())
    print(f"Saving {filename}.md")
    
    with open(f"{filename}.md", "w") as f:
        f.write(f"# {doc.title()}\n\n")
        f.write(f"[Original]({url})\n\n")
        f.write(markdown_content)
    
except Exception as e:
    print(f"Error processing {url}: {e}")