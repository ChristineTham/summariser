# news2md.py
#
# Convert a URL to markdown format using newspaper3k and markdownify.
# Usage: python news2md.py <URL>
# Example: python news2md.py https://www.example.com/article-url
#
import argparse
from newspaper import Article
from markdownify import markdownify
from lxml import etree
from urllib.parse import urlparse

def generate_slug(input_string):
    # Convert input string to lowercase and replace spaces with hyphens
    slug = input_string.lower().replace(" ", "-")

    # Remove any non-alphanumeric characters (except hyphens) using a regular expression
    import re
    slug = re.sub(r"[^a-z0-9\-]", "", slug)

    return slug

def convert_url_to_markdown(url):
    # Fetch and parse the article
    article = Article(url, keep_article_html=True)
    article.download()
    article.parse()
    article.nlp()

    html_content = etree.tostring(article.top_node, encoding='unicode')
    markdown_content = markdownify(html_content,  heading_style="ATX")
    
    filename = generate_slug(article.title)
    print(f"Saving {filename}.md")
    
    with open(f"{filename}.md", "w") as f:
        f.write(f"# {article.title}\n\n")
        f.write(f"Author: {", ".join(article.authors)}  \n")
        if article.keywords:
            f.write(f"Keywords: #{" #".join(article.keywords)}  \n")
        if article.publish_date:
            f.write(f"Publish Date: {article.publish_date}  \n")
        f.write(f"[Original]({url})\n\n")
        if article.summary:
            f.write(f"## Summary\n\n")
            f.write(article.summary)
            f.write("\n\n")
        f.write(f"## Article\n\n")
        f.write(markdown_content)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="URL of the article to convert")
    args = parser.parse_args()

    # Convert the URL to Markdown and print the result
    markdown_content = convert_url_to_markdown(args.url)
