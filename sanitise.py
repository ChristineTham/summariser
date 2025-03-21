#!/usr/bin/env python3
import os
import re
import argparse

def sanitize_filename(filename):
    # Replace spaces with underscores and remove special characters
    clean_name = filename.replace(' ', '_')
    clean_name = re.sub(r'[^a-zA-Z0-9_.-]', '', clean_name)
    return clean_name

def sanitize_directory(directory, recursive=False):
    for root, dirs, files in os.walk(directory):
        for filename in files:
            original_path = os.path.join(root, filename)
            sanitized_name = sanitize_filename(filename)

            if filename != sanitized_name:
                new_path = os.path.join(root, sanitized_name)
                os.rename(original_path, new_path)
                print(f"Renamed: [{original_path}] → [{new_path}]")

        # Break after first directory if not recursive
        if not recursive:
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanitise filenames")
    parser.add_argument("path", nargs="*", help="Path to a file or directory")
    parser.add_argument("-r", "--recursive", action="store_true", help="Process directories recursively")
    args = parser.parse_args()

    if not args.path:
        print("Processing current directory directory by default")
        path = "."
        sanitize_directory(path, recursive=args.recursive)
    else:
        for path in args.path:
            if os.path.isfile(path):
                sanitize_filename(path)
            elif os.path.isdir(path):
                sanitize_directory(path, recursive=args.recursive)
            else:
                print(f'The file {path} does not exist')
