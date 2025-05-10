import json

import pandas as pd
import os

from datasets import load_dataset


def parquet_to_md():
    # Load your .parquet file
    df = pd.read_parquet("train-00000-of-00001.parquet")  # Replace with your file name

    # Create output directory
    output_dir = "markdown_files"
    os.makedirs(output_dir, exist_ok=True)

    # Loop through each row and create a markdown file
    for idx, row in df.iterrows():
        # Optional: choose which fields to use
        title = row.get("title", f"Entry {idx}")
        content = row.to_string(index=False)

        # Create markdown content
        markdown = f"# {title}\n\n```\n{content}\n```"

        # Save to file
        filename = f"{output_dir}/{idx:03d}_{str(title).replace(' ', '_')[:30]}.md"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(markdown)

    print("Markdown files created from .parquet!")


def huggingface_to_md():
    dataset = load_dataset("EthanHan/SkyrimBooks", split="train")  # Just first 5 for example
    output_dir = "markdown_files"
    os.makedirs(output_dir, exist_ok=True)

    for idx, item in enumerate(dataset):
        title = item["input"]
        title = str(title).replace("/", "-")
        content = item["output"]  # Adjust this to your dataset's column

        content = str(content).replace("\n", "  \n")  # Two spaces before newline for markdown break
        markdown = f"# {title}\n\n{content}"


        filename = f"{output_dir}/{title}.md"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(markdown)

    print("Markdown files from HF dataset generated!")

def huggingface_to_dict():
    dataset = load_dataset("EthanHan/SkyrimBooks", split="train")  # Just first 5 for example
    books = []
    for idx, item in enumerate(dataset):
        title = item["input"]
        title = str(title).replace("/", "-")
        content = item["output"]  # Adjust this to your dataset's column

        content = str(content).replace("\n", "  \n")  # Two spaces before newline for markdown break

        books.append({"title": title, "content": content})

    filename = f"data/books.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(books, f, ensure_ascii=False, indent=2)

    print("Dict files from HF dataset generated!")


if __name__ == "__main__":
    # huggingface_to_md()
    huggingface_to_dict()
    # parquet_to_md()
