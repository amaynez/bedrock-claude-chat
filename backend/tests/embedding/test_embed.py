import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.append(".")

from embedding.loaders import UrlLoader
from embedding.loaders.base import BaseLoader
from embedding.main import embed
from embedding.wrapper import DocumentSplitter, Embedder
from llama_index.core.node_parser import SentenceSplitter
from unstructured.nlp import tokenize
from unstructured.partition.auto import partition
from unstructured.partition.html import partition_html


class TestEmbed(unittest.TestCase):
    def test_load_split(self):
        source_urls = [
            "https://github.com/aws-samples/bedrock-claude-chat",
        ]
        loader = UrlLoader(source_urls)
        documents = loader.load()
        print(documents)

        self.assertNotEqual(len(documents), 0)

        splitter = DocumentSplitter(
            splitter=SentenceSplitter(
                paragraph_separator=r"\n\n\n",
                chunk_size=100,
                chunk_overlap=50,
            )
        )

        splitted = splitter.split_documents(documents)
        print(splitted)

        self.assertNotEqual(len(splitted), 0)

    def test_partition_offline_no_spacy_download(self):
        html_text = (
            "This is an HTML narrative sentence for offline partitioning. "
            "It verifies HTML parsing works without network access."
        )
        txt_text = (
            "This is a plain text narrative sentence for offline partitioning. "
            "It verifies TXT parsing works without network access."
        )
        md_text = (
            "This is a markdown narrative sentence for offline partitioning. "
            "It verifies Markdown parsing works without network access."
        )
        tokenize._get_nlp.cache_clear()
        with patch(
            "unstructured.nlp.tokenize._install_spacy_model",
            side_effect=AssertionError("Unexpected runtime spaCy model download"),
        ):
            html_elements = partition_html(
                text=f"<html><body><p>{html_text}</p></body></html>"
            )
            self.assertIn("HTML narrative", "\n".join(map(str, html_elements)))

            with tempfile.TemporaryDirectory() as tmpdir:
                for name, content, expected in (
                    ("sample.txt", txt_text, "plain text narrative"),
                    ("sample.md", f"# Title\n\n{md_text}", "markdown narrative"),
                ):
                    path = os.path.join(tmpdir, name)
                    with open(path, "w") as f:
                        f.write(content)
                    file_elements = partition(filename=path)
                    self.assertIn(expected, "\n".join(map(str, file_elements)))


if __name__ == "__main__":
    """To run the test, use the following command on `backend` directory:

    ```
    cd backend
    docker build -f embedding.Dockerfile -t embedding .
    docker run -it -v $(pwd)/tests:/src/tests embedding /src/tests/embedding/test_embed.py
    ```
    """
    unittest.main()
