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
        narrative = (
            "This is a complete narrative sentence for offline partitioning. "
            "It verifies that multiple sentences parse without network access."
        )
        tokenize._get_nlp.cache_clear()
        with patch(
            "unstructured.nlp.tokenize._install_spacy_model",
            side_effect=AssertionError("Unexpected runtime spaCy model download"),
        ):
            html_elements = partition_html(
                text=f"<html><body><p>{narrative}</p></body></html>"
            )
            self.assertIn("offline partitioning", "\n".join(map(str, html_elements)))

            with tempfile.TemporaryDirectory() as tmpdir:
                for name, content in (
                    ("sample.txt", narrative),
                    ("sample.md", f"# Title\n\n{narrative}"),
                ):
                    path = os.path.join(tmpdir, name)
                    with open(path, "w") as f:
                        f.write(content)
                    file_elements = partition(filename=path)
                    self.assertIn(
                        "offline partitioning", "\n".join(map(str, file_elements))
                    )


if __name__ == "__main__":
    """To run the test, use the following command on `backend` directory:

    ```
    cd backend
    docker build -f embedding.Dockerfile -t embedding .
    docker run -it -v $(pwd)/tests:/src/tests embedding /src/tests/embedding/test_embed.py
    ```
    """
    unittest.main()
