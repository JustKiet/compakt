from typing import Protocol


class FileReader(Protocol):
    def read(self, file_path: str) -> str:
        """
        Reads the content of a file given its path.

        Args:
            file_path (str): The path to the file to be read.
        Returns:
            str: The content of the file as a string.
        """
        ...


class FileReaderAsMarkdown(FileReader, Protocol):
    def read(self, file_path: str) -> str:
        """
        Reads the content of a file given its path and returns it as markdown.

        Args:
            file_path (str): The path to the file to be read.
        Returns:
            str: The content of the file as a markdown string.
        """
        ...
