from docling.document_converter import DocumentConverter

from compakt.core.interfaces.file_reader import FileReader


class DoclingFileReader(FileReader):
    PAGE_BREAK_VALUE = "<--PAGEBREAK-->"

    def __init__(self, document_converter: DocumentConverter) -> None:
        super().__init__()
        self.document_converter = document_converter

    def read(self, file_path: str) -> str:
        doc = self.document_converter.convert(file_path)

        md = doc.document.export_to_markdown(page_break_placeholder=self.PAGE_BREAK_VALUE)

        return md
