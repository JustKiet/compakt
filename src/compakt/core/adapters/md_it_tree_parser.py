from markdown_it import MarkdownIt

from compakt.core.interfaces.md_tree_parser import MarkdownTreeParser
from compakt.core.models import HeaderNode


class MarkdownItTreeParser(MarkdownTreeParser):
    def __init__(self, markdown_it: MarkdownIt) -> None:
        super().__init__()
        self._md = markdown_it

    def parse(self, markdown_text: str) -> list[HeaderNode]:
        """
        Parses a markdown string and constructs a tree structure based on the headings.

        Args:
            markdown_text (str): The input markdown text to be parsed.
        Returns:
            list[HeaderNode]: A list of HeaderNode representing the hierarchical structure of the markdown.
        """
        tokens = self._md.parse(markdown_text)

        root: HeaderNode = {"title": "root", "level": 0, "children": []}
        stack = [root]

        i = 0
        while i < len(tokens):
            token = tokens[i]

            if token.type == "heading_open":
                level = int(token.tag[1])  # h1 -> 1, h2 -> 2

                inline = tokens[i + 1]
                title = inline.content

                node: HeaderNode = {"title": title, "level": level, "children": []}

                # find correct parent
                while stack and stack[-1]["level"] >= level:
                    stack.pop()

                stack[-1]["children"].append(node)
                stack.append(node)

            i += 1

        return root["children"]
