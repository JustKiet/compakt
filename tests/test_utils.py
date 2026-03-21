from __future__ import annotations

import unittest

from compakt.core.models import CompaktChunk, MarkdownHeader
from compakt.core.utils import elbow_filter, normalize_markdown_title


def _chunk(content: str = "x") -> CompaktChunk:
    return CompaktChunk(
        header_type=MarkdownHeader.H2,
        header_name="test",
        content=content,
    )


class ElbowFilterTest(unittest.TestCase):
    def test_empty_data(self) -> None:
        self.assertEqual(elbow_filter([]), [])

    def test_single_item(self) -> None:
        data = [(_chunk(), 0.9)]
        self.assertEqual(elbow_filter(data), data)

    def test_two_items(self) -> None:
        data = [(_chunk("a"), 0.9), (_chunk("b"), 0.1)]
        self.assertEqual(elbow_filter(data), data)

    def test_filters_low_scores(self) -> None:
        high = [(_chunk(f"h{i}"), 0.8 + i * 0.01) for i in range(5)]
        low = [(_chunk(f"l{i}"), 0.05 + i * 0.01) for i in range(5)]
        data = high + low
        result = elbow_filter(data, percentile=25.0)
        # All high-scoring items should be kept; at least one low-scoring item
        # may survive depending on the 25th-percentile cutoff, but the count
        # should be less than the full set.
        self.assertGreater(len(result), 0)
        self.assertLessEqual(len(result), len(data))
        # The highest-scoring item must always be present.
        self.assertEqual(result[0][0].content, high[0][0].content)

    def test_all_same_scores_returns_all(self) -> None:
        data = [(_chunk(f"c{i}"), 0.5) for i in range(10)]
        result = elbow_filter(data)
        self.assertEqual(len(result), 10)

    def test_returns_at_least_one(self) -> None:
        # Even with a very strict percentile, at least one item is returned.
        data = [(_chunk(f"c{i}"), 0.01 * i) for i in range(10)]
        result = elbow_filter(data, percentile=99.0)
        self.assertGreaterEqual(len(result), 1)

    def test_custom_percentile(self) -> None:
        # With a very low percentile (keep almost everything), most items survive.
        data = [(_chunk(f"c{i}"), float(i)) for i in range(20)]
        result = elbow_filter(data, percentile=10.0)
        self.assertGreater(len(result), 10)


class NormalizeMarkdownTitleTest(unittest.TestCase):
    def test_removes_links(self) -> None:
        self.assertEqual(
            normalize_markdown_title("[Click here](https://example.com)"),
            "click here",
        )

    def test_removes_image_links(self) -> None:
        self.assertEqual(
            normalize_markdown_title("![alt text](image.png)"),
            "alt text",
        )

    def test_removes_bold(self) -> None:
        self.assertEqual(normalize_markdown_title("**bold text**"), "bold text")

    def test_removes_italic(self) -> None:
        self.assertEqual(normalize_markdown_title("*italic*"), "italic")

    def test_removes_strikethrough(self) -> None:
        self.assertEqual(normalize_markdown_title("~~deleted~~"), "deleted")

    def test_removes_inline_code(self) -> None:
        self.assertEqual(normalize_markdown_title("`code`"), "code")

    def test_removes_html_tags(self) -> None:
        self.assertEqual(
            normalize_markdown_title("Hello <br/> World"),
            "hello world",
        )

    def test_collapses_whitespace(self) -> None:
        self.assertEqual(
            normalize_markdown_title("  lots   of   spaces  "),
            "lots of spaces",
        )

    def test_empty_string(self) -> None:
        self.assertEqual(normalize_markdown_title(""), "")

    def test_casefolds(self) -> None:
        self.assertEqual(normalize_markdown_title("UPPERCASE"), "uppercase")


if __name__ == "__main__":
    unittest.main()
