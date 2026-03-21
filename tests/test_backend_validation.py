from __future__ import annotations

import unittest

from backend.validation import validate_blob_name, validate_job_id


class ValidateBlobNameTest(unittest.TestCase):
    def test_valid_name(self) -> None:
        self.assertEqual(validate_blob_name("my-file.pdf"), "my-file.pdf")

    def test_valid_name_with_path(self) -> None:
        self.assertEqual(validate_blob_name("folder/file.pdf"), "folder/file.pdf")

    def test_empty_raises(self) -> None:
        with self.assertRaises(ValueError):
            validate_blob_name("")

    def test_path_traversal_raises(self) -> None:
        with self.assertRaises(ValueError):
            validate_blob_name("../../../etc/passwd")

    def test_leading_slash_raises(self) -> None:
        with self.assertRaises(ValueError):
            validate_blob_name("/absolute/path")

    def test_null_byte_raises(self) -> None:
        with self.assertRaises(ValueError):
            validate_blob_name("file\x00.pdf")

    def test_too_long_raises(self) -> None:
        with self.assertRaises(ValueError):
            validate_blob_name("a" * 201)

    def test_special_chars_raises(self) -> None:
        with self.assertRaises(ValueError):
            validate_blob_name("file;rm -rf /")


class ValidateJobIdTest(unittest.TestCase):
    def test_valid_uuid_style(self) -> None:
        self.assertEqual(
            validate_job_id("abc-123-def"), "abc-123-def"
        )

    def test_empty_raises(self) -> None:
        with self.assertRaises(ValueError):
            validate_job_id("")

    def test_path_traversal_raises(self) -> None:
        with self.assertRaises(ValueError):
            validate_job_id("../admin")

    def test_slash_raises(self) -> None:
        with self.assertRaises(ValueError):
            validate_job_id("path/id")

    def test_too_long_raises(self) -> None:
        with self.assertRaises(ValueError):
            validate_job_id("a" * 65)


if __name__ == "__main__":
    unittest.main()
