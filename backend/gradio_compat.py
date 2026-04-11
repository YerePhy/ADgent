"""Compatibility shims for Gradio bugs.

NiftiFile
---------
gr.File validates uploads with Path.suffix (last extension only), so compound
extensions like .nii.gz are never matched. NiftiFile subclasses gr.File and
overrides _process_single_file to use all suffixes joined.
"""

import tempfile
from pathlib import Path

from gradio.components import File
from gradio.data_classes import FileData
from gradio.exceptions import Error
from gradio.utils import NamedString


def _is_valid_file(file_path: str, file_types: list[str]) -> bool:
    full_ext = "".join(Path(file_path).suffixes).lower()
    for ft in file_types:
        if ft.startswith(".") and ft.lower() == full_ext:
            return True
    return False


class NiftiFile(File):
    """gr.File with compound-extension support (.nii.gz)."""

    @classmethod
    def get_block_name(cls) -> str:
        return "file"

    def _process_single_file(self, f: FileData) -> NamedString | bytes:
        file_name = f.path
        if self.type == "filepath":
            if self.file_types and not _is_valid_file(file_name, self.file_types):
                raise Error(
                    f"Invalid file type. Please upload one of: {self.file_types}"
                )
            file = tempfile.NamedTemporaryFile(delete=False, dir=self.GRADIO_CACHE)
            file.name = file_name
            return NamedString(file_name)
        elif self.type == "binary":
            with open(file_name, "rb") as fh:
                return fh.read()
        else:
            raise ValueError(f"Unknown type: {self.type!r}. Choose 'filepath' or 'binary'.")
