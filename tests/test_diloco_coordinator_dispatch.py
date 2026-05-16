from __future__ import annotations

import argparse
import base64
import json
import os
import zlib
from pathlib import Path

from ouroboros.coordinator import dispatch


def _decode_payload(payload: str) -> dict[str, str]:
    return json.loads(zlib.decompress(base64.b64decode(payload)).decode("utf-8"))


def _minimal_notebook(cells):
    return {
        "cells": cells,
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def test_build_kaggle_kernel_metadata_preserves_gpu_and_internet_contract():
    metadata = dispatch._build_kaggle_kernel_metadata(
        slug="weirdrunner/kaggle-utils",
        notebook_filename="kaggle-utils.ipynb",
    )

    assert metadata["id"] == "weirdrunner/kaggle-utils"
    assert metadata["title"] == "kaggle-utils"
    assert metadata["code_file"] == "kaggle-utils.ipynb"
    assert metadata["kernel_type"] == "notebook"
    assert metadata["enable_gpu"] is True
    assert metadata["accelerator"] == "NvidiaTeslaT4"
    assert metadata["enable_internet"] is True
