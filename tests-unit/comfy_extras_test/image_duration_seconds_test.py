import importlib.util
from pathlib import Path

import pytest
import torch


module_path = Path(__file__).parents[2] / "custom_nodes" / "image_duration_seconds.py"
spec = importlib.util.spec_from_file_location("image_duration_seconds", module_path)
image_duration_seconds = importlib.util.module_from_spec(spec)
spec.loader.exec_module(image_duration_seconds)


def test_resolution_duration_examples():
    assert image_duration_seconds.get_duration_seconds(256, 256) == 30
    assert image_duration_seconds.get_duration_seconds(512, 512) == 27
    assert image_duration_seconds.get_duration_seconds(640, 480) == 27
    assert image_duration_seconds.get_duration_seconds(1280, 720) == 18
    assert image_duration_seconds.get_duration_seconds(1920, 1080) == 3


def test_rejects_too_small_images():
    with pytest.raises(ValueError, match="too small"):
        image_duration_seconds.get_duration_seconds(255, 255)


def test_rejects_larger_than_full_hd():
    with pytest.raises(ValueError, match="Full HD"):
        image_duration_seconds.get_duration_seconds(1921, 1080)


def test_resolution_node_outputs_int_and_float_seconds():
    node = image_duration_seconds.ResolutionDurationSeconds()

    assert node.calculate(512, 512) == (27, 27.0)


def test_image_node_outputs_size_and_duration():
    image = torch.zeros((1, 480, 640, 3), dtype=torch.float32)
    node = image_duration_seconds.ImageDurationSeconds()

    assert node.calculate(image) == (27, 27.0, 640, 480)
