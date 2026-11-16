import importlib.util
from pathlib import Path

import pytest
from PIL import Image


module_path = Path(__file__).parents[2] / "custom_nodes" / "ComfyUI-MultiImageLoader" / "__init__.py"
spec = importlib.util.spec_from_file_location("multi_image_loader", module_path)
multi_image_loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(multi_image_loader)


def write_image(path, color):
    Image.new("RGB", (8, 6), color).save(path)


def test_blank_selection_loads_all_images(tmp_path):
    write_image(tmp_path / "b.png", "blue")
    write_image(tmp_path / "a.jpg", "red")
    (tmp_path / "notes.txt").write_text("not an image")

    node = multi_image_loader.LoadMultipleImagesFromFolder()
    images, masks, filenames, indexes = node.load_images(str(tmp_path), "", False)

    assert filenames == ["a.jpg", "b.png"]
    assert indexes == [0, 1]
    assert len(images) == 2
    assert len(masks) == 2
    assert images[0].shape == (1, 6, 8, 3)


def test_selected_images_preserve_order(tmp_path):
    write_image(tmp_path / "first.png", "red")
    write_image(tmp_path / "second.png", "blue")

    node = multi_image_loader.LoadMultipleImagesFromFolder()
    _, _, filenames, indexes = node.load_images(str(tmp_path), "second.png\nfirst.png", False)

    assert filenames == ["second.png", "first.png"]
    assert indexes == [0, 1]


def test_selected_images_accepts_multiselect_list(tmp_path):
    write_image(tmp_path / "first.png", "red")
    write_image(tmp_path / "second.png", "blue")

    node = multi_image_loader.LoadMultipleImagesFromFolder()
    _, _, filenames, indexes = node.load_images(str(tmp_path), ["second.png", "first.png"], False)

    assert filenames == ["second.png", "first.png"]
    assert indexes == [0, 1]


def test_selected_images_accepts_escaped_list_string(tmp_path):
    upload_dir = tmp_path / "multi_image_loader" / "1784718222992-selection"
    upload_dir.mkdir(parents=True)
    write_image(upload_dir / "cutout_rect_19.png", "red")

    escaped = "[&amp;amp;apos;multi_image_loader&amp;amp;#x2F;1784718222992-selection&amp;amp;#x2F;cutout_rect_19.png&amp;amp;apos;]"
    node = multi_image_loader.LoadMultipleImagesFromFolder()
    _, _, filenames, indexes = node.load_images(str(tmp_path), escaped, False)

    assert filenames == ["multi_image_loader/1784718222992-selection/cutout_rect_19.png"]
    assert indexes == [0]


def test_recursive_blank_selection_includes_subfolders(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    write_image(nested / "inside.png", "green")

    node = multi_image_loader.LoadMultipleImagesFromFolder()
    _, _, filenames, _ = node.load_images(str(tmp_path), "", True)

    assert filenames == ["nested/inside.png"]


def test_rejects_paths_outside_folder(tmp_path):
    outside = tmp_path.parent / "outside.png"
    write_image(outside, "red")

    result = multi_image_loader.LoadMultipleImagesFromFolder.VALIDATE_INPUTS(
        str(tmp_path),
        "../outside.png",
        False,
    )

    assert result != True
    assert "outside" in result


def test_validation_accepts_existing_image(tmp_path):
    write_image(tmp_path / "image.png", "red")

    assert multi_image_loader.LoadMultipleImagesFromFolder.VALIDATE_INPUTS(
        str(tmp_path),
        "image.png",
        False,
    ) is True


def test_input_uses_multiline_image_list(tmp_path, monkeypatch):
    monkeypatch.setattr(multi_image_loader.folder_paths, "get_input_directory", lambda: str(tmp_path))

    selected_images = multi_image_loader.LoadMultipleImagesFromFolder.INPUT_TYPES()["required"]["selected_images"]

    assert selected_images[0] == "STRING"
    assert selected_images[1]["multiline"] is True
