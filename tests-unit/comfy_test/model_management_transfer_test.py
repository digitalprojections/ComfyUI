import torch

import comfy.model_management as model_management
from comfy.cli_args import args


def test_windows_cuda_non_blocking_disabled_by_default(monkeypatch):
    monkeypatch.setattr(model_management, "WINDOWS", True)
    monkeypatch.setattr(args, "force_non_blocking", False)
    monkeypatch.setattr(args, "deterministic", False)
    monkeypatch.setattr(model_management, "directml_enabled", False)
    monkeypatch.setattr(model_management, "is_intel_xpu", lambda: False)

    assert not model_management.device_supports_non_blocking(torch.device("cuda"))


def test_force_non_blocking_overrides_windows_cuda_guard(monkeypatch):
    monkeypatch.setattr(model_management, "WINDOWS", True)
    monkeypatch.setattr(args, "force_non_blocking", True)

    assert model_management.device_supports_non_blocking(torch.device("cuda"))


def test_default_async_offload_disabled_for_windows_nvidia(monkeypatch):
    monkeypatch.setattr(model_management, "WINDOWS", True)
    monkeypatch.setattr(model_management, "is_nvidia", lambda: True)
    monkeypatch.setattr(model_management, "is_amd", lambda: False)

    assert not model_management.default_async_offload_enabled()


def test_default_async_offload_kept_for_non_windows_nvidia(monkeypatch):
    monkeypatch.setattr(model_management, "WINDOWS", False)
    monkeypatch.setattr(model_management, "is_nvidia", lambda: True)
    monkeypatch.setattr(model_management, "is_amd", lambda: False)

    assert model_management.default_async_offload_enabled()


def test_windows_nvidia_vae_defaults_to_torch_device(monkeypatch):
    cuda_device = torch.device("cuda")
    monkeypatch.setattr(model_management, "WINDOWS", True)
    monkeypatch.setattr(model_management, "is_nvidia", lambda: True)
    monkeypatch.setattr(model_management, "get_torch_device", lambda: cuda_device)
    monkeypatch.setattr(args, "cpu_vae", False)
    monkeypatch.setattr(args, "gpu_only", False)

    assert model_management.vae_device() == cuda_device


def test_windows_nvidia_text_encoder_defaults_to_torch_device(monkeypatch):
    cuda_device = torch.device("cuda")
    monkeypatch.setattr(model_management, "WINDOWS", True)
    monkeypatch.setattr(model_management, "is_nvidia", lambda: True)
    monkeypatch.setattr(model_management, "get_torch_device", lambda: cuda_device)
    monkeypatch.setattr(model_management, "should_use_fp16", lambda prioritize_performance=False, device=None: True)
    monkeypatch.setattr(model_management, "vram_state", model_management.VRAMState.NORMAL_VRAM)
    monkeypatch.setattr(model_management.comfy.memory_management, "aimdo_enabled", False)
    monkeypatch.setattr(args, "gpu_only", False)

    assert model_management.text_encoder_device() == cuda_device


def test_gpu_only_keeps_vae_on_torch_device(monkeypatch):
    cuda_device = torch.device("cuda")
    monkeypatch.setattr(model_management, "WINDOWS", True)
    monkeypatch.setattr(model_management, "is_nvidia", lambda: True)
    monkeypatch.setattr(model_management, "get_torch_device", lambda: cuda_device)
    monkeypatch.setattr(args, "cpu_vae", False)
    monkeypatch.setattr(args, "gpu_only", True)

    assert model_management.vae_device() == cuda_device


def test_gpu_only_keeps_text_encoder_on_torch_device(monkeypatch):
    cuda_device = torch.device("cuda")
    monkeypatch.setattr(model_management, "WINDOWS", True)
    monkeypatch.setattr(model_management, "is_nvidia", lambda: True)
    monkeypatch.setattr(model_management, "get_torch_device", lambda: cuda_device)
    monkeypatch.setattr(args, "gpu_only", True)

    assert model_management.text_encoder_device() == cuda_device


def test_gpu_only_keeps_offload_devices_on_torch_device(monkeypatch):
    cuda_device = torch.device("cuda")
    monkeypatch.setattr(model_management, "get_torch_device", lambda: cuda_device)
    monkeypatch.setattr(model_management, "vram_state", model_management.VRAMState.HIGH_VRAM)
    monkeypatch.setattr(args, "gpu_only", True)

    assert model_management.unet_offload_device() == cuda_device
    assert model_management.vae_offload_device() == cuda_device
    assert model_management.text_encoder_offload_device() == cuda_device
