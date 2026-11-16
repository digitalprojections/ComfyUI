import pytest
import torch

from comfy_extras.nodes_audio import AudioDuration


class TestAudioDuration:
    def test_returns_seconds_samples_and_sample_rate(self):
        audio = {
            "waveform": torch.zeros((1, 2, 22050), dtype=torch.float32),
            "sample_rate": 44100,
        }

        result = AudioDuration.execute(audio)

        assert result[0] == 0.5
        assert result[1] == 22050
        assert result[2] == 44100

    def test_empty_audio_returns_zero_seconds(self):
        audio = {
            "waveform": torch.zeros((1, 2, 0), dtype=torch.float32),
            "sample_rate": 44100,
        }

        result = AudioDuration.execute(audio)

        assert result[0] == 0.0
        assert result[1] == 0
        assert result[2] == 44100

    def test_none_audio_raises_clear_error(self):
        with pytest.raises(ValueError, match="input audio is None"):
            AudioDuration.execute(None)

    def test_invalid_sample_rate_raises_clear_error(self):
        audio = {
            "waveform": torch.zeros((1, 2, 100), dtype=torch.float32),
            "sample_rate": 0,
        }

        with pytest.raises(ValueError, match="sample_rate must be positive"):
            AudioDuration.execute(audio)
