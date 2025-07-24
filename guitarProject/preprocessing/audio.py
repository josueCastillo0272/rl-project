import torch
import torchaudio
from torchaudio.transforms import Resample


def load_wav_to_tensor(filepath: str, sample_rate: int = 16000) -> torch.Tensor:
    """
    Loads a WAV file and returns a mono torch tensor at the specified sample rate.
    Args:
        filepath (str): Path to the WAV file.
        sample_rate (int): Desired sample rate for the output tensor.
    Returns:
        torch.Tensor: Audio tensor of shape (num_samples,)
    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If loading or resampling fails.
    """
    try:
        waveform, orig_sr = torchaudio.load(filepath)  # waveform: (channels, samples)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {filepath}") from e
    except Exception as e:
        raise RuntimeError(f"Error loading audio file: {e}") from e

    # Convert to mono if not already
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample if needed
    if orig_sr != sample_rate:
        try:
            resampler = Resample(orig_sr, sample_rate)
            waveform = resampler(waveform)
        except Exception as e:
            raise RuntimeError(f"Error resampling audio: {e}") from e

    return waveform.squeeze()  # shape: (num_samples,)
