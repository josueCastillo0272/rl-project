import torch
import torchaudio
from torchaudio.transforms import Resample
import torch.nn.functional as F

def load_wav_to_tensor(filepath: str, sample_rate: int = 16000) -> torch.Tensor:
    """
    Loads a WAV file, normalizes, and returns a mono torch tensor at the specified sample rate.
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


def pad_or_crop(waveform: torch.Tensor, target_len: int, random_crop: bool = True) -> torch.Tensor:
    """
    Ensures waveform length is equal to the target length by random or centered cropping, or zero-padding at the end.
    Args:
        waveform (torch.Tensor): 1-D audio tensor.
        target_len (int): Desired number of samples.
        random_crop (bool): If True and waveform is longer, crop at a random start index; otherwise, crop from beginning.
    Returns:
        torch.Tensor: Tensor of exactly 'target_len' samples
    """
    cur_len = waveform.shape[-1]

    # Cutting
    if cur_len > target_len:
        if random_crop:
            start = torch.randint(0, cur_len - target_len + 1, ()).item()
        else:
            start = (cur_len - target_len) // 2
        waveform = waveform[..., start: start+target_len]

    # Padding
    elif cur_len < target_len:
        pad_amt = target_len - cur_len
        waveform = F.pad(waveform, (0, pad_amt))
    return waveform

def normalize_amplitude(waveform: torch.Tensor, target_rms: float = 0.1, eps: float = 1e-9) -> torch.Tensor:
    """
    Applies RMS normalization followed by peak normalization.
    Args:
        waveform (torch.Tensor): 1-D audio tensor.
        target_rms (float): desired root-mean-square level.
        eps (float): small constant to avoid div-by-zero.

    Returns:
        torch.Tensor: A new tensor with normalized amplitude
    """
    # RMS normalization
    rms = waveform.pow(2).mean().sqrt()
    if rms > eps:
        waveform = waveform * (target_rms/rms)

    # Peak normalization
    peak = waveform.abs().max()
    if peak > 1.0:
        waveform /= peak

    return waveform