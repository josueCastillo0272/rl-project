import torch
import torchaudio

from pathlib import Path

def tensor_to_wav(tensor: torch.Tensor, out_path: str | Path, sample_rate: int = 16000) -> None:
    """
    Converts 1D mono tensor into a .wav file.
    Args: 
        tensor (torch.Tensor): Tensor of shape (num_samples,)
        sample_rate (int): Sampling rate in Hz
        out_path (str | Path): Destination filepath for the .wav file
    """
    out_path = Path(out_path)
    if out_path.suffix.lower() != ".wav":
        raise ValueError(f"Output path must end with .wav, got {out_path.suffix}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tensor = tensor.detach().cpu().to(torch.float32)

    waveform = tensor.unsqueeze(0) 
    print(str(out_path))
    try:
        torchaudio.save(str(out_path), waveform.cpu(), sample_rate)
    except Exception as e:
        raise RuntimeError(f"Failed to save WAV to {out_path}: {e}") from e