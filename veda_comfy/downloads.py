"""Fetching released predictor bundles into models/veda.

Plain urllib (no huggingface_hub dependency), resumable, sha256-verified,
honouring HF_ENDPOINT (mirrors such as https://hf-mirror.com) and HF_TOKEN.
Errors say where to download the file by hand and where to put it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import urllib.error
import urllib.request
from collections.abc import Callable

_CHUNK = 1 << 20


@dataclasses.dataclass(frozen=True)
class KnownPredictor:
    """A released bundle, pinned to an exact revision and hash."""

    filename: str
    repo: str
    revision: str
    sha256: str
    size: int

    def url(self, endpoint: str) -> str:
        return (f'{endpoint.rstrip("/")}/{self.repo}/resolve/{self.revision}/'
                f'{self.filename}')

    @property
    def page(self) -> str:
        return f'https://huggingface.co/{self.repo}'


KNOWN_PREDICTORS = {
    p.filename: p for p in [
        KnownPredictor(
            filename='minimax_h3_t2va_veda_8nfe_600step_preview_fp8'
                     '.safetensors',
            repo='Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview',
            revision='9a1fd3a41b4a754a7886e64e82edbddf599fd1bd',
            sha256='2a8d8845c5342756a2781e8e69563940e4bb573c9a40ebb534915ff8fd'
                   '76573a',
            size=275415648),
    ]
}

DEFAULT_PREDICTOR = next(iter(KNOWN_PREDICTORS))


class DownloadError(RuntimeError):
    """Download failed; the message tells the user what to do by hand."""


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(_CHUNK), b''):
            digest.update(block)
    return digest.hexdigest()


def fetch(known: KnownPredictor, folder: str,
          progress: Callable[[int, int], None] | None = None) -> str:
    """Downloads a known predictor into folder (resuming a partial file).

    Args:
        known: Which bundle.
        folder: Destination directory (created if missing).
        progress: Called with (bytes done, bytes total).

    Returns:
        The path of the verified file.

    Raises:
        DownloadError: On network or integrity failure.
    """
    os.makedirs(folder, exist_ok=True)
    final = os.path.join(folder, known.filename)
    partial = final + '.part'
    endpoint = os.environ.get('HF_ENDPOINT', 'https://huggingface.co')
    url = known.url(endpoint)
    manual = (f'Download it by hand from {known.page} and put '
              f'{known.filename} into {folder}. Behind a firewall, set '
              'HF_ENDPOINT (e.g. https://hf-mirror.com) before starting '
              'ComfyUI.')
    done = os.path.getsize(partial) if os.path.exists(partial) else 0
    if done > known.size:
        os.remove(partial)
        done = 0
    headers = {'User-Agent': 'Veda-on-ComfyUI'}
    if os.environ.get('HF_TOKEN'):
        headers['Authorization'] = f'Bearer {os.environ["HF_TOKEN"]}'
    if 0 < done < known.size:
        headers['Range'] = f'bytes={done}-'
    try:
        if done < known.size:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=60) as response:
                if done and response.status != 206:
                    done = 0  # server ignored the range: start over
                with open(partial, 'ab' if done else 'wb') as f:
                    while True:
                        block = response.read(_CHUNK)
                        if not block:
                            break
                        f.write(block)
                        done += len(block)
                        if progress is not None:
                            progress(done, known.size)
    except (urllib.error.URLError, OSError, TimeoutError) as error:
        raise DownloadError(f'Could not download {known.filename} from '
                            f'{endpoint} ({error}). {manual}') from error
    if os.path.getsize(partial) != known.size or _sha256(partial) != (
            known.sha256):
        os.remove(partial)
        raise DownloadError(f'{known.filename} arrived corrupted (size or '
                            f'sha256 mismatch); try again. {manual}')
    os.replace(partial, final)
    return final
