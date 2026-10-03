import dataclasses
import functools
import hashlib
import http.server
import os
import socketserver
import threading

import pytest

from veda_comfy import downloads


class _Server(http.server.ThreadingHTTPServer):
    def server_bind(self):
        # HTTPServer.server_bind resolves the host's FQDN, which can take
        # tens of seconds on machines with slow reverse DNS.
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]


@pytest.fixture
def server(tmp_path):
    root = tmp_path / 'hub'
    target = root / 'org' / 'repo' / 'resolve' / 'rev'
    target.mkdir(parents=True)
    payload = os.urandom(3 * 2**20 + 17)
    (target / 'p.safetensors').write_bytes(payload)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler,
                                directory=str(root))
    httpd = _Server(('127.0.0.1', 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f'http://127.0.0.1:{httpd.server_port}', payload
    httpd.shutdown()


def _known(payload, **kwargs):
    fields = dict(filename='p.safetensors', repo='org/repo', revision='rev',
                  sha256=hashlib.sha256(payload).hexdigest(),
                  size=len(payload))
    fields.update(kwargs)
    return downloads.KnownPredictor(**fields)


def test_fetch_verifies_and_resumes(server, tmp_path, monkeypatch):
    endpoint, payload = server
    monkeypatch.setenv('HF_ENDPOINT', endpoint)
    folder = tmp_path / 'models'
    folder.mkdir()
    # a stale partial file from an interrupted run
    (folder / 'p.safetensors.part').write_bytes(payload[:1000])
    seen = []
    path = downloads.fetch(_known(payload), str(folder),
                           lambda done, total: seen.append((done, total)))
    assert open(path, 'rb').read() == payload
    assert seen[-1] == (len(payload), len(payload))
    assert not (folder / 'p.safetensors.part').exists()


def test_fetch_rejects_corruption(server, tmp_path, monkeypatch):
    endpoint, payload = server
    monkeypatch.setenv('HF_ENDPOINT', endpoint)
    with pytest.raises(downloads.DownloadError, match='corrupted'):
        downloads.fetch(_known(payload, sha256='0' * 64), str(tmp_path))
    assert not (tmp_path / 'p.safetensors.part').exists()


def test_fetch_error_says_what_to_do(tmp_path, monkeypatch):
    monkeypatch.setenv('HF_ENDPOINT', 'http://127.0.0.1:9')
    known = dataclasses.replace(downloads.KNOWN_PREDICTORS[
        downloads.DEFAULT_PREDICTOR])
    with pytest.raises(downloads.DownloadError, match='HF_ENDPOINT'):
        downloads.fetch(known, str(tmp_path))


@pytest.mark.parametrize('endpoint,sent', [
    ('https://huggingface.co', True),
    ('https://cdn-lfs.huggingface.co', True),
    ('https://hf-mirror.com', False),
    ('https://hf-mirror.com/huggingface.co/x', False),
    ('https://huggingface.co.example.com', False),
    ('https://nothuggingface.co', False),
])
def test_token_only_goes_to_hugging_face(endpoint, sent):
    """HF_ENDPOINT names a mirror, and the README tells users behind a
    firewall to set it, so the token must not follow it there."""
    assert downloads._is_huggingface(endpoint) is sent


def test_fetch_withholds_the_token_from_a_mirror(server, tmp_path,
                                                 monkeypatch):
    endpoint, payload = server
    seen = {}
    real = downloads.urllib.request.Request

    def spy(url, headers=None, **kwargs):
        seen.update(headers or {})
        return real(url, headers=headers or {}, **kwargs)

    monkeypatch.setattr(downloads.urllib.request, 'Request', spy)
    monkeypatch.setenv('HF_TOKEN', 'secret-token')
    monkeypatch.setenv('HF_ENDPOINT', endpoint)  # a 127.0.0.1 "mirror"
    downloads.fetch(_known(payload), str(tmp_path / 'out'))
    assert 'Authorization' not in seen
    assert 'secret-token' not in repr(seen)
