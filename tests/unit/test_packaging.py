"""The package metadata the Comfy Registry reads, and its invariants.

These are cheap to check and expensive to get wrong: the node id cannot be
changed after the first publish, a bare-string `license` is rejected, and
a dependency without an environment marker fails the whole node's install
under ComfyUI Manager rather than being skipped. See
docs/features/packaging_release.md.
"""

from __future__ import annotations

import os
import re

import pytest

from conftest import ROOT
import veda_comfy

tomllib = pytest.importorskip('tomllib', reason='needs Python 3.11+')

# Registry rule: under 100 characters, alphanumerics and - _ . only, no
# consecutive special characters, must not start with a digit or special.
_NAME = re.compile(r'^[A-Za-z][A-Za-z0-9]*([-_.][A-Za-z0-9]+)*$')
_SEMVER = re.compile(r'^\d+\.\d+\.\d+$')


@pytest.fixture(scope='module')
def pyproject():
    with open(os.path.join(ROOT, 'pyproject.toml'), 'rb') as f:
        return tomllib.load(f)


def test_version_matches_the_package(pyproject):
    """AGENTS.md has the release bump both; nothing else caught drift."""
    version = pyproject['project']['version']
    assert _SEMVER.fullmatch(version), 'the registry requires X.Y.Z'
    assert version == veda_comfy.__version__


def test_node_id_satisfies_the_registry(pyproject):
    name = pyproject['project']['name']
    assert len(name) < 100
    assert _NAME.fullmatch(name), name
    assert 'comfyui' not in name.lower()  # the registry advises against it


def test_license_is_a_table(pyproject):
    """A bare string such as license = "MIT" is rejected on publish."""
    license_ = pyproject['project']['license']
    assert isinstance(license_, dict)
    assert set(license_) <= {'file', 'text'} and license_
    if 'file' in license_:
        assert os.path.isfile(os.path.join(ROOT, license_['file']))


def test_comfy_section_is_complete(pyproject):
    comfy = pyproject['tool']['comfy']
    assert comfy['PublisherId']
    assert comfy['DisplayName']
    assert comfy['Icon'].startswith('https://')
    assert comfy['requires-comfyui']


def _normalise(requirement: str) -> str:
    return requirement.replace('"', "'").replace(' ', '')


def test_every_dependency_carries_a_marker(pyproject):
    """Manager installs requirements as a batch: one unbuildable wheel
    fails the node, so nothing may be unconditional."""
    for requirement in pyproject['project']['dependencies']:
        assert ';' in requirement, f'{requirement} has no marker'
    assert not any('torch' in r for r in pyproject['project']['dependencies'])


def test_requirements_txt_agrees_with_pyproject(pyproject):
    """Two files, one list: clones install from requirements.txt."""
    with open(os.path.join(ROOT, 'requirements.txt'), encoding='utf-8') as f:
        pinned = [line.strip() for line in f
                  if line.strip() and not line.startswith('#')]
    assert ([_normalise(r) for r in pinned]
            == [_normalise(r) for r in pyproject['project']['dependencies']])
