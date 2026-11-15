"""Tests for the DTW speech aligner."""

import re
from argparse import Namespace

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import soundfile as sf

import main

SR = 22050


def tones(freqs, dur=0.25):
    """Concatenate Hann-windowed sine tones.

    Args:
        freqs (list[float]): Tone frequencies in Hz.
        dur (float): Duration of each tone in seconds.

    Returns:
        np.ndarray: Audio samples.
    """
    t = np.arange(int(SR * dur)) / SR
    return np.concatenate([np.sin(2 * np.pi * f * t) * np.hanning(len(t)) for f in freqs])


@pytest.fixture
def audio(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    query = tones([300, 500, 800, 450, 650, 350])
    pre = tones([1200, 1500, 1800, 2100])
    post = tones([2500, 1100, 1900])
    reference = np.concatenate([pre, query, post])
    reference = reference + 0.001 * rng.normal(size=len(reference))
    sf.write(tmp_path / "q.wav", query, SR)
    sf.write(tmp_path / "r.wav", reference, SR)
    monkeypatch.chdir(tmp_path)
    return tmp_path, len(pre) / SR, (len(pre) + len(query)) / SR


@pytest.mark.parametrize("save_plot", [False, True])
def test_alignment_finds_embedded_query(audio, capsys, save_plot):
    path, begin, end = audio
    main.main(
        Namespace(
            query_path=str(path / "q.wav"),
            reference_path=str(path / "r.wav"),
            feat_types=["mfcc"],
            save_plot=save_plot,
        )
    )
    out = capsys.readouterr().out
    got_begin = float(re.search(r"Begin time: ([\d.]+)s", out).group(1))
    got_end = float(re.search(r"End time: ([\d.]+)s", out).group(1))
    hop = 1024 // 4 / SR
    # Within ~1.5 frames of the true boundaries.
    assert abs(got_begin - begin) <= 1.5 * hop
    assert abs(got_end - end) <= 1.5 * hop
    assert (path / "clip.wav").exists()


def test_pca_transform_rejects_mismatched_feature_dims():
    @main.pca_transform
    def identity(query, reference, label=None):
        return query, reference

    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        identity(query=rng.normal(size=(10, 13)), reference=rng.normal(size=(12, 12)), label="mfcc")

    # Different lengths with the same feature dimension are fine.
    identity(query=rng.normal(size=(10, 13)), reference=rng.normal(size=(12, 13)), label="mfcc")


def test_f0_plot_does_not_mutate_input():
    y = tones([220, 330])
    f0 = np.array([0.0, 220.0, 0.0, 330.0])
    main.f0_mel_spec_plot(y=y, f0=f0, sr=SR, n_fft=1024, hop_length=256)
    assert np.array_equal(f0, [0.0, 220.0, 0.0, 330.0])


def test_zscore_of_constant_feature_is_finite():
    assert np.array_equal(main.zscore(np.zeros((1, 5))), np.zeros((1, 5)))


def test_unvoiced_f0_does_not_break_alignment(tmp_path, monkeypatch, capsys):
    # White noise has no pitch, so F0 is all zeros.
    rng = np.random.default_rng(1)
    sf.write(tmp_path / "q.wav", 0.1 * rng.normal(size=SR), SR)
    sf.write(tmp_path / "r.wav", 0.1 * rng.normal(size=2 * SR), SR)
    monkeypatch.chdir(tmp_path)

    main.main(
        Namespace(
            query_path=str(tmp_path / "q.wav"),
            reference_path=str(tmp_path / "r.wav"),
            feat_types=["f0", "mfcc"],
            save_plot=False,
        )
    )

    assert "Begin time" in capsys.readouterr().out
