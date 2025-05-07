import argparse
import copy

from argparse import Namespace
from collections import OrderedDict
from typing import Any

import dtw
import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import wrapt

from dtw import DTW
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from scipy import stats
from sklearn.decomposition import PCA

FEAT_WEIGHTS = {
    "f0": 0.3,
    "f0_delta": 0.15,
    "f0_delta_delta": 0.075,
    "mfcc": 1.0,
    "mfcc_delta": 0.75,
    "mfcc_delta_delta": 0.5,
}


def main(args: Namespace):
    args.feat_types = tuple([feat_type.lower() for feat_type in args.feat_types])

    query_audio, sr = librosa.load(args.query_path)
    reference_audio, sr = librosa.load(args.reference_path)

    n_fft = 1024
    hop_length = n_fft // 4
    feat_types = {feat_type for feat_type in FEAT_WEIGHTS.keys() if feat_type.startswith(args.feat_types)}
    query_feats = feat_extract(
        query_audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        feat_types=feat_types,
    )

    reference_feats = feat_extract(
        reference_audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        feat_types=feat_types,
    )

    weighted_norm_query_feats = []
    weighted_norm_reference_feats = []

    for feat_type in feat_types:
        weighted_norm_query_feats.append(stats.zscore(query_feats[feat_type], axis=-1))
        weighted_norm_reference_feats.append(
            stats.zscore(reference_feats[feat_type], axis=-1)
        )

        weighted_norm_query_feats[-1] = (
                FEAT_WEIGHTS[feat_type] * weighted_norm_query_feats[-1]
        )
        weighted_norm_reference_feats[-1] = (
                FEAT_WEIGHTS[feat_type] * weighted_norm_reference_feats[-1]
        )

    query = np.vstack(weighted_norm_query_feats).T
    reference = np.vstack(weighted_norm_reference_feats).T

    alignment = dtw.dtw(
        query,
        reference,
        step_pattern=dtw.rabinerJuangStepPattern(6, slope_weighting='c'),
        open_begin=True,
        open_end=True,
        keep_internals=True,
    )
    samples = librosa.frames_to_samples(
        [alignment.index2[0], alignment.index2[-1]],
        hop_length=hop_length,
        n_fft=n_fft,
    )
    begin, end = samples[0], samples[-1]
    sf.write("clip.wav", data=reference_audio[begin: end + 1], samplerate=sr)

    times = librosa.samples_to_time(np.array([begin, end]), sr=sr)
    print(f"Begin time: {times[0]}s")
    print(f"End time: {times[-1]}s")

    if args.save_plot:
        alignment.plot(type="threeway")

        plt.savefig("alignment.png", bbox_inches="tight")

        f0_mel_spec_plot(
            y=query_audio,
            f0=query_feats["f0"] if "f0" in feat_types else None,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            title="pYIN fundamental frequency estimation (query)",
        )

        plt.savefig("f0_mel_spec_query.png", bbox_inches="tight")

        f0_mel_spec_plot(
            y=reference_audio,
            f0=reference_feats["f0"] if "f0" in feat_types else None,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            title="pYIN fundamental frequency estimation (reference)",
        )

        plt.savefig("f0_mel_spec_reference.png", bbox_inches="tight")

        frame_length = hop_length / sr

        for feat_type in feat_types:
            dtw_plot(
                alignment,
                query=query_feats[feat_type].T,
                reference=reference_feats[feat_type].T,
                frame_length=frame_length,
                label=feat_type,
            )

            plt.savefig(f"{feat_type}_dtw.png", bbox_inches="tight")


def feat_extract(
        y: np.ndarray, sr: int, n_fft: int, hop_length: int, feat_types: set[str]
) -> OrderedDict[str, np.ndarray]:
    """
    Speech feature extraction.

    Args:
        y (np.ndarray): audio time series
        sr (int): sample rate
        n_fft (int): length of the FFT window
        hop_length (int): number of samples between successive frames
        feat_types (set[str]):
            supported type: f0, f0_delta, f0_delta2, mfcc, mfcc_delta, mfcc_delta2, htt-mfcc, htt-mfcc_delta, htt-mfcc_delta2

    Returns:
        features (OrderedDict): Feature ordered dict
    """

    supported_feat_types = set(FEAT_WEIGHTS.keys())

    for feat_type in feat_types:
        if feat_type not in supported_feat_types:
            raise ValueError(f"Unsupported feature type: {feat_type}.")

    features = OrderedDict()

    if feat_types & {"f0", "f0_delta", "f0_delta_delta"}:
        f0 = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C5"),
            sr=sr,
            frame_length=n_fft,
            hop_length=hop_length,
            fill_na=0,
        )[0]

        if "f0" in feat_types:
            features["f0"] = f0

        if "f0_delta" in feat_types:
            features["f0_delta"] = librosa.feature.delta(f0)

        if "f0_delta_delta" in feat_types:
            features["f0_delta_delta"] = librosa.feature.delta(f0, order=2)

    if feat_types & {"mfcc", "mfcc_delta", "mfcc_delta_delta"}:
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=13,
            n_fft=n_fft,
            hop_length=hop_length,
        )

        if "mfcc" in feat_types:
            features["mfcc"] = mfcc

        if "mfcc_delta" in feat_types:
            features["mfcc_delta"] = librosa.feature.delta(mfcc)

        if "mfcc_delta_delta" in feat_types:
            features["mfcc_delta_delta"] = librosa.feature.delta(mfcc, order=2)

    return features


def f0_mel_spec_plot(
        y: np.ndarray = None,
        S=None,
        f0: np.ndarray = None,
        sr: int = None,
        n_fft: int = None,
        hop_length: int = None,
        **kwargs,
) -> tuple[Figure, Any]:
    """
    Draw f0 and mel spectrogram.

    Args:
        y (np.ndarray): audio time series
        S (np.ndarray): mel-spectrogram
        f0 (np.ndarray, optional): fundamental frequency
        sr (int, optional): sample rate
        n_fft (int, optional): length of the FFT window
        hop_length (int, optional): number of samples between successive frames

    Returns:
        figure (Figure): Figure
        axes (Any): Axes or array of axes
    """

    if S is None:
        spectrogram = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)) ** 2
        S = librosa.feature.melspectrogram(S=spectrogram, sr=sr)
        S = librosa.power_to_db(S, ref=np.max)

    fig, ax = plt.subplots()
    ax.set(**kwargs)
    img = librosa.display.specshow(
        S,
        x_axis="time",
        y_axis="mel",
        sr=sr,
        ax=ax,
    )
    fig.colorbar(img, ax=ax, format="%+2.f dB")

    if f0 is None:
        f0 = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C5"),
            sr=sr,
            frame_length=n_fft,
            hop_length=hop_length,
        )[0]
    else:
        f0[f0 == 0] = np.nan

    times = librosa.times_like(f0, sr=sr)
    ax.plot(times, f0, label="f0", color="cyan", linewidth=3)
    ax.legend(loc="upper right")
    return fig, ax


@wrapt.decorator
def pca_transform(wrapped, instance, args, kwargs):
    """
    PCA for query and reference.

    Args:
        wrapped: the wrapped function which in turns needs to be called by your wrapper function
        instance: the object to which the wrapped function was bound when it was called
        args: the list of positional arguments supplied when the decorated function was called
        kwargs: the dictionary of keyword arguments supplied when the decorated function was called

    Returns:
        value: wrapped function return value
    """

    if kwargs["query"].shape != kwargs["query"].shape:
        raise ValueError("query and reference shape incompatibility.")

    if kwargs["query"].ndim > 1:
        pca = PCA(n_components=1)
        kwargs["query"] = pca.fit_transform(kwargs["query"]).ravel()

        pca = PCA(n_components=1)
        kwargs["reference"] = pca.fit_transform(kwargs["reference"]).ravel()

        if kwargs["label"] is not None:
            kwargs["label"] = f"PCA_{kwargs["label"]}"

    return wrapped(*args, **kwargs)


@pca_transform
def dtw_plot(
        alignment: DTW,
        query: np.ndarray,
        reference: np.ndarray,
        frame_length: float,
        label: str = None,
) -> Any:
    """
    Draw Dynamic Time Warping (DTW).
    Args:
        alignment (DTW): DTW object
        query (np.ndarray): query time series
        reference (np.ndarray): reference time series
        frame_length (float): frame time in seconds
        label (str, optional): time series label

    Returns:
        ax (Any): Axes or array of axes
    """

    align = copy.deepcopy(alignment)
    shift_direction = None
    shifted_query = query
    query_begin, query_end = align.index1[0], align.index1[-1]
    reference_begin, reference_end = align.index2[0], align.index2[-1]
    offset = abs(reference_begin - query_begin)

    if query_begin < reference_begin:
        align.index1 = [index + offset for index in align.index1]
        shift_direction = "right"
        shifted_query = np.concatenate((np.full(offset, np.nan), shifted_query))
    elif query_end > reference_begin:
        align.index1 = [index - offset for index in align.index1]
        shift_direction = "left"
        shifted_query = shifted_query[offset:]

    ax = dtw.dtwPlotTwoWay(
        align,
        xts=shifted_query,
        yts=reference,
        xlab="time (in seconds)",
        ylab=f"{label} (in Hz)" if "f0" in label else f"{label}",
    )
    ax.set(
        title="Dynamic Time Warping (DTW)",
        xlim=(reference_begin, max(query_end + reference_begin, reference_end) + 1),
    )
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x * frame_length:.1f}"))

    lines = ax.get_lines()
    lines[0].set_label(
        f"Query {label}"
        if shift_direction is None
        else f"Shifted Query {label} ({shift_direction} shift {offset * frame_length:.1f}s)"
    )
    lines[1].set_label(f"Reference {label}")

    ax.legend(loc="upper right")
    return ax


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_path",
                        type=str,
                        required=True,
                        help="Query audio path.",
    )
    parser.add_argument("--reference_path",
                        type=str,
                        required=True,
                        help="Reference audio path.",
    )
    parser.add_argument("--feat_types",
                        nargs="+",
                        default=["mfcc"],
                        choices=["f0", "mfcc", "F0", "MFCC"],
                        help="Supported feature types: f0, MFCC. Default: MFCC.",
    )
    parser.add_argument("--save_plot", 
                        action="store_true", 
                        help="If True, save plot. Otherwise, plotting is skipped.",
    )
    main(parser.parse_args())
