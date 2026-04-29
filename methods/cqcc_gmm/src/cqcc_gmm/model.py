from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from cqcc_gmm.features import CqccConfig, extract_cqcc


@dataclass(frozen=True)
class GmmConfig:
    n_components: int = 16
    covariance_type: str = "diag"
    max_iter: int = 200
    random_state: int = 42
    max_frames_per_class: int | None = 250_000
    verbose: int = 1


class CqccGmmDetector:
    def __init__(
        self,
        feature_config: CqccConfig | None = None,
        gmm_config: GmmConfig | None = None,
    ) -> None:
        self.feature_config = feature_config or CqccConfig()
        self.gmm_config = gmm_config or GmmConfig()
        self.scaler = StandardScaler()
        self.bonafide_gmm: GaussianMixture | None = None
        self.spoof_gmm: GaussianMixture | None = None
        self.training_stats: dict[str, int | float | str] = {}

    def fit(self, bonafide_files: list[Path], spoof_files: list[Path]) -> None:
        bonafide = self._extract_many(bonafide_files, "bonafide")
        spoof = self._extract_many(spoof_files, "spoof")

        all_features = np.vstack([bonafide, spoof])
        self.scaler.fit(all_features)
        bonafide = self.scaler.transform(bonafide)
        spoof = self.scaler.transform(spoof)

        self.bonafide_gmm = self._fit_gmm(bonafide)
        self.spoof_gmm = self._fit_gmm(spoof)
        self.training_stats.update(
            {
                "bonafide_files": len(bonafide_files),
                "spoof_files": len(spoof_files),
                "bonafide_frames_used": int(bonafide.shape[0]),
                "spoof_frames_used": int(spoof.shape[0]),
                "bonafide_lower_bound": float(self.bonafide_gmm.lower_bound_),
                "spoof_lower_bound": float(self.spoof_gmm.lower_bound_),
                "gmm_components": self.gmm_config.n_components,
                "max_frames_per_class": self.gmm_config.max_frames_per_class or "all",
            }
        )

    def score_file(self, path: str | Path, threshold: float = 0.0) -> dict[str, float | str]:
        self._ensure_ready()
        features = extract_cqcc(path, self.feature_config)
        features = self.scaler.transform(features)
        assert self.bonafide_gmm is not None
        assert self.spoof_gmm is not None
        bonafide_ll = float(self.bonafide_gmm.score_samples(features).mean())
        spoof_ll = float(self.spoof_gmm.score_samples(features).mean())
        score = bonafide_ll - spoof_ll
        return {
            "path": str(path),
            "score": score,
            "bonafide_loglike": bonafide_ll,
            "spoof_loglike": spoof_ll,
            "prediction": "bonafide" if score >= threshold else "spoof",
        }

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output)

    @classmethod
    def load(cls, path: str | Path) -> "CqccGmmDetector":
        detector = joblib.load(path)
        if not isinstance(detector, cls):
            raise TypeError(f"Unexpected model type in {path}: {type(detector)!r}")
        detector._ensure_ready()
        return detector

    def _extract_many(self, files: list[Path], label: str) -> np.ndarray:
        if not files:
            raise ValueError(f"No {label} files provided.")

        features = []
        reservoir: np.ndarray | None = None
        filled = 0
        seen = 0
        frame_limit = self.gmm_config.max_frames_per_class
        rng = np.random.default_rng(self.gmm_config.random_state)
        for file in tqdm(files, desc=f"Extracting {label}", unit="file", ascii=True):
            if not file.exists():
                raise FileNotFoundError(file)
            file_features = extract_cqcc(file, self.feature_config)
            if not frame_limit:
                features.append(file_features)
                continue
            if reservoir is None:
                reservoir = np.empty((frame_limit, file_features.shape[1]), dtype=np.float32)
            file_features = file_features.astype(np.float32, copy=False)
            n_rows = file_features.shape[0]
            if filled < frame_limit:
                take = min(frame_limit - filled, n_rows)
                reservoir[filled : filled + take] = file_features[:take]
                filled += take
                remaining = file_features[take:]
                seen += take
            else:
                remaining = file_features

            if remaining.size:
                n_remaining = remaining.shape[0]
                highs = np.arange(seen + 1, seen + n_remaining + 1)
                replace_at = rng.integers(0, highs)
                mask = replace_at < frame_limit
                reservoir[replace_at[mask]] = remaining[mask]
                seen += n_remaining

        if frame_limit:
            if reservoir is None or filled == 0:
                raise ValueError(f"No frames extracted for {label}.")
            return reservoir[:filled]
        return np.vstack(features)

    def _fit_gmm(self, features: np.ndarray) -> GaussianMixture:
        components = min(self.gmm_config.n_components, max(1, features.shape[0] // 2))
        model = GaussianMixture(
            n_components=components,
            covariance_type=self.gmm_config.covariance_type,
            max_iter=self.gmm_config.max_iter,
            random_state=self.gmm_config.random_state,
            verbose=self.gmm_config.verbose,
            verbose_interval=1,
        )
        return model.fit(features)

    def _ensure_ready(self) -> None:
        if self.bonafide_gmm is None or self.spoof_gmm is None:
            raise RuntimeError("Model is not trained. Run 'cqcc-gmm train' first.")
