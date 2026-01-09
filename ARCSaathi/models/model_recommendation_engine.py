"""Intelligent model recommendation engine (rule-based, extensible).

Design goals:
- Work with the existing algorithm registry (MLModel.build_registry)
- Analyze a pandas DataFrame to extract meta-features
- Score algorithms across: Accuracy, Speed, Interpretability, Robustness (0-100)
- Provide plain-English explanations + advanced suggestions (ensembles/stacking)

This is intentionally rule-based initially; feedback learning is applied as small score
biases computed from persisted accept/reject history.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetMetaFeatures:
    n_samples: int
    n_features: int
    n_numeric: int
    n_categorical: int
    numeric_ratio: float
    categorical_ratio: float
    missing_pct: float
    class_imbalance: Optional[float]  # max_class_fraction for classification
    linearity_score: Optional[float]  # higher => more linear
    noise_level: Optional[str]  # Low/Medium/High
    corr_mean_abs: Optional[float]
    corr_high_pair_ratio: Optional[float]

    def signature(self) -> str:
        payload = {
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "n_numeric": self.n_numeric,
            "n_categorical": self.n_categorical,
            "missing_pct": round(float(self.missing_pct), 6),
            "class_imbalance": None if self.class_imbalance is None else round(float(self.class_imbalance), 6),
            "linearity_score": None if self.linearity_score is None else round(float(self.linearity_score), 6),
            "noise_level": self.noise_level,
            "corr_mean_abs": None if self.corr_mean_abs is None else round(float(self.corr_mean_abs), 6),
            "corr_high_pair_ratio": None if self.corr_high_pair_ratio is None else round(float(self.corr_high_pair_ratio), 6),
        }
        raw = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.md5(raw).hexdigest()


@dataclass(frozen=True)
class ProblemRequirements:
    task_type: str  # classification/regression/clustering/dimred
    target: Optional[str]
    interpretability: str  # High/Medium/Low
    training_time: str  # Fast/Moderate/Unlimited
    prediction_speed: str  # Fast/Moderate/Unlimited
    deployment_size: str  # Small/Medium/Unlimited
    required_accuracy: Optional[float]  # 0-1 user target; used as guidance


@dataclass(frozen=True)
class ScoreWeights:
    accuracy: int = 40
    speed: int = 25
    interpretability: int = 20
    robustness: int = 15

    def normalized(self) -> "ScoreWeights":
        total = max(1, int(self.accuracy) + int(self.speed) + int(self.interpretability) + int(self.robustness))
        return ScoreWeights(
            accuracy=int(round(self.accuracy * 100 / total)),
            speed=int(round(self.speed * 100 / total)),
            interpretability=int(round(self.interpretability * 100 / total)),
            robustness=int(round(self.robustness * 100 / total)),
        )


@dataclass(frozen=True)
class HardConstraints:
    must_be_interpretable: bool = False
    must_be_fast: bool = False
    must_handle_categorical: bool = False
    max_inference_ms: Optional[float] = None
    max_model_size_mb: Optional[float] = None


@dataclass(frozen=True)
class ModelRecommendation:
    algorithm_key: str
    model_name: str
    family: str
    score_total: int
    scores: Dict[str, int]  # accuracy/speed/interpretability/robustness
    why: str
    strengths: List[str]
    weaknesses: List[str]
    suggested_params: Dict[str, Any]


@dataclass(frozen=True)
class RecommendationResult:
    profile: str
    meta: DatasetMetaFeatures
    requirements: ProblemRequirements
    weights: ScoreWeights
    constraints: HardConstraints
    top: List[ModelRecommendation]
    advanced_suggestions: List[str]
    feature_engineering_suggestions: List[str]


class ModelRecommendationEngine:
    """Rule-based recommender that scores algorithms from the registry."""

    def extract_meta_features(
        self,
        df: pd.DataFrame,
        *,
        task_type: str,
        target: Optional[str] = None,
    ) -> DatasetMetaFeatures:
        if df is None or df.empty:
            return DatasetMetaFeatures(
                n_samples=0,
                n_features=0,
                n_numeric=0,
                n_categorical=0,
                numeric_ratio=0.0,
                categorical_ratio=0.0,
                missing_pct=0.0,
                class_imbalance=None,
                linearity_score=None,
                noise_level=None,
                corr_mean_abs=None,
                corr_high_pair_ratio=None,
            )

        n_samples = int(df.shape[0])
        n_features = int(df.shape[1])

        # basic types
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = [c for c in df.columns if c not in numeric_cols]

        # If target is provided, remove from feature counts so meta-features represent X.
        if target and target in df.columns:
            if target in numeric_cols:
                numeric_cols = [c for c in numeric_cols if c != target]
            if target in categorical_cols:
                categorical_cols = [c for c in categorical_cols if c != target]

        n_numeric = int(len(numeric_cols))
        n_categorical = int(len(categorical_cols))
        denom = max(1, n_numeric + n_categorical)
        numeric_ratio = float(n_numeric / denom)
        categorical_ratio = float(n_categorical / denom)

        missing_pct = float(df.isna().mean().mean() * 100.0)

        class_imbalance: Optional[float] = None
        linearity_score: Optional[float] = None

        # Correlation structure on numeric features only
        corr_mean_abs: Optional[float] = None
        corr_high_pair_ratio: Optional[float] = None
        if n_numeric >= 2:
            try:
                xnum = df[numeric_cols].copy()
                xnum = xnum.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
                if xnum.shape[0] >= 20:
                    corr = xnum.corr(numeric_only=True).abs()
                    # ignore diagonal
                    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                    vals = upper.stack().to_numpy(dtype=float)
                    if vals.size:
                        corr_mean_abs = float(np.nanmean(vals))
                        corr_high_pair_ratio = float(np.mean(vals > 0.9))
            except Exception:
                pass

        # Light linearity proxy using a small baseline model when target exists.
        if target and target in df.columns and task_type in ("classification", "regression"):
            try:
                from sklearn.compose import ColumnTransformer
                from sklearn.impute import SimpleImputer
                from sklearn.linear_model import LinearRegression, LogisticRegression
                from sklearn.metrics import accuracy_score, r2_score
                from sklearn.model_selection import train_test_split
                from sklearn.pipeline import Pipeline
                from sklearn.preprocessing import OneHotEncoder

                y = df[target]
                X = df.drop(columns=[target])

                # quick downsample for speed
                if X.shape[0] > 5000:
                    X = X.sample(n=5000, random_state=42)
                    y = y.loc[X.index]

                num_cols = X.select_dtypes(include=["number"]).columns.tolist()
                cat_cols = [c for c in X.columns if c not in num_cols]

                pre = ColumnTransformer(
                    transformers=[
                        (
                            "num",
                            Pipeline([
                                ("imputer", SimpleImputer(strategy="median")),
                            ]),
                            num_cols,
                        ),
                        (
                            "cat",
                            Pipeline([
                                ("imputer", SimpleImputer(strategy="most_frequent")),
                                ("ohe", OneHotEncoder(handle_unknown="ignore")),
                            ]),
                            cat_cols,
                        ),
                    ],
                    remainder="drop",
                )

                if task_type == "regression":
                    model = LinearRegression()
                else:
                    # keep it stable; avoid long convergence
                    model = LogisticRegression(max_iter=200, n_jobs=None)

                pipe = Pipeline([
                    ("pre", pre),
                    ("model", model),
                ])

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, shuffle=True
                )

                pipe.fit(X_train, y_train)
                y_pred = pipe.predict(X_test)

                if task_type == "regression":
                    s = float(r2_score(y_test, y_pred))
                    linearity_score = max(-1.0, min(1.0, s))
                else:
                    acc = float(accuracy_score(y_test, y_pred))
                    # map [0.5..1] roughly to [0..1] as a crude proxy
                    linearity_score = max(0.0, min(1.0, (acc - 0.5) / 0.5))

                # class imbalance
                if task_type == "classification":
                    vc = y.value_counts(dropna=False)
                    if len(vc) > 0:
                        class_imbalance = float(vc.max() / max(1, vc.sum()))

            except Exception:
                pass

        noise_level: Optional[str] = None
        if linearity_score is not None:
            # Very rough: low linearity often implies noise/non-linearity
            if linearity_score >= 0.7:
                noise_level = "Low"
            elif linearity_score >= 0.4:
                noise_level = "Medium"
            else:
                noise_level = "High"

        return DatasetMetaFeatures(
            n_samples=n_samples,
            n_features=n_features - (1 if (target and target in df.columns) else 0),
            n_numeric=n_numeric,
            n_categorical=n_categorical,
            numeric_ratio=numeric_ratio,
            categorical_ratio=categorical_ratio,
            missing_pct=missing_pct,
            class_imbalance=class_imbalance,
            linearity_score=linearity_score,
            noise_level=noise_level,
            corr_mean_abs=corr_mean_abs,
            corr_high_pair_ratio=corr_high_pair_ratio,
        )

    def recommend(
        self,
        *,
        registry: Dict[str, Any],
        meta: DatasetMetaFeatures,
        requirements: ProblemRequirements,
        profile: str,
        weights: ScoreWeights,
        constraints: HardConstraints,
        feedback_bias: Optional[Dict[str, float]] = None,
        automobile_mode: bool = False,
        automobile_preset: Optional[str] = None,
    ) -> RecommendationResult:
        weights_n = weights.normalized()
        bias = feedback_bias or {}

        # Candidate algorithms: available + matching task
        candidates: List[Tuple[str, Any]] = []
        for k, v in (registry or {}).items():
            if not v.get("available", True):
                continue
            card = v.get("card")
            task = getattr(card, "task_type", "")
            if requirements.task_type and task != requirements.task_type:
                continue
            candidates.append((str(k), v))

        recs: List[ModelRecommendation] = []
        for algorithm_key, entry in candidates:
            card = entry.get("card")
            name = getattr(card, "name", algorithm_key)
            family = str(getattr(card, "family", ""))

            base = self._base_scores(family=family)
            adj, strengths, weaknesses, why_bits, suggested_params = self._apply_rules(
                algorithm_key=algorithm_key,
                family=family,
                name=name,
                meta=meta,
                req=requirements,
                automobile_mode=automobile_mode,
                automobile_preset=automobile_preset,
            )

            scores = {
                "accuracy": int(self._clip(base["accuracy"] + adj.get("accuracy", 0))),
                "speed": int(self._clip(base["speed"] + adj.get("speed", 0))),
                "interpretability": int(self._clip(base["interpretability"] + adj.get("interpretability", 0))),
                "robustness": int(self._clip(base["robustness"] + adj.get("robustness", 0))),
            }

            if constraints.must_be_interpretable and scores["interpretability"] < 60:
                continue
            if constraints.must_be_fast and scores["speed"] < 60:
                continue
            if constraints.must_handle_categorical and meta.n_categorical > 0:
                # require a decent categorical capability; tree/boosting/catboost/lightgbm generally OK
                if "catboost" not in algorithm_key and "lightgbm" not in algorithm_key and family not in (
                    "Tree-based",
                    "Boosting",
                    "Linear",
                ):
                    continue

            # Apply learned feedback bias (small bump)
            bump = float(bias.get(algorithm_key, 0.0))

            total = (
                scores["accuracy"] * weights_n.accuracy
                + scores["speed"] * weights_n.speed
                + scores["interpretability"] * weights_n.interpretability
                + scores["robustness"] * weights_n.robustness
            ) / 100.0
            total = self._clip(total + bump)

            why = ". ".join([b for b in why_bits if b]).strip()
            if why and not why.endswith("."):
                why += "."

            recs.append(
                ModelRecommendation(
                    algorithm_key=algorithm_key,
                    model_name=str(name),
                    family=family,
                    score_total=int(round(total)),
                    scores={k: int(v) for k, v in scores.items()},
                    why=why or "Good general-purpose choice for this dataset and constraints.",
                    strengths=strengths[:6],
                    weaknesses=weaknesses[:6],
                    suggested_params=suggested_params,
                )
            )

        recs.sort(key=lambda r: r.score_total, reverse=True)
        top = recs[:5]

        advanced = self._advanced_suggestions(top, meta, requirements)
        fe_suggestions = self._feature_engineering_suggestions(meta, requirements, automobile_mode, automobile_preset)

        # Profile-based tweaks in narrative only; actual scoring is from weights + rules.
        profile_label = str(profile or "Best Fit")

        return RecommendationResult(
            profile=profile_label,
            meta=meta,
            requirements=requirements,
            weights=weights_n,
            constraints=constraints,
            top=top,
            advanced_suggestions=advanced,
            feature_engineering_suggestions=fe_suggestions,
        )

    # ---------------- internal scoring helpers ----------------

    def _base_scores(self, *, family: str) -> Dict[str, int]:
        f = (family or "").lower()
        if "linear" in f:
            return {"accuracy": 65, "speed": 90, "interpretability": 95, "robustness": 75}
        if "svm" in f:
            return {"accuracy": 75, "speed": 55, "interpretability": 55, "robustness": 80}
        if "tree" in f and "boost" not in f:
            return {"accuracy": 80, "speed": 75, "interpretability": 70, "robustness": 85}
        if "boost" in f:
            return {"accuracy": 88, "speed": 60, "interpretability": 35, "robustness": 80}
        if "neighbors" in f or "knn" in f:
            return {"accuracy": 70, "speed": 70, "interpretability": 45, "robustness": 60}
        if "cluster" in f:
            return {"accuracy": 60, "speed": 80, "interpretability": 55, "robustness": 60}
        if "dimred" in f or "reduction" in f:
            return {"accuracy": 55, "speed": 85, "interpretability": 60, "robustness": 60}
        return {"accuracy": 70, "speed": 70, "interpretability": 60, "robustness": 70}

    def _apply_rules(
        self,
        *,
        algorithm_key: str,
        family: str,
        name: str,
        meta: DatasetMetaFeatures,
        req: ProblemRequirements,
        automobile_mode: bool,
        automobile_preset: Optional[str],
    ) -> Tuple[Dict[str, int], List[str], List[str], List[str], Dict[str, Any]]:
        adj: Dict[str, int] = {}
        strengths: List[str] = []
        weaknesses: List[str] = []
        why: List[str] = []
        suggested_params: Dict[str, Any] = {}

        interpretability = (req.interpretability or "Medium").lower()

        # a) Dataset characteristics
        if meta.n_samples and meta.n_samples < 1000 and interpretability == "high":
            if "linear" in family.lower() or "logistic" in name.lower():
                adj["accuracy"] = adj.get("accuracy", 0) + 8
                adj["interpretability"] = adj.get("interpretability", 0) + 5
                why.append("Small dataset + high interpretability favors linear models")
                strengths.append("Stable on small datasets")
            else:
                adj["accuracy"] = adj.get("accuracy", 0) - 5

        if meta.n_features and meta.n_features > 100 and meta.n_samples and meta.n_samples > 10000:
            if "tree" in family.lower() or "boost" in family.lower():
                adj["accuracy"] = adj.get("accuracy", 0) + 6
                why.append("Many features + many rows favors tree-based models")
                strengths.append("Handles high-dimensional feature spaces")

        if meta.n_categorical and meta.n_categorical > 10:
            if "catboost" in algorithm_key or "lightgbm" in algorithm_key:
                adj["accuracy"] = adj.get("accuracy", 0) + 10
                adj["speed"] = adj.get("speed", 0) + 3
                why.append("Many categorical features: CatBoost/LightGBM are strong candidates")
                strengths.append("Strong with categorical features")

        if meta.class_imbalance is not None and meta.class_imbalance > 0.7 and req.task_type == "classification":
            if "randomforest" in name.lower() or "logistic" in name.lower() or "svm" in family.lower():
                adj["robustness"] = adj.get("robustness", 0) + 8
                why.append("High class imbalance: prefer class-weighted/robust classifiers")
                strengths.append("More resilient to imbalance with class weights")
                suggested_params.setdefault("class_weight", "balanced")
            weaknesses.append("May require imbalance handling (class_weight/SMOTE)")

        if meta.noise_level == "High":
            if "ridge" in name.lower() or "svm" in family.lower() or "tree" in family.lower():
                adj["robustness"] = adj.get("robustness", 0) + 8
                why.append("High noise: robust regularized models and ensembles help")
                strengths.append("Robustness to noise")

        # Correlation / multicollinearity
        if meta.corr_high_pair_ratio is not None and meta.corr_high_pair_ratio > 0.05:
            if "ridge" in name.lower() or "lasso" in name.lower() or "linear" in family.lower():
                adj["robustness"] = adj.get("robustness", 0) + 6
                why.append("High collinearity: regularized linear models are safer")

        # Linearity score
        if meta.linearity_score is not None and meta.linearity_score >= 0.7:
            if "linear" in family.lower():
                adj["accuracy"] = adj.get("accuracy", 0) + 6
                strengths.append("Matches linear signal well")
        if meta.linearity_score is not None and meta.linearity_score < 0.4:
            if "boost" in family.lower() or "tree" in family.lower():
                adj["accuracy"] = adj.get("accuracy", 0) + 6
                why.append("Non-linear patterns detected: trees/boosting often outperform linear models")

        # b) Problem requirements
        if interpretability == "high":
            if "boost" in family.lower() or "xgboost" in algorithm_key:
                adj["interpretability"] = adj.get("interpretability", 0) - 20
                weaknesses.append("Lower interpretability")
            if "linear" in family.lower():
                adj["interpretability"] = adj.get("interpretability", 0) + 5

        tt = (req.training_time or "Moderate").lower()
        if tt == "fast":
            if "boost" in family.lower() or "svm" in family.lower():
                adj["speed"] = adj.get("speed", 0) - 10
                weaknesses.append("Training can be slower")
            if "linear" in family.lower():
                adj["speed"] = adj.get("speed", 0) + 6

        ps = (req.prediction_speed or "Moderate").lower()
        if ps == "fast":
            if "neighbors" in family.lower() or "knn" in family.lower():
                adj["speed"] = adj.get("speed", 0) - 10
                weaknesses.append("Inference can be slow at scale")

        ds = (req.deployment_size or "Medium").lower()
        if ds == "small":
            if "boost" in family.lower() or "forest" in name.lower():
                adj["speed"] = adj.get("speed", 0) - 3
                weaknesses.append("Model can be larger")
            if "linear" in family.lower():
                strengths.append("Small deployment footprint")

        # c) Automobile analytics mode
        if automobile_mode:
            preset = (automobile_preset or "").lower().strip()
            if "car price" in preset and ("xgboost" in algorithm_key or "lightgbm" in algorithm_key or "random forest" in name.lower()):
                adj["accuracy"] = adj.get("accuracy", 0) + 8
                why.append("Automotive preset (car price): boosting/forests are commonly strong")
            if "fuel" in preset and ("gradient" in name.lower() or "linear" in family.lower()):
                adj["accuracy"] = adj.get("accuracy", 0) + 6
                why.append("Automotive preset (fuel efficiency): linear/boosting often perform well")
            if "segmentation" in preset and req.task_type == "clustering":
                if "kmeans" in algorithm_key or "dbscan" in algorithm_key:
                    adj["accuracy"] = adj.get("accuracy", 0) + 6
                    why.append("Automotive preset (segmentation): K-Means/DBSCAN are common baselines")

        # Ensure at least one strength/weakness for UI
        if not strengths:
            strengths.append("Good baseline")
        if not weaknesses:
            weaknesses.append("May need tuning")

        return adj, strengths, weaknesses, why, suggested_params

    def _advanced_suggestions(
        self,
        top: List[ModelRecommendation],
        meta: DatasetMetaFeatures,
        req: ProblemRequirements,
    ) -> List[str]:
        out: List[str] = []
        if not top:
            return out

        families = {t.family for t in top}
        if len(top) >= 2:
            out.append("Ensemble: Try a simple voting/averaging ensemble of the top 2-3 models.")

        if {"Linear", "Tree-based", "Boosting"}.intersection({f.title() for f in families}):
            out.append("Stacking: Combine a linear model with a tree/boosting model for complementary strengths.")

        if req.task_type in ("classification", "regression") and meta.n_features and meta.n_features > 50:
            out.append("Hybrid: Consider a pipeline with feature selection + a strong estimator (e.g., boosting).")

        out.append("Custom pipeline: Add scaling for linear/SVM/KNN; add target encoding/one-hot for categoricals.")
        return out[:6]

    def _feature_engineering_suggestions(
        self,
        meta: DatasetMetaFeatures,
        req: ProblemRequirements,
        automobile_mode: bool,
        automobile_preset: Optional[str],
    ) -> List[str]:
        out: List[str] = []
        if meta.missing_pct and meta.missing_pct > 1.0:
            out.append("Handle missing values (median for numeric, most-frequent for categorical).")
        if meta.n_categorical and meta.n_categorical > 0:
            out.append("Encode categoricals (one-hot for linear; native handling for CatBoost/LightGBM when available).")
        if req.task_type in ("classification", "regression"):
            out.append("Check leakage and remove ID-like columns.")
        if automobile_mode:
            out.append("Automotive FE: Age of vehicle = current_year - year.")
            out.append("Automotive FE: Price-per-mile, engine_size buckets, brand/model frequency features.")
            if automobile_preset and "maintenance" in automobile_preset.lower():
                out.append("Maintenance forecasting: create lag features and rolling aggregates for time-based signals.")
        return out[:8]

    @staticmethod
    def _clip(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
        try:
            return float(max(lo, min(hi, float(x))))
        except Exception:
            return float(lo)
