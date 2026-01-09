"""Algorithm registry and model cards for ARCSaathi.

This module defines:
- A catalog of supported algorithms across task types.
- Model cards (description, pros/cons, complexity, defaults).
- Factory methods that create estimator instances (with optional dependencies).

Notes
- Some algorithms require optional third-party packages (xgboost, lightgbm, catboost,
  hdbscan, umap-learn, sklearn-extra). They will be marked unavailable if missing.
- "Divisive" hierarchical clustering is represented using sklearn's BisectingKMeans.
- "Autoencoders" are listed as a technique, but require a deep learning backend
  (not bundled by default).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass(frozen=True)
class HyperParamSpec:
    name: str
    default: Any
    kind: str  # int|float|bool|str|choice
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    choices: Optional[List[Any]] = None


@dataclass(frozen=True)
class ModelCard:
    key: str
    name: str
    task_type: str  # regression|classification|clustering|dimred
    family: str
    description: str
    best_for: str
    pros: str
    cons: str
    time_complexity: str
    memory_usage: str
    expected_performance: str
    params: List[HyperParamSpec]
    optional_deps: List[str]


@dataclass(frozen=True)
class AlgorithmSpec:
    card: ModelCard
    factory: Callable[[Dict[str, Any]], Any]

    def is_available(self) -> tuple[bool, str]:
        for dep in self.card.optional_deps:
            try:
                __import__(dep)
            except Exception:
                return (False, f"Missing dependency: {dep}")
        return (True, "")


def _sklearn() -> Any:
    import sklearn  # noqa: F401
    return sklearn


# ---- Factory helpers (with optional deps) ----

def _make_sklearn(cls_path: str) -> Callable[[Dict[str, Any]], Any]:
    def _f(params: Dict[str, Any]) -> Any:
        module_name, class_name = cls_path.rsplit(".", 1)
        mod = __import__(module_name, fromlist=[class_name])
        cls = getattr(mod, class_name)
        return cls(**params)

    return _f


def _make_xgboost_reg(params: Dict[str, Any]) -> Any:
    from xgboost import XGBRegressor

    return XGBRegressor(**params)


def _make_xgboost_clf(params: Dict[str, Any]) -> Any:
    from xgboost import XGBClassifier

    return XGBClassifier(**params)


def _make_lightgbm_reg(params: Dict[str, Any]) -> Any:
    from lightgbm import LGBMRegressor

    return LGBMRegressor(**params)


def _make_lightgbm_clf(params: Dict[str, Any]) -> Any:
    from lightgbm import LGBMClassifier

    return LGBMClassifier(**params)


def _make_catboost_reg(params: Dict[str, Any]) -> Any:
    from catboost import CatBoostRegressor

    return CatBoostRegressor(**params)


def _make_catboost_clf(params: Dict[str, Any]) -> Any:
    from catboost import CatBoostClassifier

    return CatBoostClassifier(**params)


def _make_hdbscan(params: Dict[str, Any]) -> Any:
    import hdbscan

    return hdbscan.HDBSCAN(**params)


def _make_umap(params: Dict[str, Any]) -> Any:
    import umap

    return umap.UMAP(**params)


def _make_kmedoids(params: Dict[str, Any]) -> Any:
    from sklearn_extra.cluster import KMedoids

    return KMedoids(**params)


def build_registry() -> Dict[str, AlgorithmSpec]:
    """Return {algorithm_key: AlgorithmSpec} for all supported algorithms."""

    # Regression (18)
    reg: List[ModelCard] = [
        ModelCard(
            key="reg_linear",
            name="Linear Regression",
            task_type="regression",
            family="Linear Models",
            description="Baseline linear model for continuous targets.",
            best_for="Large numeric datasets; interpretability.",
            pros="Fast; interpretable; strong baseline.",
            cons="Underfits non-linear patterns; sensitive to outliers.",
            time_complexity="O(n·p²) (fit via least squares)",
            memory_usage="Low",
            expected_performance="Good baseline; improves with feature engineering.",
            params=[],
            optional_deps=[],
        ),
        ModelCard(
            key="reg_ridge",
            name="Ridge Regression",
            task_type="regression",
            family="Linear Models",
            description="Linear regression with L2 regularization.",
            best_for="High-dimensional data; multicollinearity.",
            pros="Stable; reduces overfitting vs. plain linear.",
            cons="Still linear; requires tuning alpha.",
            time_complexity="O(n·p²)",
            memory_usage="Low",
            expected_performance="Often better than Linear on noisy data.",
            params=[HyperParamSpec("alpha", 1.0, "float", 1e-6, 1e3, 0.1)],
            optional_deps=[],
        ),
        ModelCard(
            key="reg_lasso",
            name="Lasso Regression",
            task_type="regression",
            family="Linear Models",
            description="Linear regression with L1 regularization (sparse).",
            best_for="Feature selection; sparse solutions.",
            pros="Can zero-out irrelevant features.",
            cons="Can be unstable with correlated features.",
            time_complexity="Varies (coordinate descent)",
            memory_usage="Low",
            expected_performance="Strong when many irrelevant features exist.",
            params=[HyperParamSpec("alpha", 0.001, "float", 1e-6, 10.0, 0.001)],
            optional_deps=[],
        ),
        ModelCard(
            key="reg_elasticnet",
            name="ElasticNet",
            task_type="regression",
            family="Linear Models",
            description="Linear model with combined L1 + L2 regularization.",
            best_for="Correlated features; balance sparsity + stability.",
            pros="More flexible regularization than Lasso/Ridge.",
            cons="More hyperparameters to tune.",
            time_complexity="Varies",
            memory_usage="Low",
            expected_performance="Often robust for messy real-world data.",
            params=[
                HyperParamSpec("alpha", 0.001, "float", 1e-6, 10.0, 0.001),
                HyperParamSpec("l1_ratio", 0.5, "float", 0.0, 1.0, 0.05),
            ],
            optional_deps=[],
        ),
        ModelCard(
            key="reg_bayesianridge",
            name="Bayesian Ridge",
            task_type="regression",
            family="Linear Models",
            description="Bayesian linear regression with priors over weights.",
            best_for="Uncertainty-aware linear regression.",
            pros="Provides posterior estimates; regularized.",
            cons="Assumes linearity; Gaussian noise assumptions.",
            time_complexity="O(n·p²)",
            memory_usage="Low",
            expected_performance="Good baseline with uncertainty estimates.",
            params=[],
            optional_deps=[],
        ),
        ModelCard(
            key="reg_decision_tree",
            name="Decision Tree Regressor",
            task_type="regression",
            family="Tree-based",
            description="Non-linear tree for regression.",
            best_for="Non-linear patterns; mixed feature types (after encoding).",
            pros="Captures interactions; no scaling required.",
            cons="Can overfit; unstable.",
            time_complexity="O(n·p·log n)",
            memory_usage="Low-Medium",
            expected_performance="Good with tuned depth/min samples.",
            params=[
                HyperParamSpec("max_depth", None, "int", 1, 64, 1),
                HyperParamSpec("min_samples_split", 2, "int", 2, 50, 1),
            ],
            optional_deps=[],
        ),
        ModelCard(
            key="reg_random_forest",
            name="Random Forest Regressor",
            task_type="regression",
            family="Tree-based",
            description="Bagged ensemble of decision trees.",
            best_for="Strong general-purpose regression.",
            pros="Robust; handles non-linearities; less overfitting than single tree.",
            cons="Less interpretable; slower.",
            time_complexity="O(T·n·p·log n)",
            memory_usage="Medium",
            expected_performance="Often strong without heavy tuning.",
            params=[
                HyperParamSpec("n_estimators", 200, "int", 50, 2000, 50),
                HyperParamSpec("max_depth", None, "int", 1, 64, 1),
            ],
            optional_deps=[],
        ),
        ModelCard(
            key="reg_extra_trees",
            name="Extra Trees Regressor",
            task_type="regression",
            family="Tree-based",
            description="Extremely randomized trees ensemble.",
            best_for="Fast strong baseline; high-variance data.",
            pros="Often competitive; can be faster than RF.",
            cons="Still heavy; less interpretable.",
            time_complexity="O(T·n·p·log n)",
            memory_usage="Medium",
            expected_performance="Strong baseline; good generalization.",
            params=[HyperParamSpec("n_estimators", 500, "int", 50, 3000, 50)],
            optional_deps=[],
        ),
        ModelCard(
            key="reg_gb",
            name="Gradient Boosting Regressor",
            task_type="regression",
            family="Boosting",
            description="Boosted ensemble of shallow trees.",
            best_for="Tabular regression; non-linear relationships.",
            pros="High accuracy; flexible.",
            cons="Sensitive to params; slower.",
            time_complexity="O(T·n·p)",
            memory_usage="Medium",
            expected_performance="Very strong with tuned learning rate.",
            params=[
                HyperParamSpec("n_estimators", 300, "int", 50, 5000, 50),
                HyperParamSpec("learning_rate", 0.05, "float", 0.001, 1.0, 0.01),
            ],
            optional_deps=[],
        ),
        ModelCard(
            key="reg_adaboost",
            name="AdaBoost Regressor",
            task_type="regression",
            family="Boosting",
            description="Boosting with weak regressors.",
            best_for="Smaller tabular data; robust baseline.",
            pros="Simple; can improve over single trees.",
            cons="Can be sensitive to noise/outliers.",
            time_complexity="O(T·n·p)",
            memory_usage="Medium",
            expected_performance="Moderate; sometimes strong.",
            params=[
                HyperParamSpec("n_estimators", 300, "int", 50, 5000, 50),
                HyperParamSpec("learning_rate", 0.05, "float", 0.001, 1.0, 0.01),
            ],
            optional_deps=[],
        ),
        ModelCard(
            key="reg_xgboost",
            name="XGBoost Regressor",
            task_type="regression",
            family="Boosting",
            description="Gradient boosting with optimized tree learning.",
            best_for="High-performance tabular regression.",
            pros="Very strong; supports early stopping.",
            cons="Extra dependency; can be slow to tune.",
            time_complexity="O(T·n·p)",
            memory_usage="Medium",
            expected_performance="Often top-tier on tabular datasets.",
            params=[
                HyperParamSpec("n_estimators", 500, "int", 50, 5000, 50),
                HyperParamSpec("learning_rate", 0.05, "float", 0.001, 1.0, 0.01),
                HyperParamSpec("max_depth", 6, "int", 1, 20, 1),
                HyperParamSpec("subsample", 0.8, "float", 0.1, 1.0, 0.05),
            ],
            optional_deps=["xgboost"],
        ),
        ModelCard(
            key="reg_lightgbm",
            name="LightGBM Regressor",
            task_type="regression",
            family="Boosting",
            description="Histogram-based gradient boosting (fast).",
            best_for="Large datasets; fast training.",
            pros="Fast; strong; handles large data well.",
            cons="Extra dependency; careful tuning needed.",
            time_complexity="O(T·n·p)",
            memory_usage="Medium",
            expected_performance="Often top-tier on tabular datasets.",
            params=[
                HyperParamSpec("n_estimators", 1000, "int", 50, 10000, 50),
                HyperParamSpec("learning_rate", 0.05, "float", 0.001, 1.0, 0.01),
                HyperParamSpec("num_leaves", 31, "int", 2, 1024, 1),
            ],
            optional_deps=["lightgbm"],
        ),
        ModelCard(
            key="reg_catboost",
            name="CatBoost Regressor",
            task_type="regression",
            family="Boosting",
            description="Boosting with strong categorical handling.",
            best_for="Categorical-heavy tabular data.",
            pros="Strong out-of-the-box; good with categoricals.",
            cons="Extra dependency; slower for large tuning.",
            time_complexity="O(T·n·p)",
            memory_usage="Medium",
            expected_performance="Very strong on mixed-type data.",
            params=[
                HyperParamSpec("iterations", 2000, "int", 50, 20000, 50),
                HyperParamSpec("learning_rate", 0.05, "float", 0.001, 1.0, 0.01),
                HyperParamSpec("depth", 6, "int", 2, 12, 1),
                HyperParamSpec("verbose", False, "bool"),
            ],
            optional_deps=["catboost"],
        ),
        ModelCard(
            key="reg_knn",
            name="KNN Regressor",
            task_type="regression",
            family="Distance-based",
            description="k-nearest neighbors regression.",
            best_for="Small-medium datasets; smooth targets.",
            pros="Simple; non-parametric.",
            cons="Slow prediction; needs scaling.",
            time_complexity="O(n) per query",
            memory_usage="Medium",
            expected_performance="Good with proper scaling and k.",
            params=[HyperParamSpec("n_neighbors", 5, "int", 1, 200, 1)],
            optional_deps=[],
        ),
        ModelCard(
            key="reg_svr",
            name="SVR",
            task_type="regression",
            family="SVM",
            description="Support Vector Regression with kernels.",
            best_for="Small-medium datasets; non-linear regression.",
            pros="Strong on small datasets; flexible kernels.",
            cons="Does not scale well; needs tuning.",
            time_complexity="~O(n³) worst-case",
            memory_usage="Medium",
            expected_performance="Strong when tuned; limited scalability.",
            params=[
                HyperParamSpec("C", 1.0, "float", 1e-3, 1e3, 0.1),
                HyperParamSpec("kernel", "rbf", "choice", choices=["linear", "rbf", "poly", "sigmoid"]),
            ],
            optional_deps=[],
        ),
        ModelCard(
            key="reg_mlp",
            name="MLP Regressor",
            task_type="regression",
            family="Neural",
            description="Feed-forward neural network regressor.",
            best_for="Non-linear relationships; scaled features.",
            pros="Flexible; can model complex functions.",
            cons="Sensitive to scaling; tuning required.",
            time_complexity="O(E·n·p)",
            memory_usage="Medium",
            expected_performance="Good with tuning and enough data.",
            params=[
                HyperParamSpec("hidden_layer_sizes", (128, 64), "str"),
                HyperParamSpec("alpha", 0.0001, "float", 1e-6, 1.0, 1e-4),
                HyperParamSpec("max_iter", 500, "int", 50, 5000, 50),
            ],
            optional_deps=[],
        ),
    ]

    # Classification (14)
    clf: List[ModelCard] = [
        ModelCard(
            key="clf_logistic",
            name="Logistic Regression",
            task_type="classification",
            family="Linear",
            description="Linear classifier with probabilistic outputs.",
            best_for="Baseline classification; calibrated probabilities.",
            pros="Fast; interpretable; good baseline.",
            cons="Linear decision boundary.",
            time_complexity="O(n·p)",
            memory_usage="Low",
            expected_performance="Strong baseline for many problems.",
            params=[
                HyperParamSpec("C", 1.0, "float", 1e-3, 1e3, 0.1),
                HyperParamSpec("max_iter", 500, "int", 50, 5000, 50),
            ],
            optional_deps=[],
        ),
        ModelCard(
            key="clf_ridge",
            name="Ridge Classifier",
            task_type="classification",
            family="Linear",
            description="Linear classifier using ridge regression.",
            best_for="High-dimensional sparse features.",
            pros="Fast; robust.",
            cons="Linear boundary.",
            time_complexity="O(n·p²)",
            memory_usage="Low",
            expected_performance="Good baseline in text-like features.",
            params=[HyperParamSpec("alpha", 1.0, "float", 1e-6, 1e3, 0.1)],
            optional_deps=[],
        ),
        ModelCard(
            key="clf_decision_tree",
            name="Decision Tree Classifier",
            task_type="classification",
            family="Tree-based",
            description="Non-linear tree classifier.",
            best_for="Interpretable splits; mixed features.",
            pros="Captures non-linearities; no scaling.",
            cons="Overfits easily.",
            time_complexity="O(n·p·log n)",
            memory_usage="Low-Medium",
            expected_performance="Good with depth control.",
            params=[
                HyperParamSpec("max_depth", None, "int", 1, 64, 1),
                HyperParamSpec("min_samples_split", 2, "int", 2, 50, 1),
            ],
            optional_deps=[],
        ),
        ModelCard(
            key="clf_random_forest",
            name="Random Forest Classifier",
            task_type="classification",
            family="Tree-based",
            description="Bagged ensemble of trees.",
            best_for="Strong general-purpose classifier.",
            pros="Robust; handles non-linearities.",
            cons="Less interpretable; larger models.",
            time_complexity="O(T·n·p·log n)",
            memory_usage="Medium",
            expected_performance="Strong baseline.",
            params=[
                HyperParamSpec("n_estimators", 300, "int", 50, 3000, 50),
                HyperParamSpec("max_depth", None, "int", 1, 64, 1),
            ],
            optional_deps=[],
        ),
        ModelCard(
            key="clf_extra_trees",
            name="Extra Trees Classifier",
            task_type="classification",
            family="Tree-based",
            description="Extremely randomized trees.",
            best_for="Fast strong classifier.",
            pros="Competitive performance; less tuning.",
            cons="Less interpretable.",
            time_complexity="O(T·n·p·log n)",
            memory_usage="Medium",
            expected_performance="Strong baseline.",
            params=[HyperParamSpec("n_estimators", 500, "int", 50, 5000, 50)],
            optional_deps=[],
        ),
        ModelCard(
            key="clf_gb",
            name="Gradient Boosting Classifier",
            task_type="classification",
            family="Boosting",
            description="Boosted trees for classification.",
            best_for="Tabular classification.",
            pros="High accuracy.",
            cons="Sensitive to learning rate.",
            time_complexity="O(T·n·p)",
            memory_usage="Medium",
            expected_performance="Very strong with tuning.",
            params=[
                HyperParamSpec("n_estimators", 300, "int", 50, 5000, 50),
                HyperParamSpec("learning_rate", 0.05, "float", 0.001, 1.0, 0.01),
            ],
            optional_deps=[],
        ),
        ModelCard(
            key="clf_adaboost",
            name="AdaBoost Classifier",
            task_type="classification",
            family="Boosting",
            description="Boosting with weak learners.",
            best_for="Simple boosting baseline.",
            pros="Can improve weak models.",
            cons="Sensitive to noise.",
            time_complexity="O(T·n·p)",
            memory_usage="Medium",
            expected_performance="Moderate to strong.",
            params=[
                HyperParamSpec("n_estimators", 400, "int", 50, 5000, 50),
                HyperParamSpec("learning_rate", 0.05, "float", 0.001, 1.0, 0.01),
            ],
            optional_deps=[],
        ),
        ModelCard(
            key="clf_xgboost",
            name="XGBoost Classifier",
            task_type="classification",
            family="Boosting",
            description="Optimized gradient boosting.",
            best_for="High-performance tabular classification.",
            pros="Top-tier performance; early stopping.",
            cons="Extra dependency; tuning heavy.",
            time_complexity="O(T·n·p)",
            memory_usage="Medium",
            expected_performance="Often top-tier.",
            params=[
                HyperParamSpec("n_estimators", 800, "int", 50, 10000, 50),
                HyperParamSpec("learning_rate", 0.05, "float", 0.001, 1.0, 0.01),
                HyperParamSpec("max_depth", 6, "int", 1, 20, 1),
                HyperParamSpec("subsample", 0.8, "float", 0.1, 1.0, 0.05),
            ],
            optional_deps=["xgboost"],
        ),
        ModelCard(
            key="clf_lightgbm",
            name="LightGBM Classifier",
            task_type="classification",
            family="Boosting",
            description="Fast histogram-based boosting.",
            best_for="Large datasets; fast training.",
            pros="Fast; strong.",
            cons="Extra dependency.",
            time_complexity="O(T·n·p)",
            memory_usage="Medium",
            expected_performance="Often top-tier.",
            params=[
                HyperParamSpec("n_estimators", 1500, "int", 50, 20000, 50),
                HyperParamSpec("learning_rate", 0.05, "float", 0.001, 1.0, 0.01),
                HyperParamSpec("num_leaves", 31, "int", 2, 1024, 1),
            ],
            optional_deps=["lightgbm"],
        ),
        ModelCard(
            key="clf_catboost",
            name="CatBoost Classifier",
            task_type="classification",
            family="Boosting",
            description="Boosting with strong categorical support.",
            best_for="Categorical-heavy tabular data.",
            pros="Strong defaults; handles categoricals.",
            cons="Extra dependency.",
            time_complexity="O(T·n·p)",
            memory_usage="Medium",
            expected_performance="Very strong.",
            params=[
                HyperParamSpec("iterations", 2500, "int", 50, 50000, 50),
                HyperParamSpec("learning_rate", 0.05, "float", 0.001, 1.0, 0.01),
                HyperParamSpec("depth", 6, "int", 2, 12, 1),
                HyperParamSpec("verbose", False, "bool"),
            ],
            optional_deps=["catboost"],
        ),
        ModelCard(
            key="clf_knn",
            name="KNN Classifier",
            task_type="classification",
            family="Distance-based",
            description="k-nearest neighbors classifier.",
            best_for="Small datasets; local decision boundaries.",
            pros="Simple; non-parametric.",
            cons="Slow prediction; needs scaling.",
            time_complexity="O(n) per query",
            memory_usage="Medium",
            expected_performance="Good with scaling + k.",
            params=[HyperParamSpec("n_neighbors", 5, "int", 1, 200, 1)],
            optional_deps=[],
        ),
        ModelCard(
            key="clf_svm",
            name="SVM (SVC)",
            task_type="classification",
            family="SVM",
            description="Support Vector Classifier with kernels.",
            best_for="Small-medium datasets.",
            pros="Strong margins; flexible kernels.",
            cons="Does not scale well; tuning required.",
            time_complexity="~O(n³) worst-case",
            memory_usage="Medium",
            expected_performance="Strong when tuned.",
            params=[
                HyperParamSpec("C", 1.0, "float", 1e-3, 1e3, 0.1),
                HyperParamSpec("kernel", "rbf", "choice", choices=["linear", "rbf", "poly", "sigmoid"]),
                HyperParamSpec("probability", True, "bool"),
            ],
            optional_deps=[],
        ),
        ModelCard(
            key="clf_naive_bayes",
            name="Naive Bayes (Gaussian)",
            task_type="classification",
            family="Probabilistic",
            description="Probabilistic classifier with independence assumption.",
            best_for="Fast baseline; continuous features.",
            pros="Very fast; robust baseline.",
            cons="Independence assumption.",
            time_complexity="O(n·p)",
            memory_usage="Low",
            expected_performance="Good baseline; sometimes surprisingly strong.",
            params=[],
            optional_deps=[],
        ),
        ModelCard(
            key="clf_mlp",
            name="MLP Classifier",
            task_type="classification",
            family="Neural",
            description="Feed-forward neural network classifier.",
            best_for="Non-linear classification with enough data.",
            pros="Flexible; can model complex patterns.",
            cons="Sensitive to scaling; tuning heavy.",
            time_complexity="O(E·n·p)",
            memory_usage="Medium",
            expected_performance="Good with tuning.",
            params=[
                HyperParamSpec("hidden_layer_sizes", (128, 64), "str"),
                HyperParamSpec("alpha", 0.0001, "float", 1e-6, 1.0, 1e-4),
                HyperParamSpec("max_iter", 500, "int", 50, 5000, 50),
            ],
            optional_deps=[],
        ),
    ]

    # Clustering (10)
    clu: List[ModelCard] = [
        ModelCard(
            key="clu_kmeans",
            name="K-Means",
            task_type="clustering",
            family="Centroid-based",
            description="Centroid-based clustering.",
            best_for="Spherical clusters; scaled numeric features.",
            pros="Fast; simple.",
            cons="Needs k; sensitive to scaling/outliers.",
            time_complexity="O(n·k·i)",
            memory_usage="Low",
            expected_performance="Good baseline when k known.",
            params=[HyperParamSpec("n_clusters", 8, "int", 2, 200, 1), HyperParamSpec("random_state", 42, "int")],
            optional_deps=[],
        ),
        ModelCard(
            key="clu_minibatch_kmeans",
            name="Mini-Batch K-Means",
            task_type="clustering",
            family="Centroid-based",
            description="K-means optimized for large datasets.",
            best_for="Large datasets.",
            pros="Fast; scalable.",
            cons="Approximate.",
            time_complexity="O(b·k·i)",
            memory_usage="Low",
            expected_performance="Strong when tuned.",
            params=[HyperParamSpec("n_clusters", 8, "int", 2, 200, 1), HyperParamSpec("random_state", 42, "int")],
            optional_deps=[],
        ),
        ModelCard(
            key="clu_kmedoids",
            name="K-Medoids",
            task_type="clustering",
            family="Centroid-based",
            description="Centroid-based clustering using medoids.",
            best_for="Robust clustering; non-Euclidean distances.",
            pros="More robust than k-means.",
            cons="Optional dependency; slower.",
            time_complexity="Higher than k-means",
            memory_usage="Medium",
            expected_performance="Good when outliers present.",
            params=[HyperParamSpec("n_clusters", 8, "int", 2, 200, 1), HyperParamSpec("random_state", 42, "int")],
            optional_deps=["sklearn_extra"],
        ),
        ModelCard(
            key="clu_agglomerative",
            name="Agglomerative Clustering",
            task_type="clustering",
            family="Hierarchical",
            description="Bottom-up hierarchical clustering.",
            best_for="Small-medium datasets.",
            pros="No need to specify clusters (can cut dendrogram).",
            cons="Can be slow on large n.",
            time_complexity="O(n²)",
            memory_usage="High",
            expected_performance="Good for structured clusters.",
            params=[HyperParamSpec("n_clusters", 8, "int", 2, 200, 1), HyperParamSpec("linkage", "ward", "choice", choices=["ward", "complete", "average", "single"])],
            optional_deps=[],
        ),
        ModelCard(
            key="clu_divisive",
            name="Divisive (Bisecting K-Means)",
            task_type="clustering",
            family="Hierarchical",
            description="Top-down hierarchical clustering via bisecting k-means.",
            best_for="Large datasets needing hierarchical splits.",
            pros="Hierarchical; scalable vs. agglomerative.",
            cons="Approximate; still needs k.",
            time_complexity="O(n·k·i)",
            memory_usage="Medium",
            expected_performance="Good scalable hierarchical alternative.",
            params=[HyperParamSpec("n_clusters", 8, "int", 2, 200, 1), HyperParamSpec("random_state", 42, "int")],
            optional_deps=[],
        ),
        ModelCard(
            key="clu_dbscan",
            name="DBSCAN",
            task_type="clustering",
            family="Density-based",
            description="Density-based clustering.",
            best_for="Arbitrary-shaped clusters; noise handling.",
            pros="Finds noise; no k.",
            cons="Sensitive to eps; struggles with varying density.",
            time_complexity="O(n log n) typical",
            memory_usage="Medium",
            expected_performance="Great when density assumptions hold.",
            params=[HyperParamSpec("eps", 0.5, "float", 0.01, 10.0, 0.01), HyperParamSpec("min_samples", 5, "int", 1, 100, 1)],
            optional_deps=[],
        ),
        ModelCard(
            key="clu_hdbscan",
            name="HDBSCAN",
            task_type="clustering",
            family="Density-based",
            description="Hierarchical DBSCAN for varying densities.",
            best_for="Varying density clusters.",
            pros="Handles varying density better than DBSCAN.",
            cons="Optional dependency.",
            time_complexity="O(n log n)",
            memory_usage="Medium",
            expected_performance="Often very strong.",
            params=[HyperParamSpec("min_cluster_size", 5, "int", 2, 500, 1)],
            optional_deps=["hdbscan"],
        ),
        ModelCard(
            key="clu_optics",
            name="OPTICS",
            task_type="clustering",
            family="Density-based",
            description="Density-based clustering similar to DBSCAN.",
            best_for="Varying density; cluster structure exploration.",
            pros="Less sensitive to eps.",
            cons="Heavier than DBSCAN.",
            time_complexity="O(n log n) typical",
            memory_usage="Medium",
            expected_performance="Good for density structure discovery.",
            params=[HyperParamSpec("min_samples", 5, "int", 1, 100, 1)],
            optional_deps=[],
        ),
        ModelCard(
            key="clu_gmm",
            name="Gaussian Mixture (GMM)",
            task_type="clustering",
            family="Probabilistic",
            description="Probabilistic clustering with Gaussians.",
            best_for="Elliptical clusters; soft assignments.",
            pros="Soft clustering; probabilistic.",
            cons="Needs k; can converge poorly.",
            time_complexity="O(n·k·i)",
            memory_usage="Medium",
            expected_performance="Strong when Gaussian assumption holds.",
            params=[HyperParamSpec("n_components", 8, "int", 2, 200, 1), HyperParamSpec("random_state", 42, "int")],
            optional_deps=[],
        ),
        ModelCard(
            key="clu_spectral",
            name="Spectral Clustering",
            task_type="clustering",
            family="Spectral",
            description="Graph-based clustering using eigenvectors.",
            best_for="Non-convex clusters; smaller datasets.",
            pros="Powerful for complex cluster shapes.",
            cons="Can be expensive; sensitive to affinity.",
            time_complexity="O(n³) worst-case",
            memory_usage="High",
            expected_performance="Strong on small complex datasets.",
            params=[HyperParamSpec("n_clusters", 8, "int", 2, 200, 1), HyperParamSpec("random_state", 42, "int")],
            optional_deps=[],
        ),
    ]

    # Dimensionality reduction (5)
    dr: List[ModelCard] = [
        ModelCard(
            key="dr_pca",
            name="PCA",
            task_type="dimred",
            family="Linear",
            description="Principal Component Analysis (linear projection).",
            best_for="Noise reduction; compression; visualization.",
            pros="Fast; interpretable components.",
            cons="Linear only.",
            time_complexity="O(n·p²)",
            memory_usage="Low",
            expected_performance="Strong baseline for dim reduction.",
            params=[HyperParamSpec("n_components", 10, "int", 2, 500, 1), HyperParamSpec("random_state", 42, "int")],
            optional_deps=[],
        ),
        ModelCard(
            key="dr_lda",
            name="LDA",
            task_type="dimred",
            family="Supervised Linear",
            description="Linear Discriminant Analysis (supervised projection).",
            best_for="Classification feature extraction.",
            pros="Uses labels; good class separation.",
            cons="Requires target labels.",
            time_complexity="O(n·p²)",
            memory_usage="Low",
            expected_performance="Strong when classes linearly separable.",
            params=[HyperParamSpec("n_components", 2, "int", 1, 100, 1)],
            optional_deps=[],
        ),
        ModelCard(
            key="dr_tsne",
            name="t-SNE",
            task_type="dimred",
            family="Nonlinear",
            description="Nonlinear embedding for visualization.",
            best_for="2D/3D visualization.",
            pros="Great visualization.",
            cons="Slow; not for downstream modeling.",
            time_complexity="High",
            memory_usage="High",
            expected_performance="Excellent for visual separation.",
            params=[HyperParamSpec("n_components", 2, "int", 2, 3, 1), HyperParamSpec("perplexity", 30.0, "float", 5.0, 100.0, 1.0), HyperParamSpec("random_state", 42, "int")],
            optional_deps=[],
        ),
        ModelCard(
            key="dr_umap",
            name="UMAP",
            task_type="dimred",
            family="Nonlinear",
            description="Nonlinear embedding often faster than t-SNE.",
            best_for="Visualization and embeddings.",
            pros="Fast; preserves structure well.",
            cons="Optional dependency.",
            time_complexity="Medium",
            memory_usage="Medium",
            expected_performance="Strong embeddings.",
            params=[HyperParamSpec("n_components", 2, "int", 2, 200, 1), HyperParamSpec("n_neighbors", 15, "int", 2, 200, 1), HyperParamSpec("min_dist", 0.1, "float", 0.0, 1.0, 0.01), HyperParamSpec("random_state", 42, "int")],
            optional_deps=["umap"],
        ),
        ModelCard(
            key="dr_autoencoder",
            name="Autoencoders",
            task_type="dimred",
            family="Neural",
            description="Neural network-based dimensionality reduction.",
            best_for="Complex nonlinear embeddings.",
            pros="Highly flexible.",
            cons="Requires deep learning backend.",
            time_complexity="High",
            memory_usage="High",
            expected_performance="Strong when enough data.",
            params=[HyperParamSpec("latent_dim", 16, "int", 2, 512, 1)],
            optional_deps=["torch"],
        ),
    ]

    registry: Dict[str, AlgorithmSpec] = {}

    # Regression factories
    reg_factories: Dict[str, Callable[[Dict[str, Any]], Any]] = {
        "reg_linear": _make_sklearn("sklearn.linear_model.LinearRegression"),
        "reg_ridge": _make_sklearn("sklearn.linear_model.Ridge"),
        "reg_lasso": _make_sklearn("sklearn.linear_model.Lasso"),
        "reg_elasticnet": _make_sklearn("sklearn.linear_model.ElasticNet"),
        "reg_bayesianridge": _make_sklearn("sklearn.linear_model.BayesianRidge"),
        "reg_decision_tree": _make_sklearn("sklearn.tree.DecisionTreeRegressor"),
        "reg_random_forest": _make_sklearn("sklearn.ensemble.RandomForestRegressor"),
        "reg_extra_trees": _make_sklearn("sklearn.ensemble.ExtraTreesRegressor"),
        "reg_gb": _make_sklearn("sklearn.ensemble.GradientBoostingRegressor"),
        "reg_adaboost": _make_sklearn("sklearn.ensemble.AdaBoostRegressor"),
        "reg_xgboost": _make_xgboost_reg,
        "reg_lightgbm": _make_lightgbm_reg,
        "reg_catboost": _make_catboost_reg,
        "reg_knn": _make_sklearn("sklearn.neighbors.KNeighborsRegressor"),
        "reg_svr": _make_sklearn("sklearn.svm.SVR"),
        "reg_mlp": _make_sklearn("sklearn.neural_network.MLPRegressor"),
    }

    # Classification factories
    clf_factories: Dict[str, Callable[[Dict[str, Any]], Any]] = {
        "clf_logistic": _make_sklearn("sklearn.linear_model.LogisticRegression"),
        "clf_ridge": _make_sklearn("sklearn.linear_model.RidgeClassifier"),
        "clf_decision_tree": _make_sklearn("sklearn.tree.DecisionTreeClassifier"),
        "clf_random_forest": _make_sklearn("sklearn.ensemble.RandomForestClassifier"),
        "clf_extra_trees": _make_sklearn("sklearn.ensemble.ExtraTreesClassifier"),
        "clf_gb": _make_sklearn("sklearn.ensemble.GradientBoostingClassifier"),
        "clf_adaboost": _make_sklearn("sklearn.ensemble.AdaBoostClassifier"),
        "clf_xgboost": _make_xgboost_clf,
        "clf_lightgbm": _make_lightgbm_clf,
        "clf_catboost": _make_catboost_clf,
        "clf_knn": _make_sklearn("sklearn.neighbors.KNeighborsClassifier"),
        "clf_svm": _make_sklearn("sklearn.svm.SVC"),
        "clf_naive_bayes": _make_sklearn("sklearn.naive_bayes.GaussianNB"),
        "clf_mlp": _make_sklearn("sklearn.neural_network.MLPClassifier"),
    }

    # Clustering factories
    clu_factories: Dict[str, Callable[[Dict[str, Any]], Any]] = {
        "clu_kmeans": _make_sklearn("sklearn.cluster.KMeans"),
        "clu_minibatch_kmeans": _make_sklearn("sklearn.cluster.MiniBatchKMeans"),
        "clu_kmedoids": _make_kmedoids,
        "clu_agglomerative": _make_sklearn("sklearn.cluster.AgglomerativeClustering"),
        "clu_divisive": _make_sklearn("sklearn.cluster.BisectingKMeans"),
        "clu_dbscan": _make_sklearn("sklearn.cluster.DBSCAN"),
        "clu_hdbscan": _make_hdbscan,
        "clu_optics": _make_sklearn("sklearn.cluster.OPTICS"),
        "clu_gmm": _make_sklearn("sklearn.mixture.GaussianMixture"),
        "clu_spectral": _make_sklearn("sklearn.cluster.SpectralClustering"),
    }

    # Dimensionality reduction factories
    dr_factories: Dict[str, Callable[[Dict[str, Any]], Any]] = {
        "dr_pca": _make_sklearn("sklearn.decomposition.PCA"),
        "dr_lda": _make_sklearn("sklearn.discriminant_analysis.LinearDiscriminantAnalysis"),
        "dr_tsne": _make_sklearn("sklearn.manifold.TSNE"),
        "dr_umap": _make_umap,
        # Autoencoder is a placeholder; will be unavailable unless torch is installed AND a wrapper is implemented.
        "dr_autoencoder": lambda _p: (_ for _ in ()).throw(RuntimeError("Autoencoder backend not implemented")),
    }

    for c in reg:
        if c.key not in reg_factories:
            continue
        registry[c.key] = AlgorithmSpec(card=c, factory=reg_factories[c.key])
    for c in clf:
        if c.key not in clf_factories:
            continue
        registry[c.key] = AlgorithmSpec(card=c, factory=clf_factories[c.key])
    for c in clu:
        if c.key not in clu_factories:
            continue
        registry[c.key] = AlgorithmSpec(card=c, factory=clu_factories[c.key])
    for c in dr:
        if c.key not in dr_factories:
            continue
        registry[c.key] = AlgorithmSpec(card=c, factory=dr_factories[c.key])

    return registry
