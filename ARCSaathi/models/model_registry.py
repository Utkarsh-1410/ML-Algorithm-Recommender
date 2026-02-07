"""Algorithm registry and model cards for ARCSaathi.

This module defines a compact, maintainable registry of representative
algorithms. The registry uses non-import checks for optional dependencies
so startup remains fast.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from typing import Any, Callable, Dict, List, Optional


@dataclass(frozen=True)
class HyperParamSpec:
    name: str
    default: Any
    kind: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    choices: Optional[List[Any]] = None


@dataclass(frozen=True)
class ModelCard:
    key: str
    name: str
    task_type: str
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
            if importlib.util.find_spec(dep) is None:
                return (False, f"Missing dependency: {dep}")
        return (True, "")


def _make_sklearn(cls_path: str) -> Callable[[Dict[str, Any]], Any]:
    def _f(params: Dict[str, Any]) -> Any:
        module_name, class_name = cls_path.rsplit(".", 1)
        mod = __import__(module_name, fromlist=[class_name])
        cls = getattr(mod, class_name)
        return cls(**params)

    return _f


def _make_umap(params: Dict[str, Any]) -> Any:
    import umap

    return umap.UMAP(**params)


def build_registry() -> Dict[str, AlgorithmSpec]:
    """Return a compact registry with 5/5/5/3 algorithms (reg/clf/clu/dr)."""

    registry: Dict[str, AlgorithmSpec] = {}

    # Regression (5)
    reg_specs = {
        "reg_linear": ("Linear Regression", "Baseline linear model for continuous targets.",
                       "Large numeric datasets; interpretability.", "Fast; interpretable; strong baseline.",
                       "Underfits non-linear patterns; sensitive to outliers.", "O(n·p²)", "Low",
                       "Good baseline.", [], _make_sklearn("sklearn.linear_model.LinearRegression"), []),

        "reg_ridge": ("Ridge Regression", "Linear regression with L2 regularization.",
                      "High-dimensional data; multicollinearity.", "Stable; reduces overfitting.",
                      "Still linear; requires tuning alpha.", "O(n·p²)", "Low",
                      "Often better than Linear on noisy data.", [HyperParamSpec("alpha", 1.0, "float", 1e-6, 1e3, 0.1)],
                      _make_sklearn("sklearn.linear_model.Ridge"), []),

        "reg_lasso": ("Lasso Regression", "Linear regression with L1 regularization.",
                      "Feature selection; sparse solutions.", "Can zero-out irrelevant features.",
                      "Can be unstable with correlated features.", "Varies", "Low",
                      "Good for sparse solutions.", [HyperParamSpec("alpha", 0.001, "float", 1e-6, 10.0, 0.001)],
                      _make_sklearn("sklearn.linear_model.Lasso"), []),

        "reg_decision_tree": ("Decision Tree Regressor", "Non-linear tree for regression.",
                               "Non-linear patterns; mixed feature types.", "Captures interactions; no scaling required.",
                               "Can overfit; unstable.", "O(n·p·log n)", "Low-Medium",
                               "Good with tuned depth.", [HyperParamSpec("max_depth", None, "int", 1, 64, 1)],
                               _make_sklearn("sklearn.tree.DecisionTreeRegressor"), []),

        "reg_random_forest": ("Random Forest Regressor", "Bagged ensemble of decision trees.",
                              "Strong general-purpose regression.", "Robust; handles non-linearities; less overfitting.",
                              "Less interpretable; slower.", "O(T·n·p·log n)", "Medium",
                              "Strong without heavy tuning.", [HyperParamSpec("n_estimators", 200, "int", 50, 2000, 50)],
                              _make_sklearn("sklearn.ensemble.RandomForestRegressor"), []),
    }

    for key, (name, desc, best_for, pros, cons, time_c, mem, perf, params, factory, deps) in reg_specs.items():
        card = ModelCard(key=key, name=name, task_type="regression", family="Mixed", description=desc,
                         best_for=best_for, pros=pros, cons=cons, time_complexity=time_c,
                         memory_usage=mem, expected_performance=perf, params=params, optional_deps=deps)
        registry[key] = AlgorithmSpec(card=card, factory=factory)

    # Classification (5)
    clf_specs = {
        "clf_logistic": ("Logistic Regression", "Linear classifier with probabilistic outputs.",
                         "Baseline classification; calibrated probabilities.", "Fast; interpretable; good baseline.",
                         "Linear decision boundary.", "O(n·p)", "Low",
                         "Strong baseline for many problems.", [HyperParamSpec("C", 1.0, "float", 1e-3, 1e3, 0.1)],
                         _make_sklearn("sklearn.linear_model.LogisticRegression"), []),

        "clf_random_forest": ("Random Forest Classifier", "Bagged ensemble of trees.",
                              "Strong general-purpose classifier.", "Robust; handles non-linearities.",
                              "Less interpretable; larger models.", "O(T·n·p·log n)", "Medium",
                              "Strong baseline.", [HyperParamSpec("n_estimators", 300, "int", 50, 3000, 50)],
                              _make_sklearn("sklearn.ensemble.RandomForestClassifier"), []),

        "clf_svm": ("SVM (SVC)", "Support Vector Classifier with kernels.", "Small-medium datasets.",
                    "Strong margins; flexible kernels.", "Does not scale well; tuning required.", "O(n³)", "Medium",
                    "Strong when tuned.", [HyperParamSpec("C", 1.0, "float", 1e-3, 1e3, 0.1)],
                    _make_sklearn("sklearn.svm.SVC"), []),

        "clf_knn": ("KNN Classifier", "k-nearest neighbors classifier.", "Small datasets; local decision boundaries.",
                    "Simple; non-parametric.", "Slow prediction; needs scaling.", "O(n) per query", "Medium",
                    "Good with scaling + k.", [HyperParamSpec("n_neighbors", 5, "int", 1, 200, 1)],
                    _make_sklearn("sklearn.neighbors.KNeighborsClassifier"), []),

        "clf_naive_bayes": ("Naive Bayes (Gaussian)", "Probabilistic classifier with independence assumption.",
                             "Fast baseline; continuous features.", "Very fast; robust baseline.",
                             "Independence assumption.", "O(n·p)", "Low",
                             "Good baseline.", [], _make_sklearn("sklearn.naive_bayes.GaussianNB"), []),
    }

    for key, (name, desc, best_for, pros, cons, time_c, mem, perf, params, factory, deps) in clf_specs.items():
        card = ModelCard(key=key, name=name, task_type="classification", family="Mixed", description=desc,
                         best_for=best_for, pros=pros, cons=cons, time_complexity=time_c,
                         memory_usage=mem, expected_performance=perf, params=params, optional_deps=deps)
        registry[key] = AlgorithmSpec(card=card, factory=factory)

    # Clustering (5)
    clu_specs = {
        "clu_kmeans": ("K-Means", "Centroid-based clustering.", "Spherical clusters; scaled numeric features.",
                       "Fast; simple.", "Needs k; sensitive to scaling/outliers.", "O(n·k·i)", "Low",
                       "Good baseline when k known.", [HyperParamSpec("n_clusters", 8, "int", 2, 200, 1)],
                       _make_sklearn("sklearn.cluster.KMeans"), []),

        "clu_dbscan": ("DBSCAN", "Density-based clustering.", "Arbitrary cluster shapes; outlier detection.",
                      "No need to specify k; finds density.", "Sensitive to parameters (eps, min_samples).", "O(n²)", "Low",
                      "Good for structured point clouds.", [HyperParamSpec("eps", 0.5, "float", 0.01, 10.0, 0.1)],
                      _make_sklearn("sklearn.cluster.DBSCAN"), []),

        "clu_agglomerative": ("Agglomerative Clustering", "Bottom-up hierarchical clustering.",
                              "Small-medium datasets.", "No need to specify clusters (can cut dendrogram).",
                              "Can be slow on large n.", "O(n²)", "High",
                              "Good for structured clusters.", [HyperParamSpec("n_clusters", 8, "int", 2, 200, 1)],
                              _make_sklearn("sklearn.cluster.AgglomerativeClustering"), []),

        "clu_gmm": ("Gaussian Mixture (GMM)", "Probabilistic clustering with Gaussians.",
                    "Elliptical clusters; soft assignments.", "Soft clustering; probabilistic.",
                    "Needs k; can converge poorly.", "O(n·k·i)", "Medium",
                    "Strong when Gaussian assumption holds.", [HyperParamSpec("n_components", 8, "int", 2, 200, 1)],
                    _make_sklearn("sklearn.mixture.GaussianMixture"), []),

        "clu_optics": ("OPTICS", "Density-based clustering similar to DBSCAN.",
                      "Varying density; cluster structure exploration.", "Less sensitive to eps.",
                      "Heavier than DBSCAN.", "O(n log n)", "Medium",
                      "Good for density structure discovery.", [HyperParamSpec("min_samples", 5, "int", 1, 100, 1)],
                      _make_sklearn("sklearn.cluster.OPTICS"), []),
    }

    for key, (name, desc, best_for, pros, cons, time_c, mem, perf, params, factory, deps) in clu_specs.items():
        card = ModelCard(key=key, name=name, task_type="clustering", family="Mixed", description=desc,
                         best_for=best_for, pros=pros, cons=cons, time_complexity=time_c,
                         memory_usage=mem, expected_performance=perf, params=params, optional_deps=deps)
        registry[key] = AlgorithmSpec(card=card, factory=factory)

    # Dimensionality reduction (3)
    dr_specs = {
        "dr_pca": ("PCA", "Principal Component Analysis.", "Dimensionality reduction; visualization.",
                   "Fast; interpretable components.", "Linear only; loses non-linear structure.", "O(n·p·min(n,p))",
                   "Medium", "Good for visualization and reduction.", [HyperParamSpec("n_components", 2, "int", 1, 256, 1)],
                   _make_sklearn("sklearn.decomposition.PCA"), []),

        "dr_tsne": ("t-SNE", "Nonlinear embedding for visualization.", "2D/3D visualization.",
                    "Great visualization.", "Slow; not for downstream modeling.", "High", "High",
                    "Excellent for visual separation.", [HyperParamSpec("n_components", 2, "int", 2, 3, 1)],
                    _make_sklearn("sklearn.manifold.TSNE"), []),

        "dr_umap": ("UMAP", "Nonlinear embedding often faster than t-SNE.", "Visualization and embeddings.",
                    "Fast; preserves structure well.", "Optional dependency.", "Medium", "Medium",
                    "Strong embeddings.", [HyperParamSpec("n_components", 2, "int", 2, 200, 1)],
                    _make_umap, ["umap"]),
    }

    for key, (name, desc, best_for, pros, cons, time_c, mem, perf, params, factory, deps) in dr_specs.items():
        card = ModelCard(key=key, name=name, task_type="dimred", family="Mixed", description=desc,
                         best_for=best_for, pros=pros, cons=cons, time_complexity=time_c,
                         memory_usage=mem, expected_performance=perf, params=params, optional_deps=deps)
        registry[key] = AlgorithmSpec(card=card, factory=factory)

    return registry
