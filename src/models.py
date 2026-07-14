from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import (
    AdaBoostRegressor,
    BaggingRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.linear_model import (
    ARDRegression,
    BayesianRidge,
    ElasticNet,
    HuberRegressor,
    Lasso,
    LinearRegression,
    OrthogonalMatchingPursuit,
    PassiveAggressiveRegressor,
    Ridge,
    SGDRegressor,
)
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.cross_decomposition import PLSRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import LinearSVR, NuSVR, SVR

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None


class AdditivityGroupResidual(BaseEstimator, RegressorMixin):
    """Oxide-additivity baseline with a per-subgroup residual correction.

    Stage 1 fits ordinary least squares on the element columns (the additivity
    part: G0/H0 of cement phases are nearly linear in composition). Stage 2
    corrects the prediction with the mean training residual of the phase's
    subgroup, taken from the trailing one-hot group columns; in ``per_atom``
    mode the residual is normalised by formula-unit size before averaging.
    """

    def __init__(self, n_group_cols: int = 0, mode: str = "per_atom", min_group_rows: int = 2):
        self.n_group_cols = n_group_cols
        self.mode = mode
        self.min_group_rows = min_group_rows

    def _split(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=float)
        if self.n_group_cols <= 0:
            return X, np.zeros((len(X), 0))
        return X[:, : -self.n_group_cols], X[:, -self.n_group_cols:]

    def fit(self, X, y):
        from sklearn.linear_model import LinearRegression

        elements, groups = self._split(X)
        y = np.asarray(y, dtype=float)
        self.base_ = LinearRegression().fit(elements, y)
        residuals = y - self.base_.predict(elements)
        atoms = np.maximum(elements.sum(axis=1, keepdims=True), 1e-9)
        self.corrections_ = {}
        for index in range(groups.shape[1]):
            mask = groups[:, index] > 0.5
            if mask.sum() >= self.min_group_rows:
                if self.mode == "per_atom":
                    self.corrections_[index] = (residuals[mask] / atoms[mask]).mean(axis=0)
                else:
                    self.corrections_[index] = residuals[mask].mean(axis=0)
        return self

    def predict(self, X):
        elements, groups = self._split(X)
        prediction = self.base_.predict(elements)
        atoms = np.maximum(elements.sum(axis=1, keepdims=True), 1e-9)
        for row in range(len(elements)):
            if groups.shape[1] == 0:
                continue
            index = int(np.argmax(groups[row]))
            if groups[row, index] > 0.5 and index in self.corrections_:
                correction = self.corrections_[index]
                if self.mode == "per_atom":
                    correction = correction * atoms[row]
                prediction[row] = prediction[row] + correction
        return prediction


def candidate_models(n_rows: int, model_set: str = "expanded", n_group_cols: int = 0) -> dict[str, object]:
    k3 = min(3, n_rows)
    k5 = min(5, n_rows)
    k9 = min(9, n_rows)
    k15 = min(15, n_rows)

    fast = {
        "IDW kNN k=3": Pipeline([
            ("scale", StandardScaler()),
            ("model", KNeighborsRegressor(n_neighbors=k3, weights="distance")),
        ]),
        "IDW kNN k=5": Pipeline([
            ("scale", StandardScaler()),
            ("model", KNeighborsRegressor(n_neighbors=k5, weights="distance")),
        ]),
        "kNN mean k=5": Pipeline([
            ("scale", StandardScaler()),
            ("model", KNeighborsRegressor(n_neighbors=k5, weights="uniform")),
        ]),
        "IDW kNN k=9": Pipeline([
            ("scale", StandardScaler()),
            ("model", KNeighborsRegressor(n_neighbors=k9, weights="distance")),
        ]),
        "IDW kNN k=15": Pipeline([
            ("scale", StandardScaler()),
            ("model", KNeighborsRegressor(n_neighbors=k15, weights="distance")),
        ]),
        "Linear regression": Pipeline([
            ("scale", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "Ridge alpha=1": Pipeline([
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]),
        "Ridge alpha=10": Pipeline([
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ]),
        "Quadratic ridge alpha=10": Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ]),
        "Quadratic ridge alpha=100": Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=100.0)),
        ]),
        "PLS regression 2": Pipeline([
            ("scale", StandardScaler()),
            ("model", PLSRegression(n_components=min(2, n_rows - 1))),
        ]),
        "PLS regression 5": Pipeline([
            ("scale", StandardScaler()),
            ("model", PLSRegression(n_components=min(5, n_rows - 1))),
        ]),
        "Random forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=2,
            random_state=42,
        ),
        "Extra trees": ExtraTreesRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
        ),
        "Bagging trees": BaggingRegressor(
            n_estimators=120,
            random_state=42,
        ),
        "Gradient boosting": MultiOutputRegressor(GradientBoostingRegressor(
            n_estimators=120,
            learning_rate=0.05,
            max_depth=2,
            min_samples_leaf=2,
            random_state=42,
        )),
        "Hist gradient boosting": MultiOutputRegressor(HistGradientBoostingRegressor(
            max_iter=150,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=0.1,
            random_state=42,
        )),
        "SVR rbf": Pipeline([
            ("scale", StandardScaler()),
            ("model", MultiOutputRegressor(SVR(C=10.0, epsilon=0.05, gamma="scale"))),
        ]),
        "Kernel ridge rbf": Pipeline([
            ("scale", StandardScaler()),
            ("model", KernelRidge(alpha=1.0, kernel="rbf", gamma=0.1)),
        ]),
    }

    fast["Additivity OLS"] = AdditivityGroupResidual(n_group_cols=n_group_cols, mode="none", min_group_rows=10**9)
    if n_group_cols > 0:
        fast["Additivity + group residual"] = AdditivityGroupResidual(n_group_cols=n_group_cols, mode="mean")
        fast["Additivity + group residual/atom"] = AdditivityGroupResidual(n_group_cols=n_group_cols, mode="per_atom")

    if model_set == "fast":
        return fast

    expanded = {
        **fast,
        "ElasticNet alpha=0.001": Pipeline([
            ("scale", StandardScaler()),
            ("model", ElasticNet(alpha=0.001, l1_ratio=0.3, max_iter=20000, random_state=42)),
        ]),
        "ElasticNet alpha=0.01": Pipeline([
            ("scale", StandardScaler()),
            ("model", ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=20000, random_state=42)),
        ]),
        "Lasso alpha=0.001": Pipeline([
            ("scale", StandardScaler()),
            ("model", Lasso(alpha=0.001, max_iter=20000, random_state=42)),
        ]),
        "Bayesian ridge": Pipeline([
            ("scale", StandardScaler()),
            ("model", MultiOutputRegressor(BayesianRidge())),
        ]),
        "ARD regression": Pipeline([
            ("scale", StandardScaler()),
            ("model", MultiOutputRegressor(ARDRegression())),
        ]),
        "Huber regression": Pipeline([
            ("scale", StandardScaler()),
            ("model", MultiOutputRegressor(HuberRegressor(max_iter=1000))),
        ]),
        "Linear SVR": Pipeline([
            ("scale", StandardScaler()),
            ("model", MultiOutputRegressor(LinearSVR(C=1.0, epsilon=0.0, max_iter=10000, random_state=42))),
        ]),
        "NuSVR rbf": Pipeline([
            ("scale", StandardScaler()),
            ("model", MultiOutputRegressor(NuSVR(C=3.0, nu=0.5, gamma="scale"))),
        ]),
        "AdaBoost trees": MultiOutputRegressor(AdaBoostRegressor(
            n_estimators=120,
            learning_rate=0.05,
            random_state=42,
        )),
        "Regressor chain ridge": RegressorChain(
            Ridge(alpha=10.0),
            order=[0, 1, 2, 3, 4],
        ),
        "MLP small": Pipeline([
            ("scale", StandardScaler()),
            ("model", MLPRegressor(
                hidden_layer_sizes=(48, 24),
                alpha=0.01,
                learning_rate_init=0.003,
                max_iter=3000,
                early_stopping=True,
                random_state=42,
            )),
        ]),
    }

    if XGBRegressor is not None:
        expanded.update({
            "XGBoost conservative": XGBRegressor(
                n_estimators=200,
                max_depth=2,
                learning_rate=0.03,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.0,
                reg_lambda=10.0,
                objective="reg:squarederror",
                tree_method="hist",
                random_state=42,
            ),
            "XGBoost shallow": XGBRegressor(
                n_estimators=120,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_alpha=0.0,
                reg_lambda=5.0,
                objective="reg:squarederror",
                tree_method="hist",
                random_state=42,
            ),
            "XGBoost regularized": XGBRegressor(
                n_estimators=300,
                max_depth=2,
                learning_rate=0.02,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=20.0,
                objective="reg:squarederror",
                tree_method="hist",
                random_state=42,
            ),
        })

    if model_set == "expanded":
        return expanded

    if model_set != "full":
        raise ValueError("model_set must be one of: fast, expanded, full")

    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
    full = {
        **expanded,
        "Orthogonal matching pursuit": Pipeline([
            ("scale", StandardScaler()),
            ("model", MultiOutputRegressor(OrthogonalMatchingPursuit(n_nonzero_coefs=8))),
        ]),
        "Passive aggressive": Pipeline([
            ("scale", StandardScaler()),
            ("model", MultiOutputRegressor(PassiveAggressiveRegressor(
                C=0.1,
                max_iter=5000,
                random_state=42,
            ))),
        ]),
        "SGD elastic net": Pipeline([
            ("scale", StandardScaler()),
            ("model", MultiOutputRegressor(SGDRegressor(
                penalty="elasticnet",
                alpha=0.0001,
                l1_ratio=0.15,
                max_iter=5000,
                random_state=42,
            ))),
        ]),
        "Gaussian process rbf": Pipeline([
            ("scale", StandardScaler()),
            ("model", GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                normalize_y=True,
                random_state=42,
            )),
        ]),
    }
    return full
