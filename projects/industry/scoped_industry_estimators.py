"""Reusable fitted estimators for the group-restricted five-industry rerun.

The caller owns as-of filtering, group restrictions, transformation, imputation,
standardisation and chronological tuning.  This module receives only the current
training window.  Every forecast, attribution and scenario must use the returned
object's predict method; no separate explanatory surrogate is fitted.

Reliability here preserves the earlier AdaLASSO training-residual shrinkage
formula.  Its weights are train-only, but are not out-of-fold success estimates.
AdaLASSO_Reliability_Strict requires complete-case data supplied by the caller.
All outputs are in the same transformed outcome units as the supplied y.
"""
from __future__ import annotations

import math

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                              HistGradientBoostingRegressor, RandomForestRegressor)
from sklearn.linear_model import (BayesianRidge, ElasticNet, HuberRegressor,
                                  Lasso, LogisticRegression, Ridge)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer
from sklearn.svm import SVR

CORE_COLUMNS = ("AR1", "AR2", "AR3", "AR6", "AR12", "SIN1", "COS1", "SIN2", "COS2")
RANDOM_STATE = 20260901
MODEL_NAMES = [
    "AR_Fourier_Ridge", "Ridge", "Elastic_Net", "Sparse_Group_LASSO",
    "Bayesian_Ridge", "PLS", "RBF_SVR", "Random_Forest", "Extra_Trees",
    "Gradient_Boosting", "Hist_Gradient_Boosting", "AdaLASSO",
    "AdaLASSO_Leverage", "AdaLASSO_GroupOne", "AdaLASSO_Reliability",
    "AdaLASSO_Reliability_Strict", "Central_AR_AdaReliability", "XGBoost",
    "LightGBM", "CatBoost", "Spline_GAM", "TwoStage_Momentum_Huber",
    "Quantile_GB_Median",
]

PARAM_GRIDS = {
    "AR_Fourier_Ridge": [{"alpha": 10.0}, {"alpha": 100.0}],
    "Ridge": [{"alpha": 10.0}, {"alpha": 100.0}],
    "Elastic_Net": [{"alpha": 0.001, "l1_ratio": 0.5}, {"alpha": 0.005, "l1_ratio": 0.5}],
    "Sparse_Group_LASSO": [{"lambda": 0.001, "mix": 0.5}, {"lambda": 0.005, "mix": 0.5}],
    "Bayesian_Ridge": [{}],
    "PLS": [{"components": 2}, {"components": 4}],
    "RBF_SVR": [{"C": 0.5, "epsilon": 0.01}, {"C": 2.0, "epsilon": 0.01}],
    "Random_Forest": [{"max_depth": 3, "min_leaf": 4}, {"max_depth": 5, "min_leaf": 5}],
    "Extra_Trees": [{"max_depth": 3, "min_leaf": 4}, {"max_depth": 5, "min_leaf": 5}],
    "Gradient_Boosting": [{"n_estimators": 50, "max_depth": 1}, {"n_estimators": 70, "max_depth": 2}],
    "Hist_Gradient_Boosting": [{"max_iter": 70, "max_leaf_nodes": 5}, {"max_iter": 80, "max_leaf_nodes": 7}],
    "XGBoost": [
        {"n_estimators": 100, "max_depth": 1, "learning_rate": 0.04, "min_child_weight": 6},
        {"n_estimators": 140, "max_depth": 2, "learning_rate": 0.035, "min_child_weight": 8}],
    "LightGBM": [
        {"n_estimators": 100, "max_depth": 2, "num_leaves": 4, "min_child_samples": 15},
        {"n_estimators": 140, "max_depth": 3, "num_leaves": 6, "min_child_samples": 18}],
    "CatBoost": [{"iterations": 110, "depth": 2, "l2_leaf_reg": 8},
                 {"iterations": 150, "depth": 3, "l2_leaf_reg": 10}],
    "Spline_GAM": [{"n_knots": 4, "degree": 2, "alpha": 20.0},
                   {"n_knots": 5, "degree": 2, "alpha": 60.0}],
    "TwoStage_Momentum_Huber": [{"threshold": 0.005, "C": 0.15, "magnitude_alpha": 0.01},
                                {"threshold": 0.015, "C": 0.5, "magnitude_alpha": 0.01}],
    "Quantile_GB_Median": [{"n_estimators": 70, "max_depth": 1},
                           {"n_estimators": 90, "max_depth": 2}],
}
_ADA_GRID = [{"ridge_alpha": 10.0, "lasso_alpha": a, "gamma": 1.0}
             for a in (0.0005, 0.003, 0.01)]
for _model_name in MODEL_NAMES:
    if _model_name.startswith("AdaLASSO") or _model_name == "Central_AR_AdaReliability":
        PARAM_GRIDS[_model_name] = [dict(p) for p in _ADA_GRID]


def _screen(x, y, k):
    if x.shape[1] <= k:
        return np.arange(x.shape[1])
    score = np.abs(x.T.dot(y - y.mean())) / len(y)
    return np.sort(np.argsort(score, kind="mergesort")[-k:])


class _SubsetPredictor:
    def __init__(self, estimator, indexes):
        self.estimator = estimator
        self.indexes = np.asarray(indexes, dtype=int)

    def predict(self, x):
        return np.asarray(self.estimator.predict(x[:, self.indexes]), dtype=float).reshape(-1)


class _LinearPredictor:
    def __init__(self, intercept, coef, reliability=None, caps=None, lower=None, upper=None):
        self.intercept_ = float(intercept)
        self.coef_ = np.asarray(coef, dtype=float)
        self.reliability_ = np.ones_like(self.coef_) if reliability is None else reliability
        self.caps_ = np.full_like(self.coef_, np.inf) if caps is None else caps
        self.lower_ = lower
        self.upper_ = upper

    def contributions(self, x):
        if self.lower_ is not None:
            x = np.clip(x, self.lower_, self.upper_)
        raw = x * self.coef_
        return np.clip(raw, -self.caps_, self.caps_) * self.reliability_

    def predict(self, x):
        return self.intercept_ + self.contributions(x).sum(axis=1)


def _ada(x, y, names, groups, params, variant):
    lower = upper = None
    if variant == "AdaLASSO_Leverage":
        lower, upper = np.quantile(x, [0.01, 0.99], axis=0)
        x = np.clip(x, lower, upper)
    ridge = Ridge(alpha=float(params.get("ridge_alpha", 10.0))).fit(x, y)
    weights = 1.0 / (np.abs(ridge.coef_) + 1e-3) ** float(params.get("gamma", 1.0))
    core = np.array([name in CORE_COLUMNS for name in names])
    weights[core] = np.minimum(weights[core], 0.25)
    lasso = Lasso(alpha=float(params.get("lasso_alpha", 0.003)), max_iter=10000,
                  tol=1e-5, selection="cyclic", random_state=RANDOM_STATE)
    lasso.fit(x / weights, y)
    coef = np.asarray(lasso.coef_) / weights
    reliability = np.ones_like(coef)
    caps = np.full_like(coef, np.inf)
    intercept = float(lasso.intercept_)
    if variant == "AdaLASSO_GroupOne":
        keep = core.copy()
        for group in sorted(set(groups)):
            indexes = [j for j, label in enumerate(groups) if label == group and not core[j]
                       and abs(coef[j]) > 1e-9]
            if indexes:
                keep[max(indexes, key=lambda j: abs(coef[j]))] = True
        coef = np.where(keep, coef, 0.0)
        intercept = float(np.mean(y - x.dot(coef)))
    if variant in ("AdaLASSO_Reliability", "AdaLASSO_Reliability_Strict", "AdaLASSO_Leverage"):
        residual = y - intercept - x.dot(coef)
        recency = float(params.get("rho", 0.8)) ** np.arange(len(y) - 1, -1, -1)
        for j in np.where((~core) & (np.abs(coef) > 1e-9))[0]:
            signal = x[:, j] * coef[j]
            prior = float(params.get("prior_n", 8.0)) * max(float(np.median(signal ** 2)), 1e-8)
            num = prior + float(np.sum(recency * signal * (residual + signal)))
            den = prior + float(np.sum(recency * signal ** 2))
            reliability[j] = float(np.clip(num / max(den, 1e-12), 0.0, 1.0))
            caps[j] = float(np.quantile(np.abs(signal), 0.90))
        train_contributions = np.clip(x * coef, -caps, caps) * reliability
        intercept = float(np.mean(y - train_contributions.sum(axis=1)))
    return _LinearPredictor(intercept, coef, reliability, caps, lower, upper)


def _sparse_group(x, y, groups, params):
    """FISTA for squared loss plus L1 and sqrt(group-size) group-L2 penalties."""
    n, p = x.shape
    center = x.mean(axis=0)
    xc, yc = x - center, y - y.mean()
    group_indexes = [np.where(np.asarray(groups) == label)[0] for label in sorted(set(groups))]
    gram = xc.T.dot(xc) / n
    cross = xc.T.dot(yc) / n
    # Gershgorin bound is deterministic and avoids repeated costly SVDs.
    lipschitz = max(float(np.max(np.sum(np.abs(gram), axis=1))), 1e-8)
    step = 1.0 / lipschitz
    lam, mix = float(params.get("lambda", 0.003)), float(params.get("mix", 0.5))
    beta, momentum = np.zeros(p), np.zeros(p)
    t = 1.0
    converged = False
    for iteration in range(int(params.get("max_iter", 1500))):
        z = momentum - step * (gram.dot(momentum) - cross)
        z = np.sign(z) * np.maximum(np.abs(z) - step * lam * mix, 0.0)
        for indexes in group_indexes:
            norm = float(np.linalg.norm(z[indexes]))
            if norm > 0.0:
                z[indexes] *= max(0.0, 1.0 - step * lam * (1 - mix) * math.sqrt(len(indexes)) / norm)
        change = float(np.max(np.abs(z - beta)))
        next_t = (1.0 + math.sqrt(1.0 + 4.0 * t * t)) / 2.0
        momentum = z + ((t - 1.0) / next_t) * (z - beta)
        beta, t = z, next_t
        if change < 1e-7 and iteration >= 30:
            converged = True
            break
    model = _LinearPredictor(float(y.mean() - center.dot(beta)), beta)
    model.converged_ = converged
    model.n_iter_ = iteration + 1
    return model


class _CentralPredictor:
    def __init__(self, base, full_ada, core, cap, volatility):
        self.base, self.full_ada = base, full_ada
        self.core = core
        self.cap, self.volatility = cap, volatility

    def predict(self, x):
        baseline = self.base.predict(x[:, self.core])
        gap = self.full_ada.predict(x) - baseline
        return baseline + np.clip(gap, -self.cap, self.cap) * self.volatility


class EnsemblePredictor:
    """Portable saved ensemble with member-specific train-fitted preprocessing.

    Input scenarios use the soft prepared design as their reference. A strict
    member receives only its own retained columns, mapped to its own centering
    and scale. Defining this class in an importable module avoids __main__ pickle
    references when forecasts are produced by a command-line worker.
    """
    def __init__(self, fitted, prepared, weights):
        self.fitted, self.prepared, self.weights = fitted, prepared, weights
        self.reference = prepared[False]
        self.reliability_ = None

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        raw = x * self.reference.scales + self.reference.centers
        prediction = np.zeros(len(x))
        for model, weight in self.weights.items():
            p = self.prepared[model == "AdaLASSO_Reliability_Strict"]
            indexes = [self.reference.names.index(n) for n in p.names]
            mapped = (raw[:, indexes] - p.centers) / p.scales
            prediction += weight * self.fitted[model].predict(mapped)
        return prediction


class _TwoStagePredictor:
    def __init__(self, classifier, constant_class, magnitude, indexes, previous, cap):
        self.classifier, self.constant_class = classifier, constant_class
        self.magnitude, self.indexes = magnitude, indexes
        self.previous, self.cap = previous, cap

    def predict(self, x):
        z = x[:, self.indexes]
        label = (np.full(len(x), self.constant_class) if self.classifier is None
                 else self.classifier.predict(z))
        # Capping in log space is equivalent and avoids numerical overflow.
        magnitude = np.exp(np.minimum(self.magnitude.predict(z), math.log(self.cap)))
        return self.previous + label * magnitude


def _boost(name, params):
    if name == "XGBoost":
        from xgboost import XGBRegressor
        return XGBRegressor(objective="reg:squarederror", n_estimators=int(params.get("n_estimators", 100)),
                            max_depth=int(params.get("max_depth", 2)), learning_rate=float(params.get("learning_rate", 0.04)),
                            min_child_weight=float(params.get("min_child_weight", 6)), subsample=0.8,
                            colsample_bytree=0.65, reg_alpha=0.03, reg_lambda=8.0,
                            random_state=RANDOM_STATE, n_jobs=1, verbosity=0)
    if name == "LightGBM":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(objective="regression", n_estimators=int(params.get("n_estimators", 100)),
                             max_depth=int(params.get("max_depth", 2)), num_leaves=int(params.get("num_leaves", 4)),
                             learning_rate=float(params.get("learning_rate", 0.04)),
                             min_child_samples=int(params.get("min_child_samples", 15)), subsample=0.8,
                             colsample_bytree=0.65, reg_alpha=0.03, reg_lambda=8.0,
                             random_state=RANDOM_STATE, n_jobs=1, verbosity=-1)
    from catboost import CatBoostRegressor
    return CatBoostRegressor(loss_function="RMSE", iterations=int(params.get("iterations", 110)),
                             depth=int(params.get("depth", 2)), learning_rate=float(params.get("learning_rate", 0.04)),
                             l2_leaf_reg=float(params.get("l2_leaf_reg", 8)), random_seed=RANDOM_STATE,
                             verbose=False, allow_writing_files=False, thread_count=1)


class FittedEstimator:
    """Checked prediction interface; metadata stays aligned to all input columns."""
    def __init__(self, name, delegate, names, groups, params, selected=None):
        self.model_name_, self.delegate = name, delegate
        self.names_, self.groups_, self.params_ = list(names), list(groups), dict(params)
        self.n_features_in_ = len(names)
        self.selected_ = np.ones(len(names), dtype=bool) if selected is None else np.asarray(selected, dtype=bool)
        self.reliability_ = np.full(len(names), np.nan)
        self.caps_ = np.full(len(names), np.nan)
        if isinstance(delegate, _LinearPredictor):
            self.coef_ = delegate.coef_.copy()
            self.selected_ = np.abs(self.coef_) > 1e-9
            self.reliability_ = delegate.reliability_.copy()
            self.caps_ = delegate.caps_.copy()
        if isinstance(delegate, _CentralPredictor):
            self.reliability_ = delegate.full_ada.reliability_.copy()
            self.caps_ = delegate.full_ada.caps_.copy()
            self.selected_ = np.abs(delegate.full_ada.coef_) > 1e-9
            self.selected_[delegate.core] = True
        if isinstance(delegate, _SubsetPredictor) and hasattr(delegate.estimator, "coef_"):
            coef = np.asarray(delegate.estimator.coef_).reshape(-1)
            if len(coef) == len(delegate.indexes):
                self.coef_ = np.zeros(len(names))
                self.coef_[delegate.indexes] = coef
                self.selected_ = np.abs(self.coef_) > 1e-9

    def predict(self, X):
        x = np.asarray(X, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.n_features_in_:
            raise ValueError("predict expects a 2D matrix with the fitted feature ordering")
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite prediction features: caller must apply train-fitted preprocessing")
        values = np.asarray(self.delegate.predict(x), dtype=float).reshape(-1)
        if len(values) != len(x) or not np.isfinite(values).all():
            raise FloatingPointError("Nonfinite or malformed predictions from " + self.model_name_)
        return values


def feature_reliability(model):
    """Per-feature diagnostics of this exact fitted model; NaN means not applicable."""
    return [{"feature": name, "group": model.groups_[j], "selected": bool(model.selected_[j]),
             "reliability": float(model.reliability_[j]), "signal_cap": float(model.caps_[j])}
            for j, name in enumerate(model.names_)]


def fit_model(name, X, y, names, groups, params=None):
    """Fit using current-window rows only and return a reusable predictor.

    names/groups must follow X column order.  No fit exception is replaced by an
    unrelated baseline.  The caller should record any failure and exclude that
    candidate transparently from an otherwise comparable evaluation.
    """
    if name not in MODEL_NAMES:
        raise ValueError("Unknown model: " + name)
    x, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float).reshape(-1)
    params = dict(PARAM_GRIDS[name][0] if params is None else params)
    if x.ndim != 2 or len(x) != len(y) or len(y) < 8 or x.shape[1] != len(names) or len(groups) != len(names):
        raise ValueError("Incompatible training matrix, labels or feature metadata")
    if x.shape[1] == 0 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Training data must be finite with at least one feature")
    indexes = np.arange(x.shape[1])
    if name.startswith("AdaLASSO"):
        delegate = _ada(x, y, names, groups, params, name)
    elif name == "Sparse_Group_LASSO":
        delegate = _sparse_group(x, y, groups, params)
    elif name == "Central_AR_AdaReliability":
        core = np.where([n in CORE_COLUMNS for n in names])[0]
        external = np.where([n not in CORE_COLUMNS for n in names])[0]
        if not len(core) or not len(external):
            raise ValueError("Central model requires both core and external features")
        baseline = Ridge(alpha=float(params.get("base_alpha", 10.0))).fit(x[:, core], y)
        full_ada = _ada(x, y, names, groups, params, "AdaLASSO_Reliability")
        historical_gap = full_ada.predict(x) - baseline.predict(x[:, core])
        cap = float(np.quantile(np.abs(historical_gap), 0.8))
        volatility = float(np.clip(np.std(y[-12:], ddof=1) / max(np.std(y, ddof=1), 1e-8), 0.65, 1.35))
        delegate = _CentralPredictor(baseline, full_ada, core, cap, volatility)
    elif name == "TwoStage_Momentum_Huber":
        indexes = _screen(x, y, 35)
        delta = np.diff(y)
        z = x[1:, indexes]
        threshold = float(params.get("threshold", 0.005))
        labels = np.where(delta > threshold, 1, np.where(delta < -threshold, -1, 0))
        classifier, constant = None, int(labels[-1])
        if len(np.unique(labels)) > 1:
            classifier = LogisticRegression(C=float(params.get("C", 0.15)), class_weight="balanced",
                                            max_iter=2000, random_state=RANDOM_STATE).fit(z, labels)
        magnitude = np.maximum(np.abs(delta), 1e-4)
        magnitude_model = HuberRegressor(epsilon=1.35, alpha=float(params.get("magnitude_alpha", 0.01)),
                                         max_iter=2000).fit(z, np.log(magnitude))
        delegate = _TwoStagePredictor(classifier, constant, magnitude_model, indexes,
                                      float(y[-1]), float(np.quantile(magnitude, 0.9)))
    else:
        if name == "AR_Fourier_Ridge":
            indexes = np.where([n in CORE_COLUMNS for n in names])[0]
            if not len(indexes):
                raise ValueError("AR_Fourier_Ridge requires named core columns")
            estimator = Ridge(alpha=float(params.get("alpha", 10.0)))
        elif name == "Ridge":
            estimator = Ridge(alpha=float(params.get("alpha", 10.0)))
        elif name == "Elastic_Net":
            estimator = ElasticNet(alpha=float(params.get("alpha", 0.001)), l1_ratio=float(params.get("l1_ratio", 0.5)),
                                   max_iter=10000, tol=1e-5, selection="cyclic", random_state=RANDOM_STATE)
        elif name == "Bayesian_Ridge":
            estimator = BayesianRidge()
        elif name == "PLS":
            indexes = _screen(x, y, 30)
            estimator = PLSRegression(n_components=min(int(params.get("components", 2)), len(indexes), len(y) - 1),
                                      scale=False, max_iter=1000)
        elif name == "RBF_SVR":
            indexes = _screen(x, y, 35)
            estimator = SVR(C=float(params.get("C", 0.5)), epsilon=float(params.get("epsilon", 0.01)),
                            gamma=params.get("gamma", "scale"))
        elif name in ("Random_Forest", "Extra_Trees"):
            indexes = _screen(x, y, 40)
            cls = RandomForestRegressor if name == "Random_Forest" else ExtraTreesRegressor
            estimator = cls(n_estimators=int(params.get("n_estimators", 50)), max_depth=int(params.get("max_depth", 3)),
                             min_samples_leaf=int(params.get("min_leaf", 4)), max_features=float(params.get("max_features", 0.5)),
                             random_state=RANDOM_STATE, n_jobs=1)
        elif name in ("Gradient_Boosting", "Quantile_GB_Median"):
            indexes = _screen(x, y, 35)
            quantile = name == "Quantile_GB_Median"
            estimator = GradientBoostingRegressor(loss="quantile" if quantile else "huber",
                alpha=float(params.get("quantile", 0.5)) if quantile else 0.9,
                n_estimators=int(params.get("n_estimators", 70)), learning_rate=float(params.get("learning_rate", 0.035 if quantile else 0.04)),
                max_depth=int(params.get("max_depth", 1)), min_samples_leaf=5 if quantile else 4, random_state=RANDOM_STATE)
        elif name == "Hist_Gradient_Boosting":
            indexes = _screen(x, y, 35)
            estimator = HistGradientBoostingRegressor(learning_rate=float(params.get("learning_rate", 0.04)),
                max_iter=int(params.get("max_iter", 70)), max_leaf_nodes=int(params.get("max_leaf_nodes", 5)),
                min_samples_leaf=8, l2_regularization=1.0, early_stopping=False, random_state=RANDOM_STATE)
        elif name in ("XGBoost", "LightGBM", "CatBoost"):
            indexes = _screen(x, y, 35)
            estimator = _boost(name, params)
        elif name == "Spline_GAM":
            indexes = _screen(x, y, 10)
            estimator = make_pipeline(SplineTransformer(n_knots=int(params.get("n_knots", 4)),
                degree=int(params.get("degree", 2)), include_bias=False), Ridge(alpha=float(params.get("alpha", 20.0))))
        else:
            raise ValueError("Unimplemented model: " + name)
        estimator.fit(x[:, indexes], y)
        delegate = _SubsetPredictor(estimator, indexes)
    selected = np.zeros(len(names), dtype=bool)
    selected[indexes] = True
    result = FittedEstimator(name, delegate, names, groups, params, selected)
    result.predict(x[:2])  # Detect malformed fits immediately rather than during explanation.
    return result
