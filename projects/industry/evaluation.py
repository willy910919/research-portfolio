"""Actual runner evaluation definitions; no data loading or runner side effects."""
import numpy as np
import pandas as pd

NEW_MODELS = ["XGBoost", "LightGBM", "CatBoost", "Spline_GAM", "TwoStage_Momentum_Huber", "Quantile_GB_Median", "Dynamic_Ensemble"]

ENSEMBLE_MODELS = ["Bayesian_Ridge", "Elastic_Net", "AdaLASSO_Reliability_Strict", "Gradient_Boosting"] + NEW_MODELS[:-1]

WINDOWS = {"Long_2015_2023": ("2015-01", "2023-12"), "Short_2024_2026": ("2024-01", "2026-06")}

METRICS_VERSION = "observed_prior_momentum_turn_f1_v2"

def ensemble_weights(history):
    if len(history) < 4:
        return {m: 1./len(ENSEMBLE_MODELS) for m in ENSEMBLE_MODELS}
    hist = pd.DataFrame(history[-8:])
    scores = []
    for model in ENSEMBLE_MODELS:
        pred, actual, prev = (hist[model+"_growth"].to_numpy(), hist.Actual_growth.to_numpy(), hist.Previous_growth.to_numpy())
        scores.append(float(np.sqrt(np.mean((pred-actual)**2)) + .015*np.mean(np.sign(pred)!=np.sign(actual)) +
                            .015*np.mean(np.sign(pred-prev)!=np.sign(actual-prev))))
    scores = np.asarray(scores)
    w = np.exp(-(scores-scores.min()) / max(float(np.median(scores)), .01))
    w /= w.sum()
    return dict(zip(ENSEMBLE_MODELS, w))

def metric(frame, model, window):
    start, end = WINDOWS[window]
    block = frame.loc[start:end]
    actual, pred = block.Actual_growth.to_numpy(), block[model+"_growth"].to_numpy()
    previous = block.Previous_growth.to_numpy()
    if not np.isfinite(pred).all():
        raise ValueError("Nonfinite backtest predictions for %s" % model)
    am, pm = actual-previous, pred-previous
    adir, pdir = np.sign(am), np.sign(pm)
    # A current turn is judged against the last OBSERVED momentum, which was
    # known at the forecast origin. Do not compare two old forecast directions.
    # Build the lag on the complete path before slicing the evaluation window:
    # the recent window can use December 2023's observed prior momentum.
    prior_momentum = frame.Previous_growth.diff().reindex(block.index).to_numpy()
    valid_turn = np.isfinite(prior_momentum) & np.isfinite(am) & np.isfinite(pm)
    turns = np.sign(am[valid_turn]) != np.sign(prior_momentum[valid_turn])
    pturns = np.sign(pm[valid_turn]) != np.sign(prior_momentum[valid_turn])
    true_positive = int(np.sum(turns & pturns))
    false_positive = int(np.sum(~turns & pturns))
    false_negative = int(np.sum(turns & ~pturns))
    true_negative = int(np.sum(~turns & ~pturns))
    f1_denominator = 2 * true_positive + false_positive + false_negative
    err = pred-actual
    material = np.abs(actual) >= np.log1p(.02)
    nominal = len(block)
    return {"Outcome": block.Outcome.iloc[0], "Model": model, "Window": window, "N": nominal,
            "RMSLE": float(np.sqrt(np.mean(err**2))), "MAPE_percent": float(np.mean(np.abs(np.expm1(err)))*100),
            "Bias_percent": float(np.mean(np.expm1(err))*100),
            "YoY_direction_accuracy": float(np.mean(np.sign(actual)==np.sign(pred))),
            "MoM_direction_accuracy": float(np.mean(np.sign(block[model]-block.Previous_level)==np.sign(block.Actual-block.Previous_level))),
            "Material_direction_accuracy": float(np.mean(np.sign(actual[material])==np.sign(pred[material]))) if material.any() else None,
            "Momentum_accuracy": float(np.mean(adir==pdir)),
            "Turn_accuracy": float(np.mean(turns==pturns)) if len(turns) else None,
            "Turn_precision": true_positive/float(pturns.sum()) if pturns.any() else 0.,
            "Turn_recall": true_positive/float(turns.sum()) if turns.any() else 0.,
            "Turn_F1": 2.*true_positive/f1_denominator if f1_denominator else 0.,
            "Turn_N": int(valid_turn.sum()), "Actual_turn_N": int(turns.sum()),
            "Predicted_turn_N": int(pturns.sum()), "Turn_TP": true_positive,
            "Turn_FP": false_positive, "Turn_FN": false_negative, "Turn_TN": true_negative,
            "Path_correlation": float(np.corrcoef(actual,pred)[0,1]) if np.std(pred)>1e-10 else 0.,
            "Calibration_slope": float(np.cov(pred,actual,ddof=1)[0,1]/np.var(pred,ddof=1)) if np.std(pred)>1e-10 else 0.}

def select(performance):
    long = performance[performance.Window=="Long_2015_2023"].set_index("Model")
    if not (long.N == 108).all():
        raise AssertionError("Every model must have 108 monthly selection predictions")
    score = pd.Series(0., index=long.index)
    for name, high, weight in [("RMSLE",False,.30),("YoY_direction_accuracy",True,.20),
                               ("Momentum_accuracy",True,.20),("Turn_F1",True,.15),("Path_correlation",True,.10)]:
        score += weight*long[name].rank(pct=True, ascending=not high)
    score += .05*(long.Calibration_slope-1).abs().rank(pct=True)
    return str(score.idxmin()), score
