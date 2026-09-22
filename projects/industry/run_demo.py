"""New synthetic driver using actual project estimator and evaluation code."""
import json
import numpy as np
import scoped_industry_estimators as estimators
from evaluation import ensemble_weights

rng = np.random.default_rng(20260922)
x = rng.normal(size=(100, 6))
y = .12*x[:, 0] - .08*x[:, 3] + rng.normal(0, .03, 100)
names = ['AR1', 'AR2', 'SIN1', 'external_a', 'external_b', 'external_c']
groups = ['core', 'core', 'core', 'a', 'a', 'b']
results = []
for name in ['AR_Fourier_Ridge', 'Elastic_Net', 'Sparse_Group_LASSO', 'AdaLASSO_Reliability', 'Central_AR_AdaReliability']:
    model = estimators.fit_model(name, x[:80], y[:80], names, groups)
    prediction = model.predict(x[80:])
    results.append(dict(model=name, held_out_n=20, synthetic_rmse=float(np.sqrt(np.mean((prediction-y[80:])**2)))))
print(json.dumps(dict(synthetic_data_only=True, training_n=80, results=results,
    limitation='Smoke demonstration, not reproduction of private-data forecasting accuracy.',
    initial_ensemble_weights=ensemble_weights([])), indent=2))
