"""New synthetic checks against the actual estimator/evaluation source."""
import unittest
from types import SimpleNamespace
import numpy as np
import pandas as pd
import scoped_industry_estimators as estimators
import evaluation


class EstimatorChecks(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(20260922)
        self.x = rng.normal(size=(90, 7))
        self.y = .12 * self.x[:, 0] - .07 * self.x[:, 3] + rng.normal(0, .025, 90)
        self.names = ['AR1', 'AR2', 'SIN1', 'external_1', 'external_2', 'external_3', 'external_4']
        self.groups = ['core', 'core', 'core', 'a', 'a', 'b', 'b']

    def test_all_standard_estimators_finite_without_mutating_training_input(self):
        before = self.x.copy()
        optional = {'XGBoost', 'LightGBM', 'CatBoost'}
        for name in estimators.MODEL_NAMES:
            if name in optional:
                continue
            with self.subTest(name=name):
                fitted = estimators.fit_model(name, self.x[:72], self.y[:72], self.names, self.groups)
                prediction = fitted.predict(self.x[72:])
                self.assertEqual(prediction.shape, (18,))
                self.assertTrue(np.isfinite(prediction).all())
                self.assertEqual(len(estimators.feature_reliability(fitted)), len(self.names))
        np.testing.assert_array_equal(self.x, before)

    def test_contract_rejects_invalid_features_and_unknown_models(self):
        with self.assertRaises(ValueError):
            estimators.fit_model('unknown', self.x, self.y, self.names, self.groups)
        model = estimators.fit_model('Ridge', self.x, self.y, self.names, self.groups)
        for invalid in [self.x[:, :-1], np.full_like(self.x, np.nan)]:
            with self.assertRaises(ValueError):
                model.predict(invalid)

    def test_reliability_is_bounded_and_contributions_reconstruct_prediction(self):
        model = estimators.fit_model('AdaLASSO_Reliability', self.x, self.y, self.names, self.groups)
        d = model.delegate
        np.testing.assert_allclose(d.contributions(self.x).sum(axis=1) + d.intercept_, model.predict(self.x))
        self.assertTrue(((model.reliability_ >= 0) & (model.reliability_ <= 1)).all())

    def test_ensemble_maps_member_specific_preprocessing(self):
        soft = SimpleNamespace(names=self.names, centers=np.zeros(7), scales=np.ones(7))
        keep = [0, 1, 3]
        strict = SimpleNamespace(names=[self.names[j] for j in keep], centers=np.ones(3), scales=np.full(3, 2.))
        a = estimators.fit_model('Ridge', self.x, self.y, self.names, self.groups)
        sx = (self.x[:, keep] - strict.centers) / strict.scales
        b = estimators.fit_model('AdaLASSO_Reliability_Strict', sx, self.y, strict.names, ['core','core','a'])
        combined = estimators.EnsemblePredictor({'Ridge': a, 'AdaLASSO_Reliability_Strict': b},
            {False: soft, True: strict}, {'Ridge': .4, 'AdaLASSO_Reliability_Strict': .6})
        np.testing.assert_allclose(combined.predict(self.x), .4 * a.predict(self.x) + .6 * b.predict(sx))


class EvaluationChecks(unittest.TestCase):
    def test_selection_uses_long_window_and_rejects_incomplete_coverage(self):
        rows = []
        for model, loss in [('good', .01), ('bad', .4)]:
            rows.append(dict(Model=model, Window='Long_2015_2023', N=108, RMSLE=loss,
                YoY_direction_accuracy=1-loss, Momentum_accuracy=1-loss, Turn_F1=1-loss,
                Path_correlation=1-loss, Calibration_slope=1+loss))
        frame = pd.DataFrame(rows)
        self.assertEqual(evaluation.select(frame)[0], 'good')
        # A reversed recent-period ranking cannot alter selection.
        recent = frame.assign(Window='Short_2024_2026', RMSLE=[100, 0])
        self.assertEqual(evaluation.select(pd.concat([frame, recent]))[0], 'good')
        with self.assertRaises(AssertionError):
            evaluation.select(frame.assign(N=107))

    def test_ensemble_weights_normalize_and_prefer_lower_past_error(self):
        models = evaluation.ENSEMBLE_MODELS
        equal = evaluation.ensemble_weights([])
        self.assertAlmostEqual(sum(equal.values()), 1.)
        history = []
        for i in range(10):
            actual = .03 * np.sin(i)
            row = dict(Actual_growth=actual, Previous_growth=.03*np.sin(i-1))
            row.update({m+'_growth': actual + j*.02 for j,m in enumerate(models)})
            history.append(row)
        weights = evaluation.ensemble_weights(history)
        self.assertAlmostEqual(sum(weights.values()), 1.)
        self.assertGreater(weights[models[0]], weights[models[-1]])

    def test_turn_metric_uses_observed_prior_momentum_at_window_boundary(self):
        index = pd.date_range('2023-11-01', periods=8, freq='MS')
        actual = np.array([.01,.03,.02,.04,.01,.05,.02,.03])
        previous = np.array([0.,.01,.03,.02,.04,.01,.05,.02])
        frame = pd.DataFrame(dict(Outcome='synthetic', Actual_growth=actual,
            Previous_growth=previous, exact_growth=actual, Actual=100*np.exp(actual),
            Previous_level=100*np.exp(previous), exact=100*np.exp(actual)), index=index)
        result = evaluation.metric(frame, 'exact', 'Short_2024_2026')
        self.assertEqual(result['N'], 6)
        self.assertEqual(result['Turn_N'], 6)
        self.assertAlmostEqual(result['RMSLE'], 0.)
        self.assertAlmostEqual(result['Turn_F1'], 1.)


if __name__ == '__main__':
    unittest.main()
