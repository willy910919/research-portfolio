import unittest

from research_room import (
    ExecutionManifest,
    ExecutionStep,
    bind_user_requirement_runtime,
    build_analysis_graph,
    build_user_requirement,
    compile_lead_plan,
    deterministic_contract_audit,
    deterministic_requirement_audit,
    executable_tool_ids,
    get_tool_manifest,
    reconcile_research_plan,
    requirement_execution_gate,
    resolve_tool_requests,
    tool_outputs,
    validate_analysis_graph,
    verify_user_requirement,
)
from research_room.clarification_memory import (
    build_clarification_resume_text,
    build_pending_clarification,
)
from research_room.reconciliation import extract_deterministic_facts


class TypedResearchContractTests(unittest.TestCase):
    def setUp(self):
        self.columns = [
            "AGE",
            "SEX",
            "BMI",
            "HYPERTENSION_SELF",
            "LUNG_CA_SELF",
            "BREAST_CA_SELF",
            "FASTING_GLUCOSE",
            "HBA1C",
            "DIABETES",
            "DRK",
            "GOUT",
            "GOUT_SELF",
            "URIC_ACID",
            "EDUCATION",
        ]

    def test_user_requirement_hash_is_semantic_not_runtime_lineage(self):
        question = (
            "以 HYPERTENSION_SELF 為 outcome，評估 BMI 的關聯，"
            "調整 AGE 與 SEX，使用 logistic regression。"
        )
        spec = {
            "goal": "association",
            "outcomes": ["HYPERTENSION_SELF"],
            "exposures": ["BMI"],
            "forced_covariates": ["AGE", "SEX"],
            "requested_methods": ["logistic_regression"],
            "required_outputs": ["odds_ratio"],
        }
        facts = {
            "outcomes": ["HYPERTENSION_SELF"],
            "exposures": ["BMI"],
            "forced_covariates": ["AGE", "SEX"],
            "requested_methods": ["logistic_regression"],
            "required_outputs": ["odds_ratio"],
        }
        requirement = build_user_requirement(question, spec, facts)
        first = bind_user_requirement_runtime(
            requirement,
            task_id="TASK_RUN_A",
            dataset_version="DATA_001",
        )
        second = bind_user_requirement_runtime(
            requirement,
            task_id="TASK_RUN_B",
            dataset_version="DATA_001",
        )
        self.assertNotEqual(first["task_id"], second["task_id"])
        self.assertEqual(first["requirement_hash"], second["requirement_hash"])
        self.assertTrue(verify_user_requirement(first))
        self.assertTrue(verify_user_requirement(second))
        explicit_sources = {
            (item["field"], item["source"])
            for item in first["provenance"]
            if item["certainty"] == "explicit"
        }
        self.assertIn(
            ("outcomes", "user_original_text"),
            explicit_sources,
        )

    def test_role_before_column_clarification_answer_resolves_outcome(self):
        facts = extract_deterministic_facts(
            "outcome幫我使用HYPERTENSION_SELF",
            self.columns,
        )
        self.assertEqual(facts["outcomes"], ["HYPERTENSION_SELF"])

    def test_single_scoped_feature_selection_column_is_the_outcome(self):
        question = "幫我針對 DIABETES 特徵選擇，提供給我候選特徵。"
        facts = extract_deterministic_facts(question, self.columns)
        self.assertEqual(facts["outcomes"], ["DIABETES"])

        reconciled = reconcile_research_plan(
            question,
            {
                "task_spec": {
                    "goal": "feature_discovery",
                    "outcomes": [],
                    "requested_methods": [],
                    "required_outputs": ["selected_features"],
                    "needs_clarification": True,
                    "clarification_question": (
                        "請確認要進行特徵選擇的 outcome 欄位。"
                    ),
                },
                "tool_requests": [
                    {"tool": "frequency", "columns": ["DIABETES"]},
                    {"tool": "dataset_profile", "columns": ["DIABETES"]},
                ],
                "needs_clarification": True,
                "clarification_question": (
                    "請確認要進行特徵選擇的 outcome 欄位。"
                ),
            },
            self.columns,
        )
        self.assertEqual(reconciled["task_spec"]["outcomes"], ["DIABETES"])
        self.assertFalse(reconciled["needs_clarification"])
        self.assertEqual(reconciled["clarification_question"], "")

    def test_unscoped_multiple_feature_selection_columns_remain_ambiguous(self):
        facts = extract_deterministic_facts(
            "使用 AGE 和 BMI 做特徵選擇。",
            self.columns,
        )
        self.assertEqual(facts["outcomes"], [])

    def test_pending_clarification_keeps_original_requirement_and_role(self):
        original = "用 LASSO 找出五個穩定候選特徵。"
        plan = {
            "task_spec": {
                "goal": "feature_discovery",
                "task_family": "classification",
                "outcomes": [],
                "requested_methods": ["lasso"],
                "required_outputs": ["selected_features"],
                "top_k": 5,
                "needs_clarification": True,
            },
            "user_requirement": {"original_question": original},
            "analysis_graph": {"nodes": []},
        }
        pending = build_pending_clarification(
            run_id="RUN_001",
            original_question=original,
            clarification_question="請確認要進行特徵選擇的 outcome 欄位。",
            plan=plan,
            phase="waiting_input",
            event_id=42,
        )
        self.assertEqual(pending["requested_fields"], ["outcomes"])
        resumed = build_clarification_resume_text(
            pending,
            "outcome幫我使用HYPERTENSION_SELF",
            self.columns,
        )
        facts = extract_deterministic_facts(resumed, self.columns)
        self.assertIn("LASSO", resumed)
        self.assertEqual(facts["outcomes"], ["HYPERTENSION_SELF"])
        self.assertEqual(facts["top_k"], 5)

    def test_minimal_gate_blocks_only_explicit_method_substitution(self):
        requirement = build_user_requirement(
            "以 BREAST_CA_SELF 為 outcome，用 LASSO 找出五個特徵。",
            {
                "goal": "feature_discovery",
                "outcomes": ["BREAST_CA_SELF"],
                "candidate_scope": "auto_discover",
                "requested_methods": ["lasso"],
                "required_outputs": ["selected_features"],
                "top_k": 5,
            },
            {
                "outcomes": ["BREAST_CA_SELF"],
                "requested_methods": ["lasso"],
                "required_outputs": ["selected_features"],
                "top_k": 5,
            },
        )
        blocked = requirement_execution_gate(
            requirement,
            [
                {
                    "tool": "binary_model_comparison",
                    "outcome": "BREAST_CA_SELF",
                    "methods": ["random_forest"],
                }
            ],
            self.columns,
        )
        self.assertFalse(blocked["allow_execution"])
        self.assertEqual(
            blocked["hard_errors"][0]["code"],
            "explicit_method_not_executed",
        )
        allowed = requirement_execution_gate(
            requirement,
            [
                {
                    "tool": "binary_model_comparison",
                    "outcome": "BREAST_CA_SELF",
                    "methods": ["lasso_logistic"],
                }
            ],
            self.columns,
        )
        self.assertTrue(allowed["allow_execution"], allowed)

    def test_non_core_branch_failure_cannot_erase_completed_analysis(self):
        requirement = build_user_requirement(
            "用 LASSO 找出五個特徵，並提供圖與文獻。",
            {
                "goal": "feature_discovery",
                "outcomes": ["BREAST_CA_SELF"],
                "requested_methods": ["lasso"],
                "required_outputs": [
                    "selected_features",
                    "figure",
                    "literature_evidence",
                ],
                "top_k": 5,
            },
            {
                "outcomes": ["BREAST_CA_SELF"],
                "requested_methods": ["lasso"],
                "required_outputs": [
                    "selected_features",
                    "figure",
                    "literature_evidence",
                ],
                "top_k": 5,
                "explicit_resource_requests": {"literature": True},
            },
        )
        manifest = {
            "requirement_hash": requirement["requirement_hash"],
            "completed_outputs": ["selected_features"],
            "steps": [
                {
                    "node_id": "N_ANALYSIS",
                    "status": "completed",
                    "request": {
                        "tool": "binary_model_comparison",
                        "outcome": "BREAST_CA_SELF",
                        "methods": ["lasso_logistic"],
                    },
                    "evidence_manifest": {
                        "declared_outputs": ["selected_features"],
                        "result_terms": ["AGE", "BMI"],
                    },
                },
                {
                    "node_id": "N_FIGURE",
                    "status": "failed",
                    "request": {"tool": "visualization", "kind": "bar"},
                    "evidence_manifest": {},
                },
            ],
        }
        audit = deterministic_requirement_audit(
            requirement,
            manifest,
            self.columns,
        )
        self.assertNotEqual(audit["decision"], "block")
        self.assertEqual(audit["completion"], "partial")
        self.assertTrue(audit["preserve_completed_artifacts"])
        self.assertIn("selected_features", audit["completed_requirements"])

    def test_tool_registry_is_single_output_source_for_executors(self):
        outputs = tool_outputs()
        self.assertIn("adjusted_logistic_association", executable_tool_ids())
        manifest = get_tool_manifest("adjusted_logistic_association")
        self.assertIsNotNone(manifest)
        self.assertIn("odds_ratio", outputs[manifest.id])
        self.assertIn("exposure_covariate_separation", manifest.guards)

    def test_tool_manifest_exposes_boundary_and_resource_contracts(self):
        manifest = get_tool_manifest("binary_model_comparison")
        snapshot = manifest.to_dict()
        self.assertEqual(snapshot["manifest_version"], "2.0")
        self.assertIn("tabular_data", snapshot["input_contract"]["accepts"])
        self.assertIn("train_split", snapshot["input_contract"]["requires"])
        self.assertIn("selected_features", snapshot["output_contract"]["outputs"])
        self.assertEqual(snapshot["resource_profile"]["queue_class"], "ml")
        self.assertEqual(snapshot["resource_profile"]["timeout_sec"], 600)
        self.assertFalse(snapshot["security"]["network_access"])

    def test_tool_broker_preserves_registered_request_parameters(self):
        request = {
            "tool": "binary_model_comparison",
            "outcome": "BREAST_CA_SELF",
            "methods": ["lasso_logistic"],
            "top_k": 5,
            "forced_covariates": ["AGE", "SEX"],
        }
        resolution = resolve_tool_requests(
            [request, {"tool": "unregistered_remote_tool"}]
        )
        self.assertEqual(resolution["accepted"], [request])
        self.assertEqual(
            resolution["rejected"][0]["reason"],
            "tool_not_registered",
        )
        self.assertFalse(resolution["complete"])

    def test_lead_compiler_restores_only_explicit_lasso_requirements(self):
        question = (
            "Use BREAST_CA_SELF as the outcome. Use LASSO to return five "
            "stable candidate features while forcing AGE and SEX into the model."
        )
        compiled = compile_lead_plan(
            question,
            {
                "task_spec": {
                    "goal": "feature_discovery",
                    "task_family": "classification",
                    "outcomes": ["BREAST_CA_SELF"],
                    "candidate_scope": "auto_discover",
                    "forced_covariates": ["AGE", "SEX"],
                    "requested_methods": ["random_forest"],
                    "required_outputs": ["selected_features"],
                    "top_k": 2,
                },
                "tool_requests": [
                    {
                        "tool": "binary_model_comparison",
                        "outcome": "BREAST_CA_SELF",
                        "methods": ["lasso_logistic"],
                        "top_k": 5,
                        "forced_covariates": ["AGE", "SEX"],
                    }
                ],
            },
            self.columns,
        )
        requirement = compiled["user_requirement"]
        self.assertEqual(requirement["outcomes"], ["BREAST_CA_SELF"])
        self.assertEqual(requirement["requested_methods"], ["lasso"])
        self.assertEqual(requirement["forced_covariates"], ["AGE", "SEX"])
        self.assertEqual(requirement["top_k"], 5)
        self.assertEqual(
            compiled["lead_compilation"]["authority"],
            "lead_agent_plus_explicit_facts",
        )
        self.assertTrue(compiled["tool_broker_resolution"]["complete"])

    def test_group_qualified_count_outputs_are_canonicalized(self):
        question = (
            "篩選 BMI > 30 且 HYPERTENSION_SELF = 1 的個案，"
            "按 SEX 顯示人數與百分比。"
        )
        variants = [
            ["count_by_sex", "percentage_by_sex"],
            ["counts_by_group", "percentages_by_group"],
            ["frequency_by_site", "proportion_by_site"],
        ]
        for required_outputs in variants:
            with self.subTest(required_outputs=required_outputs):
                plan = {
                    "task_spec": {
                        "goal": "descriptive",
                        "task_family": "descriptive",
                        "group_by": ["SEX"],
                        "filters": [
                            {"field": "BMI", "operator": ">", "value": 30},
                            {
                                "field": "HYPERTENSION_SELF",
                                "operator": "==",
                                "value": 1,
                            },
                        ],
                        "required_outputs": required_outputs,
                    },
                    "tool_requests": [
                        {
                            "tool": "filter_group",
                            "filters": [
                                {
                                    "field": "BMI",
                                    "operator": ">",
                                    "value": 30,
                                },
                                {
                                    "field": "HYPERTENSION_SELF",
                                    "operator": "==",
                                    "value": 1,
                                },
                            ],
                            "logic": "and",
                            "group_by": ["SEX"],
                        }
                    ],
                }
                reconciled = reconcile_research_plan(
                    question,
                    plan,
                    self.columns,
                )
                self.assertEqual(
                    reconciled["task_spec"]["required_outputs"],
                    ["count", "percentage"],
                )
                self.assertTrue(reconciled["contract_validation"]["valid"])
                self.assertTrue(reconciled["capability_fit"]["complete"])

    def test_filter_and_group_roles_override_conflicting_plan(self):
        question = (
            "篩選 BMI > 30 且 HYPERTENSION_SELF = 1 的個案，"
            "並且分別去看SEX在這個情境下的人數差異。"
        )
        wrong_plan = {
            "research_goal": "conditional count",
            "tool_requests": [
                {
                    "tool": "filter_group",
                    "filters": [
                        {"field": "SEX", "operator": ">", "value": 30},
                        {"field": "BMI", "operator": "!=", "value": 0},
                    ],
                    "group_by": [],
                }
            ],
        }
        reconciled = reconcile_research_plan(question, wrong_plan, self.columns)
        request = reconciled["tool_requests"][0]
        self.assertEqual(
            request["filters"],
            [
                {"field": "BMI", "operator": ">", "value": 30},
                {
                    "field": "HYPERTENSION_SELF",
                    "operator": "==",
                    "value": 1,
                },
            ],
        )
        self.assertEqual(request["group_by"], ["SEX"])
        self.assertTrue(reconciled["contract_validation"]["valid"])

    def test_linear_regression_roles_are_repaired_from_explicit_text(self):
        question = "使用AGE、SEX建立線性迴歸模型，來預測BMI。"
        wrong_plan = {
            "research_goal": "regression",
            "tool_requests": [
                {
                    "tool": "linear_regression",
                    "outcome": "AGE",
                    "predictors": ["BMI"],
                }
            ],
        }
        reconciled = reconcile_research_plan(question, wrong_plan, self.columns)
        request = reconciled["tool_requests"][0]
        self.assertEqual(request["outcome"], "BMI")
        self.assertEqual(request["predictors"], ["AGE", "SEX"])
        self.assertEqual(reconciled["task_spec"]["outcomes"], ["BMI"])
        self.assertTrue(reconciled["contract_validation"]["valid"])

    def test_forced_covariates_do_not_become_auto_candidate_universe(self):
        question = (
            "以 LUNG_CA_SELF 為 outcome，從所有合格欄位使用 LASSO "
            "找出五個穩定候選特徵，AGE、SEX 必須保留為調整變項。"
        )
        narrow_plan = {
            "research_goal": "feature discovery",
            "tool_requests": [
                {
                    "tool": "binary_model_comparison",
                    "outcome": "LUNG_CA_SELF",
                    "features": ["AGE", "SEX"],
                    "methods": ["random_forest"],
                }
            ],
        }
        reconciled = reconcile_research_plan(question, narrow_plan, self.columns)
        spec = reconciled["task_spec"]
        request = reconciled["tool_requests"][0]
        self.assertEqual(spec["candidate_scope"], "auto_discover")
        self.assertEqual(spec["forced_covariates"], ["AGE", "SEX"])
        self.assertEqual(spec["candidate_features"], [])
        self.assertEqual(spec["top_k"], 5)
        self.assertIn("lasso_logistic", request["methods"])
        self.assertEqual(request["features"], [])
        self.assertEqual(request["candidate_scope"], "auto_discover")
        self.assertEqual(request["forced_covariates"], ["AGE", "SEX"])
        self.assertTrue(reconciled["contract_validation"]["valid"])

    def test_explicit_candidate_features_are_not_expanded(self):
        question = (
            "只使用 AGE、SEX、BMI、HBA1C 預測 HYPERTENSION_SELF，"
            "採用 logistic regression。"
        )
        expanded_plan = {
            "research_goal": "prediction",
            "tool_requests": [
                {
                    "tool": "binary_model_comparison",
                    "outcome": "HYPERTENSION_SELF",
                    "features": [
                        "AGE",
                        "SEX",
                        "BMI",
                        "HBA1C",
                        "FASTING_GLUCOSE",
                    ],
                    "methods": ["logistic_regression"],
                }
            ],
        }
        reconciled = reconcile_research_plan(question, expanded_plan, self.columns)
        self.assertEqual(
            reconciled["tool_requests"][0]["features"],
            ["AGE", "SEX", "BMI", "HBA1C"],
        )
        self.assertEqual(
            reconciled["task_spec"]["candidate_scope"],
            "explicit",
        )
        self.assertTrue(reconciled["contract_validation"]["valid"])

    def test_deterministic_auditor_blocks_modified_execution_request(self):
        plan = reconcile_research_plan(
            "使用 AGE、SEX 建立線性迴歸模型，預測 BMI。",
            {
                "tool_requests": [
                    {
                        "tool": "linear_regression",
                        "outcome": "BMI",
                        "predictors": ["AGE", "SEX"],
                    }
                ]
            },
            self.columns,
        )
        graph = plan["analysis_graph"]
        node = graph["nodes"][0]
        manifest = ExecutionManifest(
            spec_hash=plan["task_spec"]["spec_hash"],
            graph_hash=graph["graph_hash"],
        )
        step = ExecutionStep(
            node_id=node["node_id"],
            capability="linear_regression",
            request={
                "tool": "linear_regression",
                "outcome": "AGE",
                "predictors": ["BMI"],
            },
        )
        step.start()
        step.complete({"artifact_id": "ART_TEST"})
        manifest.steps.append(step)
        manifest.finish()
        audit = deterministic_contract_audit(
            plan["task_spec"],
            graph,
            manifest.to_dict(),
            self.columns,
        )
        self.assertEqual(audit["decision"], "block")
        self.assertTrue(
            any("實際參數" in item for item in audit["missing_requirements"])
        )

    def test_contract_hashes_are_stable_for_same_input(self):
        question = "描述 AGE 與 BMI。"
        plan = {"tool_requests": [{"tool": "describe", "columns": ["AGE", "BMI"]}]}
        first = reconcile_research_plan(question, plan, self.columns)
        second = reconcile_research_plan(question, plan, self.columns)
        self.assertEqual(
            first["task_spec"]["spec_hash"],
            second["task_spec"]["spec_hash"],
        )
        self.assertEqual(
            first["analysis_graph"]["graph_hash"],
            second["analysis_graph"]["graph_hash"],
        )

    def test_required_outputs_are_bound_to_typed_provider_nodes(self):
        plan = reconcile_research_plan(
            (
                "以 HYPERTENSION_SELF 為 outcome，評估 BMI 的關聯，"
                "調整 AGE 與 SEX，提供 OR、95% CI 與 p-value。"
            ),
            {
                "tool_requests": [
                    {
                        "tool": "adjusted_logistic_association",
                        "outcome": "HYPERTENSION_SELF",
                        "exposures": ["BMI"],
                        "covariates": ["AGE", "SEX"],
                    }
                ]
            },
            self.columns,
        )
        graph = plan["analysis_graph"]
        node = graph["nodes"][0]
        self.assertEqual(
            set(graph["required_outputs"]),
            {"odds_ratio", "confidence_interval", "p_value"},
        )
        for output in graph["required_outputs"]:
            self.assertEqual(graph["output_bindings"][output], [node["node_id"]])
            self.assertIn(output, node["output_contract"])

    def test_recomposition_audit_requires_completed_typed_outputs(self):
        plan = reconcile_research_plan(
            "使用 AGE、SEX 建立線性迴歸模型，預測 BMI 並提供 p-value。",
            {
                "tool_requests": [
                    {
                        "tool": "linear_regression",
                        "outcome": "BMI",
                        "predictors": ["AGE", "SEX"],
                    }
                ]
            },
            self.columns,
        )
        graph = plan["analysis_graph"]
        node = graph["nodes"][0]
        manifest = ExecutionManifest(
            spec_hash=plan["task_spec"]["spec_hash"],
            graph_hash=graph["graph_hash"],
        )
        step = ExecutionStep(
            node_id=node["node_id"],
            capability=node["capability"],
            request=node["request"],
        )
        step.start()
        step.complete(
            {
                "declared_outputs": node["output_contract"],
                "has_rendered_table_or_html": True,
            }
        )
        manifest.steps.append(step)
        manifest.finish()
        audit = deterministic_contract_audit(
            plan["task_spec"],
            graph,
            manifest.to_dict(),
            self.columns,
        )
        self.assertEqual(audit["decision"], "pass")
        self.assertIn(
            "typed_output_recomposition",
            audit["completed_requirements"],
        )

    def test_graph_rejects_unknown_or_forward_dependencies(self):
        plan = reconcile_research_plan(
            "描述 AGE，並計算 AGE 與 BMI 的相關係數。",
            {
                "tool_requests": [
                    {
                        "tool": "describe",
                        "columns": ["AGE"],
                        "depends_on": ["correlation"],
                    },
                    {
                        "tool": "correlation",
                        "columns": ["AGE", "BMI"],
                    },
                ]
            },
            self.columns,
        )
        self.assertFalse(plan["contract_validation"]["valid"])
        self.assertTrue(
            any(
                "依賴未先完成" in item
                for item in plan["contract_validation"]["errors"]
            )
        )

    def test_multiple_explicit_outcomes_are_preserved(self):
        question = (
            "分別預測 LUNG_CA_SELF 與 BREAST_CA_SELF，"
            "比較兩個 outcome 的候選特徵來源。"
        )
        plan = {
            "tool_requests": [
                {
                    "tool": "binary_model_comparison",
                    "outcome": "LUNG_CA_SELF",
                    "features": ["AGE", "SEX", "BMI"],
                    "methods": ["lasso_logistic"],
                },
                {
                    "tool": "binary_model_comparison",
                    "outcome": "BREAST_CA_SELF",
                    "features": ["AGE", "SEX", "BMI"],
                    "methods": ["lasso_logistic"],
                },
            ]
        }
        reconciled = reconcile_research_plan(question, plan, self.columns)
        self.assertEqual(
            reconciled["task_spec"]["outcomes"],
            ["LUNG_CA_SELF", "BREAST_CA_SELF"],
        )
        self.assertTrue(reconciled["contract_validation"]["valid"])
        self.assertTrue(
            all(
                not node["depends_on"]
                for node in reconciled["analysis_graph"]["nodes"]
            )
        )

    def test_adjusted_logistic_contract_preserves_all_required_outputs(self):
        question = (
            "以 HYPERTENSION_SELF 為 outcome，評估 BMI 與高血壓的關聯，"
            "並調整 AGE 與 SEX。請使用 logistic regression，"
            "提供 OR、95% CI 與 p-value。"
        )
        plan = {
            "tool_requests": [
                {
                    "tool": "adjusted_logistic_association",
                    "outcome": "HYPERTENSION_SELF",
                    "exposures": ["BMI"],
                    "covariates": ["AGE", "SEX"],
                }
            ]
        }
        reconciled = reconcile_research_plan(question, plan, self.columns)
        spec = reconciled["task_spec"]
        self.assertEqual(spec["outcomes"], ["HYPERTENSION_SELF"])
        self.assertEqual(spec["exposures"], ["BMI"])
        self.assertEqual(spec["forced_covariates"], ["AGE", "SEX"])
        self.assertEqual(
            set(spec["required_outputs"]),
            {"odds_ratio", "confidence_interval", "p_value"},
        )
        self.assertTrue(reconciled["contract_validation"]["valid"])

    def test_association_wording_does_not_make_outcome_an_explicit_exposure(self):
        question = (
            "評估 BMI 與 HYPERTENSION_SELF 的關聯，調整 AGE 與 SEX，"
            "並以 PubMed 文獻比較資料結果；只引用實際檢索到的 PMID。"
        )
        compiled = compile_lead_plan(
            question,
            {
                "task_spec": {
                    "goal": "association_analysis",
                    "task_family": "inference",
                    "outcomes": ["HYPERTENSION_SELF"],
                    "exposures": ["BMI"],
                    "forced_covariates": ["AGE", "SEX"],
                    "required_outputs": [
                        "odds_ratio",
                        "confidence_interval",
                        "p_value",
                    ],
                },
                "tool_requests": [
                    {
                        "tool": "adjusted_logistic_association",
                        "outcome": "HYPERTENSION_SELF",
                        "exposures": ["BMI"],
                        "covariates": ["AGE", "SEX"],
                    }
                ],
            },
            self.columns,
        )
        requirement = compiled["user_requirement"]
        self.assertEqual(requirement["outcomes"], ["HYPERTENSION_SELF"])
        self.assertEqual(requirement["exposures"], ["BMI"])
        self.assertFalse(
            any(
                item["field"] == "exposures"
                and item["value"] == "HYPERTENSION_SELF"
                and item["certainty"] == "explicit"
                for item in requirement["provenance"]
            )
        )
        gate = requirement_execution_gate(
            requirement,
            compiled["tool_requests"],
            self.columns,
        )
        self.assertTrue(gate["allow_execution"], gate)

    def test_inferred_outcome_exposure_overlap_is_normalized_not_blocked(self):
        requirement = build_user_requirement(
            "評估 BMI 與 HYPERTENSION_SELF 的關聯。",
            {
                "goal": "association_analysis",
                "outcomes": ["HYPERTENSION_SELF"],
                "exposures": ["BMI", "HYPERTENSION_SELF"],
            },
            {
                "outcomes": [],
                "exposures": [],
            },
        )
        self.assertEqual(requirement["outcomes"], ["HYPERTENSION_SELF"])
        self.assertEqual(requirement["exposures"], ["BMI"])
        gate = requirement_execution_gate(
            requirement,
            [
                {
                    "tool": "adjusted_logistic_association",
                    "outcome": "HYPERTENSION_SELF",
                    "exposures": ["BMI"],
                }
            ],
            self.columns,
        )
        self.assertTrue(gate["allow_execution"], gate)

    def test_two_explicit_incompatible_roles_remain_a_hard_error(self):
        requirement = build_user_requirement(
            (
                "以 HYPERTENSION_SELF 為 outcome，並將 "
                "HYPERTENSION_SELF 作為 exposure。"
            ),
            {
                "goal": "association_analysis",
                "outcomes": ["HYPERTENSION_SELF"],
                "exposures": ["HYPERTENSION_SELF"],
            },
            {
                "outcomes": ["HYPERTENSION_SELF"],
                "exposures": ["HYPERTENSION_SELF"],
            },
        )
        gate = requirement_execution_gate(requirement, [], self.columns)
        self.assertFalse(gate["allow_execution"])
        self.assertIn(
            "explicit_role_conflict",
            {item["code"] for item in gate["hard_errors"]},
        )

    def test_covariate_cue_stops_at_the_next_analysis_clause(self):
        question = (
            "控制 AGE 和 SEX 後，以邏輯斯迴歸檢驗 BMI 是否與 "
            "HYPERTENSION_SELF 有關，請報 OR、95% CI、p 值。"
        )
        plan = {
            "task_spec": {
                "goal": "association",
                "task_family": "classification",
                "outcomes": ["HYPERTENSION_SELF"],
                "exposures": ["BMI"],
                "forced_covariates": [
                    "AGE",
                    "SEX",
                    "BMI",
                    "HYPERTENSION_SELF",
                ],
                "requested_methods": ["logistic_regression"],
                "required_outputs": [
                    "odds_ratio",
                    "confidence_interval",
                    "p_value",
                ],
            },
            "tool_requests": [
                {
                    "tool": "adjusted_logistic_association",
                    "outcome": "HYPERTENSION_SELF",
                    "exposures": ["BMI"],
                    "covariates": [
                        "AGE",
                        "SEX",
                        "BMI",
                        "HYPERTENSION_SELF",
                    ],
                }
            ],
        }
        reconciled = reconcile_research_plan(question, plan, self.columns)
        spec = reconciled["task_spec"]
        request = reconciled["tool_requests"][0]
        self.assertEqual(spec["outcomes"], ["HYPERTENSION_SELF"])
        self.assertEqual(spec["exposures"], ["BMI"])
        self.assertEqual(spec["forced_covariates"], ["AGE", "SEX"])
        self.assertEqual(request["exposures"], ["BMI"])
        self.assertEqual(request["covariates"], ["AGE", "SEX"])
        self.assertTrue(reconciled["contract_validation"]["valid"])

    def test_pca_loadings_are_bound_to_the_verified_executor(self):
        reconciled = reconcile_research_plan(
            "使用 AGE、BMI、FASTING_GLUCOSE 做 PCA 並解釋前三個主成分。",
            {
                "task_spec": {
                    "goal": "unsupervised",
                    "task_family": "unsupervised",
                    "predictors": ["AGE", "BMI", "FASTING_GLUCOSE"],
                    "requested_methods": ["pca"],
                    "required_outputs": [
                        "components",
                        "explained_variance",
                        "loadings",
                    ],
                },
                "tool_requests": [
                    {
                        "tool": "pca",
                        "columns": ["AGE", "BMI", "FASTING_GLUCOSE"],
                        "components": 3,
                    }
                ],
            },
            self.columns,
        )
        self.assertTrue(reconciled["contract_validation"]["valid"])
        self.assertTrue(reconciled["capability_fit"]["complete"])
        self.assertIn(
            "loadings",
            reconciled["analysis_graph"]["nodes"][0]["output_contract"],
        )

    def test_explicit_pca_prunes_redundant_derived_scatter(self):
        question = (
            "使用 AGE、BMI、FASTING_GLUCOSE、HBA1C 進行 PCA，"
            "顯示解釋變異比例、loadings 與前兩個主成分圖。"
            "直接做分析就好。"
        )
        reconciled = reconcile_research_plan(
            question,
            {
                "task_spec": {
                    "goal": "unsupervised",
                    "task_family": "unsupervised",
                    "predictors": [
                        "AGE",
                        "BMI",
                        "FASTING_GLUCOSE",
                        "HBA1C",
                    ],
                    "candidate_features": [
                        "AGE",
                        "BMI",
                        "FASTING_GLUCOSE",
                        "HBA1C",
                    ],
                    "requested_methods": ["pca"],
                    "required_outputs": ["figure"],
                },
                "tool_requests": [
                    {
                        "tool": "pca",
                        "columns": [
                            "AGE",
                            "BMI",
                            "FASTING_GLUCOSE",
                            "HBA1C",
                        ],
                        "components": 2,
                    },
                    {
                        "tool": "scatter",
                        "x": "PC1",
                        "y": "PC2",
                    },
                ],
            },
            self.columns,
        )
        self.assertTrue(reconciled["contract_validation"]["valid"])
        self.assertEqual(
            [item["tool"] for item in reconciled["tool_requests"]],
            ["pca"],
        )
        self.assertEqual(
            set(reconciled["task_spec"]["required_outputs"]),
            {"explained_variance", "loadings", "figure"},
        )
        self.assertTrue(
            any(
                "redundant scatter" in note
                for note in reconciled["task_spec"]["reconciliation_notes"]
            )
        )

    def test_literature_only_contract_does_not_require_analysis_nodes(self):
        reconciled = reconcile_research_plan(
            "比較剛才的 LASSO 結果與 PubMed 文獻，提供 PMID 並說明限制。",
            {
                "task_spec": {
                    "goal": "research_discussion",
                    "task_family": "other",
                    "requested_methods": ["lasso"],
                    "method_selection_authority": "user_explicit",
                    "required_outputs": [
                        "pmid_list",
                        "literature_evidence",
                        "evidence_limitation_summary",
                    ],
                },
                "tool_requests": [
                    {
                        "tool": "binary_model_comparison",
                        "outcome": "HYPERTENSION_SELF",
                        "methods": ["lasso"],
                    }
                ],
            },
            self.columns,
        )
        self.assertTrue(reconciled["contract_validation"]["valid"])
        self.assertEqual(reconciled["task_spec"]["requested_methods"], [])
        self.assertEqual(reconciled["analysis_graph"]["nodes"], [])
        self.assertEqual(reconciled["analysis_graph"]["required_outputs"], [])
        self.assertTrue(reconciled["capability_fit"]["complete"])
        self.assertTrue(reconciled["needs_literature"])

    def test_reply_context_cannot_become_current_explicit_contract(self):
        question = (
            "關於這個結果，有沒有文獻支持或相關研究討論？"
            "\n\n[可驗證的對話脈絡]\n"
            "上一輪以 HYPERTENSION_SELF 為 outcome，用 LASSO 選擇特徵，"
            "AGE 與 SEX 固定保留，輸出 selected_features。"
        )
        reconciled = reconcile_research_plan(
            question,
            {
                "task_spec": {
                    "goal": "research_discussion",
                    "task_family": "other",
                    "outcomes": ["HYPERTENSION_SELF"],
                    "forced_covariates": ["AGE", "SEX"],
                    "requested_methods": ["lasso"],
                    "method_selection_authority": "user_explicit",
                    "required_outputs": [
                        "selected_features",
                        "pmid_list",
                        "evidence_limitation_summary",
                    ],
                },
                "tool_requests": [
                    {
                        "tool": "binary_model_comparison",
                        "outcome": "HYPERTENSION_SELF",
                        "forced_covariates": ["AGE", "SEX"],
                        "methods": ["lasso"],
                    }
                ],
            },
            self.columns,
        )
        facts = reconciled["task_spec"]["explicit_facts"]
        self.assertEqual(facts["mentioned_columns"], [])
        self.assertEqual(facts["requested_methods"], [])
        self.assertTrue(reconciled["contract_validation"]["valid"])
        self.assertEqual(reconciled["task_spec"]["requested_methods"], [])
        self.assertNotIn(
            "selected_features",
            reconciled["task_spec"]["required_outputs"],
        )
        self.assertEqual(reconciled["analysis_graph"]["nodes"], [])

    def test_literature_followup_may_name_prior_method_and_outcome(self):
        questions = [
            "剛才 HYPERTENSION_SELF 的 LASSO 結果有 PubMed 文獻支持嗎？",
            "請把前面的候選特徵和相關研究比較，列出 PMID。",
            "這個模型結果在文獻中是否一致？請說明相反證據與限制。",
        ]
        for question in questions:
            with self.subTest(question=question):
                reconciled = reconcile_research_plan(
                    question,
                    {
                        "task_spec": {
                            "goal": "research_discussion",
                            "task_family": "literature_review",
                            "outcomes": ["HYPERTENSION_SELF"],
                            "requested_methods": ["lasso"],
                            "required_outputs": [
                                "pmid_list",
                                "literature_evidence",
                                "evidence_limitation_summary",
                            ],
                        },
                        "tool_requests": [
                            {
                                "tool": "binary_model_comparison",
                                "outcome": "HYPERTENSION_SELF",
                                "methods": ["lasso"],
                            }
                        ],
                    },
                    self.columns,
                )
                self.assertTrue(reconciled["contract_validation"]["valid"])
                self.assertEqual(reconciled["analysis_graph"]["nodes"], [])
                self.assertEqual(
                    reconciled["task_spec"]["requested_methods"],
                    [],
                )
                self.assertTrue(reconciled["needs_literature"])

    def test_analysis_plus_literature_keeps_the_analysis_graph(self):
        reconciled = reconcile_research_plan(
            (
                "以 HYPERTENSION_SELF 為 outcome，評估 BMI 的關聯，"
                "提供 OR、95% CI 與 p-value，並找 PubMed 文獻比較。"
            ),
            {
                "task_spec": {
                    "goal": "association_analysis",
                    "task_family": "regression",
                    "outcomes": ["HYPERTENSION_SELF"],
                    "exposures": ["BMI"],
                    "requested_methods": ["logistic_regression"],
                    "required_outputs": [
                        "odds_ratio",
                        "confidence_interval",
                        "p_value",
                    ],
                },
                "tool_requests": [
                    {
                        "tool": "adjusted_logistic_association",
                        "outcome": "HYPERTENSION_SELF",
                        "exposures": ["BMI"],
                        "covariates": [],
                    }
                ],
            },
            self.columns,
        )
        self.assertTrue(reconciled["contract_validation"]["valid"])
        self.assertEqual(
            [item["tool"] for item in reconciled["tool_requests"]],
            ["adjusted_logistic_association"],
        )
        self.assertTrue(reconciled["needs_literature"])

    def test_kmeans_contract_keeps_requested_cluster_count_and_profiles(self):
        reconciled = reconcile_research_plan(
            "用 AGE、BMI、FASTING_GLUCOSE 與 HBA1C 將樣本分成三群並比較群組特徵。",
            {
                "task_spec": {
                    "goal": "unsupervised",
                    "task_family": "unsupervised",
                    "predictors": ["AGE", "BMI", "FASTING_GLUCOSE", "HBA1C"],
                    "requested_methods": ["kmeans_clustering", "describe"],
                    "required_outputs": [
                        "cluster_assignments",
                        "group_descriptive_statistics",
                        "group_means",
                        "group_distributions",
                    ],
                },
                "tool_requests": [
                    {
                        "tool": "kmeans_clustering",
                        "features": ["AGE", "BMI", "FASTING_GLUCOSE", "HBA1C"],
                    },
                    {
                        "tool": "filter_group",
                        "group_by": ["cluster"],
                        "filters": [],
                    },
                    {
                        "tool": "describe",
                        "columns": [
                            "AGE",
                            "BMI",
                            "FASTING_GLUCOSE",
                            "HBA1C",
                        ],
                    },
                ],
            },
            self.columns,
        )
        self.assertTrue(reconciled["contract_validation"]["valid"])
        self.assertTrue(reconciled["capability_fit"]["complete"])
        self.assertEqual(len(reconciled["tool_requests"]), 1)
        self.assertNotIn(
            "group_descriptive_statistics",
            reconciled["task_spec"]["required_outputs"],
        )
        self.assertIn(
            "group_descriptive_stats",
            reconciled["task_spec"]["required_outputs"],
        )
        self.assertEqual(reconciled["tool_requests"][0]["n_clusters"], 3)
        self.assertEqual(
            reconciled["tool_requests"][0]["features"],
            ["AGE", "BMI", "FASTING_GLUCOSE", "HBA1C"],
        )
        self.assertIn(
            "group_means",
            reconciled["analysis_graph"]["nodes"][0]["output_contract"],
        )
        self.assertIn(
            "group_distributions",
            reconciled["analysis_graph"]["nodes"][0]["output_contract"],
        )

    def test_external_method_exploration_is_not_blocked_as_analysis_execution(self):
        question = (
            "評估 BMI 與 HYPERTENSION_SELF 的非線性關聯，如果內建能力不足，"
            "請探索 restricted cubic spline logistic regression 的可信方法學"
            "與官方實作，但不要直接執行未核准遠端程式。"
        )
        reconciled = reconcile_research_plan(
            question,
            {
                "task_spec": {
                    "goal": "association",
                    "task_family": "inference",
                    "outcomes": ["HYPERTENSION_SELF"],
                    "exposures": ["BMI"],
                    "requested_methods": [
                        "restricted_cubic_spline_logistic",
                    ],
                    "required_outputs": [
                        "methodology overview",
                        "recommended software packages",
                        "implementation steps",
                    ],
                },
                "tool_requests": [
                    {
                        "tool": "dataset_profile",
                        "columns": ["BMI", "HYPERTENSION_SELF"],
                    }
                ],
                "needs_literature": True,
                "needs_external_information": True,
                "delegations": [
                    {"agent": "methods"},
                    {"agent": "external"},
                ],
            },
            self.columns,
        )
        self.assertEqual(
            reconciled["task_spec"]["requested_methods"],
            ["restricted_cubic_spline_logistic"],
        )
        self.assertEqual(reconciled["task_spec"]["outcomes"], ["HYPERTENSION_SELF"])
        self.assertEqual(reconciled["task_spec"]["exposures"], ["BMI"])
        self.assertEqual(reconciled["analysis_graph"]["required_outputs"], [])
        self.assertTrue(reconciled["contract_validation"]["valid"])
        self.assertTrue(reconciled["needs_literature"])
        self.assertTrue(reconciled["needs_external_information"])

    def test_builtin_heatmap_prunes_unnecessary_agents_and_external_resources(self):
        question = "直接提供 BMI 和 AGE 的關係熱力圖。"
        overplanned = {
            "research_goal": "draw a heatmap",
            "task_spec": {
                "goal": "descriptive",
                "task_family": "descriptive",
                "requested_methods": ["seaborn"],
                "required_outputs": ["figure"],
                "needs_clarification": True,
                "clarification_question": "請上傳資料並指定 BMI 範圍。",
            },
            "tool_requests": [
                {"tool": "package_status"},
                {"tool": "dataset_profile", "columns": ["AGE", "BMI"]},
                {"tool": "scatter", "x": "AGE", "y": "BMI"},
            ],
            "delegations": [
                {"agent": "methods"},
                {"agent": "external"},
                {"agent": "capabilities"},
            ],
            "needs_literature": True,
            "literature_queries": ["matplotlib heatmap"],
            "needs_external_information": True,
            "external_information_queries": [
                {"source": "pypi", "query": "seaborn heatmap"}
            ],
            "needs_code": True,
            "needs_clarification": True,
            "clarification_question": "請上傳資料並指定 BMI 範圍。",
        }
        reconciled = reconcile_research_plan(
            question,
            overplanned,
            self.columns,
        )
        self.assertEqual(
            reconciled["tool_requests"],
            [
                {
                    "tool": "heatmap",
                    "kind": "density",
                    "x": "BMI",
                    "y": "AGE",
                    "bins": 35,
                }
            ],
        )
        self.assertFalse(reconciled["needs_clarification"])
        self.assertFalse(reconciled["needs_literature"])
        self.assertFalse(reconciled["needs_external_information"])
        self.assertFalse(reconciled["needs_code"])
        self.assertEqual(reconciled["delegations"], [])
        self.assertTrue(reconciled["capability_fit"]["complete"])
        self.assertTrue(
            reconciled["capability_fit"]["minimum_sufficient_resources"]
        )
        self.assertTrue(reconciled["contract_validation"]["valid"])

    def test_capability_fit_prunes_hallucinated_external_work_for_correlation(self):
        reconciled = reconcile_research_plan(
            "計算 AGE 和 BMI 的相關係數。",
            {
                "task_spec": {
                    "goal": "descriptive",
                    "task_family": "descriptive",
                    "required_outputs": ["correlation"],
                },
                "tool_requests": [
                    {"tool": "correlation", "columns": ["AGE", "BMI"]}
                ],
                "needs_external_information": True,
                "external_information_queries": [
                    {"source": "github", "query": "correlation implementation"}
                ],
                "delegations": [{"agent": "external"}],
            },
            self.columns,
        )
        self.assertTrue(reconciled["capability_fit"]["complete"])
        self.assertFalse(reconciled["needs_external_information"])
        self.assertEqual(reconciled["external_information_queries"], [])
        self.assertEqual(reconciled["delegations"], [])

    def test_standard_boxplot_is_reconciled_as_one_builtin_visualization(self):
        reconciled = reconcile_research_plan(
            "依 SEX 繪製 BMI 箱型圖。",
            {
                "task_spec": {
                    "goal": "descriptive",
                    "task_family": "descriptive",
                    "required_outputs": ["figure"],
                },
                "tool_requests": [
                    {"tool": "package_status"},
                    {"tool": "histogram", "column": "BMI", "group": "SEX"},
                ],
                "needs_external_information": True,
                "external_information_queries": [
                    {"source": "pypi", "query": "seaborn boxplot"}
                ],
                "delegations": [{"agent": "capabilities"}],
            },
            self.columns,
        )
        self.assertEqual(
            reconciled["tool_requests"],
            [
                {
                    "tool": "visualization",
                    "kind": "boxplot",
                    "x": "SEX",
                    "y": "BMI",
                    "group": "",
                }
            ],
        )
        self.assertTrue(reconciled["capability_fit"]["complete"])
        self.assertFalse(reconciled["needs_external_information"])
        self.assertEqual(reconciled["delegations"], [])

    def test_grouped_distribution_accepts_equivalent_visualization_schemas(self):
        question = "幫我根據性別區分出兩組 BMI，我要看這兩組 BMI 的分布。"
        requests = [
            {"tool": "histogram", "column": "BMI", "group": "SEX"},
            {
                "tool": "visualization",
                "kind": "boxplot",
                "x": "SEX",
                "y": "BMI",
                "group": "",
            },
            {
                "tool": "visualization",
                "kind": "violin",
                "x": "SEX",
                "y": "BMI",
                "group": "",
            },
        ]
        for request in requests:
            with self.subTest(request=request):
                reconciled = reconcile_research_plan(
                    question,
                    {
                        "task_spec": {
                            "goal": "grouped_distribution",
                            "task_family": "descriptive",
                            "group_by": ["SEX"],
                            "required_outputs": ["figure"],
                        },
                        "tool_requests": [request],
                    },
                    self.columns,
                )
                self.assertEqual(
                    reconciled["task_spec"]["group_by"],
                    ["SEX"],
                )
                self.assertTrue(reconciled["contract_validation"]["valid"])
                self.assertTrue(reconciled["capability_fit"]["complete"])

    def test_grouped_distribution_repairs_missing_tool_group_binding(self):
        reconciled = reconcile_research_plan(
            "幫我根據性別區分出兩組 BMI，我要看這兩組 BMI 的分布。",
            {
                "task_spec": {
                    "goal": "grouped_distribution",
                    "task_family": "descriptive",
                    "group_by": ["SEX"],
                    "required_outputs": ["figure"],
                },
                "tool_requests": [
                    {"tool": "histogram", "column": "BMI", "group": ""}
                ],
            },
            self.columns,
        )
        self.assertEqual(reconciled["tool_requests"][0]["group"], "SEX")
        self.assertTrue(reconciled["contract_validation"]["valid"])

    def test_grouped_distribution_role_is_not_specific_to_bmi_or_sex(self):
        cases = [
            (
                "根據飲酒狀態區分不同組別，我要看各組 URIC_ACID 的分布。",
                "DRK",
                "URIC_ACID",
            ),
            (
                "依教育程度分組，比較每組 AGE 的分布。",
                "EDUCATION",
                "AGE",
            ),
        ]
        for question, group, measure in cases:
            with self.subTest(question=question):
                reconciled = reconcile_research_plan(
                    question,
                    {
                        "task_spec": {
                            "goal": "grouped_distribution",
                            "task_family": "descriptive",
                            "group_by": [group],
                            "required_outputs": ["figure"],
                        },
                        "tool_requests": [
                            {
                                "tool": "histogram",
                                "column": measure,
                                "group": group,
                            }
                        ],
                    },
                    self.columns,
                )
                self.assertEqual(reconciled["task_spec"]["group_by"], [group])
                self.assertEqual(
                    reconciled["tool_requests"][0],
                    {
                        "tool": "histogram",
                        "column": measure,
                        "group": group,
                    },
                )
                self.assertTrue(reconciled["contract_validation"]["valid"])

    def test_delegated_method_selection_is_not_locked_as_user_request(self):
        question = (
            "召集必要的agent，針對DRK與GOUT進行合適的關聯性分析，"
            "並且提供給我圖進行參考。"
        )
        reconciled = reconcile_research_plan(
            question,
            {
                "task_spec": {
                    "goal": "association",
                    "task_family": "inference",
                    "outcomes": ["GOUT"],
                    "exposures": ["DRK"],
                    "requested_methods": ["chi_square"],
                    "required_outputs": ["figure"],
                    "needs_clarification": True,
                    "clarification_question": (
                        "請說明要維持原指定方法，或允許替代方案。"
                    ),
                },
                "needs_clarification": True,
                "clarification_question": (
                    "請說明要維持原指定方法，或允許替代方案。"
                ),
                "tool_requests": [
                    {"tool": "chi_square", "x": "DRK", "y": "GOUT"},
                    {
                        "tool": "visualization",
                        "kind": "bar",
                        "x": "DRK",
                        "y": "",
                        "aggregation": "count",
                        "group": "GOUT",
                    },
                ],
            },
            self.columns,
        )
        spec = reconciled["task_spec"]
        self.assertEqual(spec["method_selection_authority"], "chair_delegated")
        self.assertEqual(spec["requested_methods"], [])
        self.assertIn("chi_square", spec["selected_methods"])
        self.assertFalse(spec["needs_clarification"])
        self.assertEqual(reconciled["clarification_question"], "")
        self.assertTrue(reconciled["contract_validation"]["valid"])

    def test_explicit_method_remains_user_locked(self):
        question = (
            "以 HYPERTENSION_SELF 為 outcome，用 logistic regression "
            "評估 BMI，調整 AGE 與 SEX。"
        )
        reconciled = reconcile_research_plan(
            question,
            {
                "task_spec": {
                    "goal": "association",
                    "task_family": "regression",
                    "outcomes": ["HYPERTENSION_SELF"],
                    "exposures": ["BMI"],
                    "forced_covariates": ["AGE", "SEX"],
                    "requested_methods": ["logistic_regression"],
                },
                "tool_requests": [
                    {
                        "tool": "adjusted_logistic_association",
                        "outcome": "HYPERTENSION_SELF",
                        "exposures": ["BMI"],
                        "covariates": ["AGE", "SEX"],
                    }
                ],
            },
            self.columns,
        )
        spec = reconciled["task_spec"]
        self.assertEqual(spec["method_selection_authority"], "user_explicit")
        self.assertEqual(spec["requested_methods"], ["logistic_regression"])
        self.assertTrue(reconciled["contract_validation"]["valid"])

    def test_logistic_exposure_is_always_bound_to_the_model(self):
        question = (
            "Use HYPERTENSION_SELF as the outcome and BMI as the exposure; "
            "adjust for AGE and SEX with logistic regression and report OR, "
            "95% CI, and p-value."
        )
        reconciled = reconcile_research_plan(
            question,
            {
                "task_spec": {
                    "goal": "association",
                    "task_family": "inference",
                    "outcomes": ["HYPERTENSION_SELF"],
                    "exposures": ["BMI"],
                    "predictors": ["AGE", "SEX"],
                    "forced_covariates": ["AGE", "SEX"],
                    "requested_methods": ["logistic_regression"],
                    "required_outputs": [
                        "odds_ratio",
                        "confidence_interval",
                        "p_value",
                    ],
                },
                "tool_requests": [
                    {
                        "tool": "logistic_regression",
                        "outcome": "HYPERTENSION_SELF",
                        "predictors": ["AGE", "SEX"],
                    }
                ],
            },
            self.columns,
        )
        request = reconciled["analysis_graph"]["nodes"][0]["request"]
        self.assertEqual(request["predictors"], ["BMI", "AGE", "SEX"])
        self.assertTrue(reconciled["contract_validation"]["valid"])

    def test_validator_blocks_a_graph_that_drops_the_exposure(self):
        task_spec = reconcile_research_plan(
            (
                "Use HYPERTENSION_SELF as the outcome and BMI as the exposure; "
                "adjust for AGE and SEX with logistic regression."
            ),
            {
                "task_spec": {
                    "goal": "association",
                    "task_family": "inference",
                    "outcomes": ["HYPERTENSION_SELF"],
                    "exposures": ["BMI"],
                    "forced_covariates": ["AGE", "SEX"],
                    "requested_methods": ["logistic_regression"],
                },
                "tool_requests": [
                    {
                        "tool": "logistic_regression",
                        "outcome": "HYPERTENSION_SELF",
                        "predictors": ["BMI", "AGE", "SEX"],
                    }
                ],
            },
            self.columns,
        )["task_spec"]
        invalid_graph = build_analysis_graph(
            task_spec,
            [
                {
                    "tool": "logistic_regression",
                    "outcome": "HYPERTENSION_SELF",
                    "predictors": ["AGE", "SEX"],
                }
            ],
        )
        validation = validate_analysis_graph(
            task_spec,
            invalid_graph,
            self.columns,
        )
        self.assertFalse(validation["valid"])
        self.assertTrue(
            any(
                "requested exposure" in item
                for item in validation["errors"]
            )
        )

    def test_auditor_requires_an_exposure_estimate(self):
        plan = reconcile_research_plan(
            (
                "Use HYPERTENSION_SELF as the outcome and BMI as the exposure; "
                "adjust for AGE and SEX with logistic regression and report OR."
            ),
            {
                "task_spec": {
                    "goal": "association",
                    "task_family": "inference",
                    "outcomes": ["HYPERTENSION_SELF"],
                    "exposures": ["BMI"],
                    "forced_covariates": ["AGE", "SEX"],
                    "requested_methods": ["logistic_regression"],
                    "required_outputs": ["odds_ratio"],
                },
                "tool_requests": [
                    {
                        "tool": "logistic_regression",
                        "outcome": "HYPERTENSION_SELF",
                        "predictors": ["BMI", "AGE", "SEX"],
                    }
                ],
            },
            self.columns,
        )
        graph = plan["analysis_graph"]
        node = graph["nodes"][0]
        manifest = ExecutionManifest(
            spec_hash=plan["task_spec"]["spec_hash"],
            graph_hash=graph["graph_hash"],
        )
        step = ExecutionStep(
            node_id=node["node_id"],
            capability=node["capability"],
            request=node["request"],
        )
        step.start()
        step.complete(
            {
                "declared_outputs": node["output_contract"],
                "actual_input_contract": {
                    "columns": ["HYPERTENSION_SELF", "BMI", "AGE", "SEX"],
                    "source": "tool_evidence",
                },
                "result_terms": ["const", "AGE", "SEX"],
            }
        )
        manifest.steps.append(step)
        manifest.finish()
        audit = deterministic_contract_audit(
            plan["task_spec"],
            graph,
            manifest.to_dict(),
            self.columns,
        )
        self.assertEqual(audit["decision"], "block")
        self.assertTrue(
            any(
                "does not contain an estimate" in item
                for item in audit["missing_requirements"]
            )
        )

    def test_feature_selection_auditor_checks_selected_feature_roles(self):
        plan = reconcile_research_plan(
            (
                "Use HYPERTENSION_SELF as the outcome. Use LASSO to return "
                "five stable candidate features while forcing AGE and SEX "
                "into the model."
            ),
            {
                "task_spec": {
                    "goal": "feature_discovery",
                    "task_family": "classification",
                    "outcomes": ["HYPERTENSION_SELF"],
                    "candidate_scope": "auto_discover",
                    "forced_covariates": ["AGE", "SEX"],
                    "requested_methods": ["lasso"],
                    "required_outputs": ["selected_features"],
                    "top_k": 5,
                },
                "tool_requests": [
                    {
                        "tool": "binary_model_comparison",
                        "outcome": "HYPERTENSION_SELF",
                        "features": [],
                        "methods": ["lasso_logistic"],
                        "top_k": 5,
                        "candidate_scope": "auto_discover",
                        "forced_covariates": ["AGE", "SEX"],
                    }
                ],
            },
            self.columns,
        )
        graph = plan["analysis_graph"]
        node = graph["nodes"][0]
        manifest = ExecutionManifest(
            spec_hash=plan["task_spec"]["spec_hash"],
            graph_hash=graph["graph_hash"],
        )
        step = ExecutionStep(
            node_id=node["node_id"],
            capability=node["capability"],
            request=node["request"],
        )
        step.start()
        step.complete(
            {
                "declared_outputs": node["output_contract"],
                "actual_input_contract": {
                    "columns": [
                        "HYPERTENSION_SELF",
                        "AGE",
                        "SEX",
                        "BMI",
                        "URIC_ACID",
                    ],
                    "source": "tool_evidence",
                },
                "result_terms": ["AGE", "BMI", "URIC_ACID"],
            }
        )
        manifest.steps.append(step)
        manifest.finish()
        audit = deterministic_contract_audit(
            plan["task_spec"],
            graph,
            manifest.to_dict(),
            self.columns,
        )
        self.assertEqual(audit["decision"], "block")
        self.assertTrue(
            any(
                "Forced covariates or outcomes" in item
                for item in audit["missing_requirements"]
            )
        )

    def test_feature_selection_auditor_accepts_exact_top_k_evidence(self):
        selected = [
            "BMI",
            "URIC_ACID",
            "EDUCATION",
            "FASTING_GLUCOSE",
            "HBA1C",
        ]
        plan = reconcile_research_plan(
            (
                "Use HYPERTENSION_SELF as the outcome. Use LASSO to return "
                "five stable candidate features while forcing AGE and SEX "
                "into the model."
            ),
            {
                "task_spec": {
                    "goal": "feature_discovery",
                    "task_family": "classification",
                    "outcomes": ["HYPERTENSION_SELF"],
                    "candidate_scope": "auto_discover",
                    "forced_covariates": ["AGE", "SEX"],
                    "requested_methods": ["lasso"],
                    "required_outputs": ["selected_features"],
                    "top_k": 5,
                },
                "tool_requests": [
                    {
                        "tool": "binary_model_comparison",
                        "outcome": "HYPERTENSION_SELF",
                        "features": [],
                        "methods": ["lasso_logistic"],
                        "top_k": 5,
                        "candidate_scope": "auto_discover",
                        "forced_covariates": ["AGE", "SEX"],
                    }
                ],
            },
            self.columns,
        )
        graph = plan["analysis_graph"]
        node = graph["nodes"][0]
        manifest = ExecutionManifest(
            spec_hash=plan["task_spec"]["spec_hash"],
            graph_hash=graph["graph_hash"],
        )
        step = ExecutionStep(
            node_id=node["node_id"],
            capability=node["capability"],
            request=node["request"],
        )
        step.start()
        step.complete(
            {
                "declared_outputs": node["output_contract"],
                "actual_input_contract": {
                    "columns": [
                        "HYPERTENSION_SELF",
                        "AGE",
                        "SEX",
                        *selected,
                    ],
                    "source": "tool_evidence",
                },
                "result_terms": selected,
            }
        )
        manifest.steps.append(step)
        manifest.finish()
        audit = deterministic_contract_audit(
            plan["task_spec"],
            graph,
            manifest.to_dict(),
            self.columns,
        )
        self.assertEqual(audit["decision"], "pass")

    def test_mixed_lasso_and_literature_outputs_validate_separate_branches(self):
        plan = reconcile_research_plan(
            (
                "以 BREAST_CA_SELF 為 outcome，用 LASSO 找出前五個穩定候選"
                "特徵，並與 PubMed 文獻比較。"
            ),
            {
                "task_spec": {
                    "goal": "feature_discovery",
                    "task_family": "classification",
                    "outcomes": ["BREAST_CA_SELF"],
                    "candidate_scope": "auto_discover",
                    "requested_methods": ["lasso"],
                    "required_outputs": [
                        "selected_features",
                        "pmid_list",
                        "literature_evidence",
                        "evidence_limitation_summary",
                    ],
                    "top_k": 5,
                },
                "tool_requests": [
                    {
                        "tool": "binary_model_comparison",
                        "outcome": "BREAST_CA_SELF",
                        "features": [],
                        "methods": ["lasso_logistic"],
                        "top_k": 5,
                        "candidate_scope": "auto_discover",
                    }
                ],
                "needs_literature": True,
            },
            self.columns,
        )
        self.assertTrue(plan["contract_validation"]["valid"])
        self.assertTrue(plan["capability_fit"]["complete"])
        self.assertTrue(plan["needs_literature"])
        self.assertEqual(
            plan["task_spec"]["required_outputs"],
            [
                "selected_features",
                "pmid_list",
                "literature_evidence",
                "evidence_limitation_summary",
            ],
        )
        graph_outputs = {
            output
            for node in plan["analysis_graph"]["nodes"]
            for output in node["output_contract"]
        }
        self.assertIn("selected_features", graph_outputs)
        self.assertNotIn("pmid_list", graph_outputs)
        self.assertNotIn("literature_evidence", graph_outputs)
        self.assertNotIn("evidence_limitation_summary", graph_outputs)
        self.assertEqual(
            plan["evidence_contract"]["required_outputs"],
            [
                "pmid_list",
                "literature_evidence",
                "evidence_limitation_summary",
            ],
        )
        self.assertEqual(
            plan["evidence_contract"]["citation_policy"],
            "retrieved_records_only",
        )


if __name__ == "__main__":
    unittest.main()
