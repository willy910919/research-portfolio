"""New metadata-only example using the unchanged original plan compiler."""
import json
from research_room import compile_lead_plan, verify_user_requirement

plan = compile_lead_plan(
    '請描述 AGE 與 BMI。',
    {'task_spec': {'goal': 'descriptive', 'required_outputs': ['summary_statistics']},
     'tool_requests': [{'tool': 'describe', 'columns': ['AGE', 'BMI']}]},
    ['AGE', 'BMI'],
)
assert verify_user_requirement(plan['user_requirement'])
assert plan['analysis_graph']['nodes']
assert plan['contract_validation']['valid'], plan['contract_validation']
print(json.dumps({
    'synthetic_metadata_only': True,
    'requirement_hash': plan['user_requirement']['requirement_hash'],
    'analysis_graph': plan['analysis_graph'],
    'contract_validation': plan['contract_validation'],
    'limitation': 'Plan compilation only; statistical executors and patient data are not included.',
}, ensure_ascii=False, indent=2))
