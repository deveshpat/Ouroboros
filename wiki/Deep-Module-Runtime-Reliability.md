# Deep Modules -> Runtime Reliability

Goal -> small public surface, deep internal safety.

| Seam | Owner | Reason |
|---|---|---|
| runtime readiness | Bootstrap | one setup contract before heavy imports |
| hard lessons | Bootstrap guardrails | known failures -> executable classifier/test |
| training session | Coconut | train/DGAC/checkpoint hidden behind package root |
| model loading | Models | HF quirks stay internal |
| dispatch/aggregation | Coordinator | orchestration decisions stay together |
| provider IO | Utils | helpers do not decide workflow |
| eval gates | Eval | quality checks live one place |

Rule -> new helper must deepen owner module, not widen public API.
