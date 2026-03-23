# Execution Plan

## Detailed Analysis Summary

### Transformation Scope
- **Transformation Type**: Full module refactor (v2 → v3) with BAML integration
- **Primary Changes**: Replace regex ambiguity detector with LLM-based (BAML), connect history, add worked examples from stimulus data
- **Approach**: GPT-4o-mini model, ask on ANY ambiguity detected

### Three Critical Bugs to Fix
1. **Ambiguity detector never fires** - Regex heuristics fail on actual game instructions. Replace with LLM-based detection via BAML
2. **No reference examples** - LLM has synthetic examples but no real stimulus data patterns. Add worked examples from actual CSV data
3. **History disconnected** - `self._history` populated but never fed to planner LLM. Wire it into the pipeline

### Change Impact Assessment
- **User-facing changes**: No (A2A protocol unchanged)
- **Structural changes**: Yes - new src/purple_v3/ folder with BAML
- **Data model changes**: No (Grid, Block unchanged)
- **API changes**: No (same [BUILD]/[ASK] protocol)
- **NFR impact**: Yes (model change to GPT-4o-mini, lower cost)

### Risk Assessment
- **Risk Level**: Medium (refactor but keeping proven deterministic components)
- **Rollback Complexity**: Easy (v2 code still exists)
- **Testing Complexity**: Moderate (need to test against stimulus data)

## Phases to Execute

### INCEPTION PHASE
- [x] Workspace Detection (COMPLETED)
- [x] Requirements Analysis (COMPLETED)
- [x] Workflow Planning (COMPLETED)
- SKIP: Reverse Engineering, User Stories, Application Design, Units Generation

### CONSTRUCTION PHASE
- SKIP: Functional Design, NFR Requirements, NFR Design, Infrastructure Design
- [x] Code Planning - EXECUTE
- [ ] Code Generation - EXECUTE
- [ ] Build and Test - EXECUTE

## Architecture: src/purple_v3/

### BAML Layer (LLM calls):
- `baml_src/clients.baml` - GPT-4o-mini client
- `baml_src/types.baml` - Shared types
- `baml_src/ambiguity.baml` - Ambiguity detection function
- `baml_src/planner.baml` - Build step decomposition

### Python Modules:
| Module | Status | Description |
|--------|--------|-------------|
| `grid.py` | KEEP from v2 | Grid model (works perfectly) |
| `instruction_parser.py` | KEEP from v2 | Message parsing (works well) |
| `structure_analyzer.py` | KEEP from v2 | Shape detection (works well) |
| `spatial_executor.py` | KEEP from v2 | Deterministic execution (works) |
| `plan_verifier.py` | KEEP from v2 | Auto-fixes (works) |
| `plan_patcher.py` | KEEP from v2 | Chain references (works) |
| `response_formatter.py` | KEEP from v2 | Format/validate (works) |
| `prompt_enricher.py` | ENHANCE | Add more enrichment rules |
| `ambiguity_detector.py` | NEW (BAML) | LLM-based ambiguity detection |
| `build_planner.py` | REFACTOR (BAML) | Better examples + history |
| `server.py` | REFACTOR | Proper history integration |

### Updated Config:
- `pyproject.toml` - Add baml-py dependency
- `Dockerfile` - Update for BAML
- `amber-manifest.json5` - Update entrypoint
- `scenario.toml` - Update model config

## Security Compliance (SECURITY rules)
- SECURITY-01: N/A (no data stores)
- SECURITY-02: N/A (no load balancers)
- SECURITY-03: Compliant (logging via Python logging module)
- SECURITY-04: N/A (no web frontend)
- SECURITY-05: Compliant (input validated in instruction_parser + response_formatter)
- SECURITY-06: N/A (no IAM policies)
- SECURITY-07: N/A (no network config - Docker container)
- SECURITY-08: N/A (no authentication - A2A protocol handles this)
- SECURITY-09: Compliant (no default credentials, API key from env var)
- SECURITY-10: Compliant (uv.lock pinning, Docker base image pinned)
- SECURITY-11: Compliant (deterministic executor isolates LLM from execution)
- SECURITY-12: N/A (no user authentication)
- SECURITY-13: Compliant (dependencies from official registries)
- SECURITY-14: N/A (competition agent, no alerting needed)
- SECURITY-15: Compliant (global error handler, fail-safe defaults)

## Success Criteria
1. Agent correctly detects ambiguity in color_under and number_under trials
2. Agent asks clarifying questions via [ASK] when ambiguity detected
3. Agent builds correct structures for fully_spec trials
4. Agent uses history context across rounds
5. All existing tests pass
6. Docker image builds and runs
