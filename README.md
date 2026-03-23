# Purple Agent v3 -- Build What I Mean

**Team TRIBUIA** | AgentBeats Competition 2026

## Overview

Purple Agent v3 is a hybrid LLM + deterministic spatial reasoning agent for the *Build What I Mean* game. The agent interprets natural language instructions to build block structures on a 9x9 3D grid, balancing accuracy against the cost of asking clarifying questions.

Our core insight: **offload spatial computation to deterministic code, not the LLM.** The LLM handles only language understanding (ambiguity detection + instruction decomposition into atomic steps). All coordinate arithmetic, stacking, validation, and auto-corrections are deterministic -- eliminating the model's worst failure modes.

## Architecture

```
                    Natural Language Instruction
                              |
                              v
                  +-------------------------+
                  |   Instruction Parser    |  Parse [TASK_DESCRIPTION],
                  |                         |  [SPEAKER], [START_STRUCTURE]
                  +-------------------------+
                              |
                              v
                  +-------------------------+
                  |   Ambiguity Detector    |  BAML + GPT-4o-mini
                  |   (LLM-based)          |  Detect missing color/count
                  +-------------------------+
                         /          \
                   [ASK]              [continue]
                   (if needed)            |
                              v           v
                  +-------------------------+
                  |  Structure Analyzer     |  Detect lines, stacks,
                  |  (deterministic)        |  L-shapes, T-shapes
                  +-------------------------+
                              |
                              v
                  +-------------------------+
                  |    Build Planner        |  BAML + GPT-4o-mini
                  |    (LLM-based)         |  Instruction -> JSON steps
                  +-------------------------+
                              |
                              v
                  +-------------------------+
                  |  Deterministic Fixes    |  Chain ref patching,
                  |  (no LLM)              |  direction fix, T-shape fix,
                  |                         |  each-end cap recomputation
                  +-------------------------+
                              |
                              v
                  +-------------------------+
                  |   Plan Verifier         |  Direction consistency,
                  |   (deterministic)       |  block count sanity,
                  |                         |  ground-level validation
                  +-------------------------+
                              |
                              v
                  +-------------------------+
                  |   Spatial Executor      |  Deterministic execution
                  |   (no LLM)             |  on Grid model (y auto-computed)
                  +-------------------------+
                              |
                              v
                  +-------------------------+
                  |   Response Formatter    |  [BUILD];Color,x,y,z;...
                  |   + Validator           |  Coordinate bounds check
                  +-------------------------+
                              |
                              v
                       [BUILD] response
```

## Key Design Decisions

### 1. Two-Dimensional LLM Thinking

The LLM only reasons about **x,z coordinates** (the horizontal plane). Y-coordinates are **auto-computed** by the deterministic executor using gravity-based stacking:

- Ground level: y=50
- Each additional block at same (x,z): y += 100
- Max height: 5 blocks (y=450)

This eliminates off-by-one y-coordinate errors, stack height miscounting, and vertical arithmetic mistakes -- the most common LLM spatial reasoning failures.

### 2. BAML for Structured LLM Outputs

We use [BAML](https://docs.boundaryml.com/) (Boundary ML) for type-safe LLM function calls:

- **`DetectAmbiguity`**: Returns typed `AmbiguityResult` with boolean flags, suggested questions, and inferred defaults
- **`DecomposeBuildInstruction`**: Returns typed `BuildPlan` with `BuildStep[]` -- each step has a validated action enum, color, count, position, and direction

BAML provides automatic JSON schema generation, retry logic, and type validation -- reducing parsing failures to near zero.

### 3. EV-Based ASK Strategy

Asking a clarifying question costs -5 points. We use expected value analysis to decide when asking is worth it:

```
EV(ask)  = -5 + P(correct_with_answer) * 10
EV(guess) = P(correct_guess) * 10 - P(wrong_guess) * 10
```

**Rules derived from data analysis:**
- **Never ask about count** -- infer from adjacent stack height or default to 3. Count questions have the worst ROI (-5 cost, ~65% heuristic accuracy makes guessing better)
- **Only ask about color** when: instruction has 0 colors AND grid has 2+ colors, OR instruction has 2+ colors with a colorless phrase
- **Hard cap at 20%** -- if ASK rate exceeds 20% of rounds, switch to always-build mode
- **Never ask twice** per instruction round

### 4. Deterministic Post-Processing Pipeline

After the LLM produces build steps, four deterministic correction passes run:

1. **Chain Reference Patching** -- "the green one" resolves to where green was *last placed*, not its original position
2. **Direction Auto-Fix** -- if instruction says "left" but LLM said "right", flip it
3. **Each-End Cap Recomputation** -- after extending a row, recalculate endpoints for "each end" placements
4. **T-Shape Stem Fix** -- detect T-shapes, compute correct stem extension direction away from junction

### 5. Cross-Round Learning

The agent maintains conversation history and a feedback cache:

- **History context** (8 rounds) is fed to the planner LLM, allowing it to learn from previous rounds
- **Feedback cache** stores failed instruction patterns (normalized by stripping numbers and colors) and provides correction hints when similar patterns reappear
- **Round separators** in history prevent output contamination between rounds

## Theoretical Foundation

### Game Theory

The competition is modeled as a **repeated game with incomplete information**:

- **Bayesian updating**: beliefs about instruction ambiguity are updated based on observed patterns
- **Value of Information (VOI)**: each ASK decision is evaluated against its expected information gain vs. cost
- **Mixed strategy equilibrium**: the agent's behavior is non-deterministic (stochastic LLM outputs + deterministic corrections), reducing exploitability by predictable patterns
- **Multi-objective utility**: `U = E[score] - lambda_risk * variance - lambda_latency * latency - lambda_format * format_error`

### Philosophy of Mind

- **BDI Architecture** (Beliefs, Desires, Intentions): the agent maintains explicit beliefs about the grid state, desires to maximize score, and intentions as concrete build plans
- **Metacognition**: the plan verifier acts as a metacognitive audit -- checking whether the plan is consistent with the instruction before execution
- **Epistemic Discipline**: the system distinguishes between *facts* (parsed grid state), *inferences* (LLM-generated plans), and *conjectures* (inferred colors/counts), treating each with appropriate confidence

## Evolution: v1 -> v2 -> v3

### v1: Pure LLM
- Direct GPT call, no structured pipeline
- Low accuracy, frequent format errors

### v2: Hybrid Pipeline (game theory + deterministic executor)
- Added structured pipeline: parse -> underspec detect -> plan -> execute
- Regex-based ambiguity detection (never fired on real instructions)
- No history integration
- 12 synthetic worked examples

### v3: BAML-Powered (current)
- **LLM-based ambiguity detection** via BAML (replaces broken regex)
- **9 worked examples** from actual game patterns
- **History wired into planner** for cross-round learning
- **EV-based ASK strategy** with hard cap
- **5 deterministic bug fixes** from post-mortem analysis:
  - Purple default color eliminated (use context inference)
  - extend_row off-by-one fixed (skip duplicates, not auto-advance)
  - Ground-level validator added (prevent accidental stacking)
  - T-shape cap positions fixed at crossbar endpoints
  - Answer re-verification guard (detect aberrant post-answer plans)

## Post-Mortem Driven Development

After each competition run, we performed rigorous post-mortem analysis:

### PR #15 Analysis (GPT-4o-mini, score: +100)
- Identified 5 root causes across 42 error rounds
- Categorized errors: block count (33%), Purple default (26%), position (24%), color (12%), empty (5%)
- Implemented targeted fixes for each category

### Combined Analysis (GPT-4o-mini vs GPT-4o)
- **Key finding**: GPT-4o-mini (+100) outperformed GPT-4o (+35) by 65 points
- GPT-4o asks 70% more questions and "over-thinks" spatial positions
- **Insight**: for structured pipelines with deterministic execution, smaller literal models outperform larger creative models
- Led to EV-based ASK strategy and hard cap implementation

## Model Selection Rationale

We use **GPT-4o-mini** for both LLM calls (ambiguity detection + planning). Empirical data shows:

| Metric | GPT-4o-mini | GPT-4o |
|---|---|---|
| Final score | **+100** | +35 |
| Accuracy | **71.25%** | 63.13% |
| ASK rate | **31.5%** | 53.9% |
| Position errors | **10** | 23 |

The deterministic pipeline compensates for the smaller model's limitations while avoiding the larger model's tendency to over-reason and produce creative but incorrect spatial outputs.

## Technical Stack

| Component | Technology |
|---|---|
| Language | Python 3.11+ |
| LLM Client | BAML (Boundary ML) v0.220.0 |
| LLM Model | GPT-4o-mini (OpenAI) |
| Agent Protocol | A2A (Google Agent-to-Agent) |
| Server | Uvicorn + Starlette |
| Package Manager | uv (Astral) |
| Container | Docker (ghcr.io/astral-sh/uv:python3.13-bookworm) |
| Testing | pytest (35 tests) |

## Project Structure

```
src/
  server.py                    # Entry point (delegates to purple_v3)
  purple_v3/
    server.py                  # A2A server, pipeline orchestration, history
    ambiguity_detector.py      # BAML LLM-based ambiguity detection
    build_planner.py           # BAML LLM-based instruction decomposition
    grid.py                    # 9x9 grid model with gravity stacking
    instruction_parser.py      # Parse green agent messages
    structure_analyzer.py      # Detect lines, stacks, L/T shapes
    spatial_executor.py        # Deterministic plan execution
    plan_verifier.py           # Post-plan verification + auto-fixes
    plan_patcher.py            # Chain reference coordinate patching
    prompt_enricher.py         # 18 contextual enrichment rules
    response_formatter.py      # [BUILD] response formatting + validation
    baml_src/                  # BAML definitions
      clients.baml             # LLM client configuration
      types.baml               # Shared types (BuildStep, AmbiguityResult)
      ambiguity.baml           # Ambiguity detection function
      planner.baml             # Build step decomposition function
    baml_client/               # Auto-generated BAML Python client
tests/
  test_v3_grid.py              # Grid model tests (10 tests)
  test_v3_ambiguity.py         # Ambiguity detection tests (9 tests)
  test_v3_spatial.py           # Spatial executor tests (6 tests)
  test_v3_structure.py         # Structure analyzer + verifier tests (10 tests)
```

## Running Locally

```bash
# Install dependencies
uv sync

# Run tests
PYTHONPATH=src uv run python -m pytest tests/test_v3_*.py -v

# Start server
OPENAI_API_KEY=sk-... uv run python src/server.py --host 0.0.0.0 --port 9018 --debug

# Docker
docker build -t purple-agent-v3 .
docker run -e OPENAI_API_KEY=sk-... -p 9018:9018 purple-agent-v3
```

## Team

- **Team TRIBUIA**
- Julian Anibal Henao ([@julianAnibal](https://github.com/julianAnibal))
- Daniel Santiago Sandoval

## License

Competition submission for AgentBeats 2026 -- Build What I Mean.
