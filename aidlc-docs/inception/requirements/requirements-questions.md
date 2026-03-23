# Requirements Clarification Questions

Please answer the following questions to help clarify the approach.

## Question 1
Which approach should we take for fixing the agent?

A) Fix the 3 bugs in-place (faster, lower risk, keeps working deployment)
B) Full refactor into new folder with BAML (cleaner but higher risk, more time)
C) Hybrid: fix bugs in-place AND add BAML for the underspec detection LLM call only
D) Other (please describe after [Answer]: tag below)

[Answer]: B) Full refactor into new folder with BAML (cleaner but higher risk, more time)


## Question 2
Should security extension rules be enforced for this project?

A) Yes - enforce all SECURITY rules as blocking constraints (recommended for production-grade applications)
B) No - skip all SECURITY rules (suitable for PoCs, prototypes, and experimental projects)
C) Other (please describe after [Answer]: tag below)

[Answer]: A) Yes - enforce all SECURITY rules as blocking constraints (recommended for production-grade applications)


## Question 3
What model should the agent use for LLM calls?

A) GPT-4o (current setting - best spatial reasoning, higher cost)
B) GPT-4o-mini (competitor uses this - cheaper, good enough with our deterministic pipeline)
C) Claude Sonnet (via Anthropic API or OpenAI-compatible endpoint)
D) Keep GPT-4o as primary, add GPT-4o-mini as fallback
E) Other (please describe after [Answer]: tag below)

[Answer]: B) GPT-4o-mini (competitor uses this - cheaper, good enough with our deterministic pipeline)


## Question 4
How aggressive should we be with asking questions ([ASK])?

A) Ask whenever ANY ambiguity is detected (maximize information, accept -5 cost)
B) Only ask when EV(ask) > EV(guess) based on probability analysis (balanced)
C) Never ask - always build best guess (avoid -5 penalty entirely)
D) Other (please describe after [Answer]: tag below)

[Answer]:  A) Ask whenever ANY ambiguity is detected (maximize information, accept -5 cost)

