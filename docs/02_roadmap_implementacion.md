# Roadmap de Implementacion

## Fase 0: Baseline

1. Congelar baseline actual.
2. Registrar metricas iniciales:
   - score,
   - latencia,
   - tasa de errores de formato,
   - tasa de preguntas innecesarias.

## Fase 1: Motor estrategico

1. Implementar `BeliefState` + actualizacion bayesiana.
2. Implementar `OpponentModel`.
3. Implementar `UtilityPlanner`.
4. Implementar `MixedStrategyPolicy`.

## Fase 2: Capa epistemica y mental

1. Integrar `VOIGate`.
2. Integrar `MindModel` (BDI).
3. Integrar `MetaReasoner`.

## Fase 3: Hardening

1. Validacion estricta de salida (`OutputValidator`).
2. Fallback determinista seguro.
3. Manejo robusto de incertidumbre alta.

## Fase 4: Evaluacion y tuning

1. Pruebas adversariales por perfil de oponente.
2. A/B test de politicas (v1 vs v2).
3. Calibracion de:
   - temperatura,
   - lambdas de utilidad,
   - umbral VOI.

## Criterio de exito

- Mejor score esperado sin incremento de errores de formato.
- Menor explotabilidad observada frente a rivales adaptativos.
- Menor varianza entre rondas bajo ambiguedad.
