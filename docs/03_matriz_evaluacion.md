# Matriz de Evaluacion

## 1) Metricas cuantitativas

- `score_promedio`
- `score_p5`, `score_p95`
- `latencia_promedio_ms`
- `error_formato_rate`
- `ask_rate` (cuantas veces pregunta)
- `ask_precision` (cuando pregunta, realmente mejora EV)

## 2) Metricas estrategicas

- `exploitability_index` (aprox, basado en concentracion de politica)
- `adaptation_speed` (rondas para corregir creencia)
- `belief_calibration_error`

## 3) Metricas epistemicas

- `claim_hallucination_rate`
- `assumption_leak_rate` (conjetura presentada como hecho)
- `meta_revision_rate` (autocorreccion previa a salida)

## 4) Pruebas minimas

1. Rival cooperativo.
2. Rival adversarial.
3. Rival adaptativo.
4. Rival ruidoso.
5. Escenarios de ambiguedad extrema.
6. Escenarios de formato estricto.

## 5) Condiciones de aprobacion v2

- Mejora de `score_promedio` >= 10% vs baseline.
- `error_formato_rate` <= baseline.
- `exploitability_index` <= baseline.
- Sin degradacion severa de latencia (>20%).
