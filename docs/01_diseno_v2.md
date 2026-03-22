# Diseno Tecnico v2

## 1) Objetivo

Disenar un agente competitivo robusto en entornos con ambiguedad y rivales adaptativos.

**Meta realista**: maximizar rendimiento esperado minimizando explotabilidad.

## 2) Funcion de utilidad

Se define una utilidad compuesta por accion:

`U = E[score] - lambda_riesgo * varianza - lambda_tiempo * latencia - lambda_formato * error_formato`

Donde:

- `E[score]`: valor esperado de puntos.
- `varianza`: volatilidad estrategica no deseada.
- `latencia`: costo por respuestas lentas.
- `error_formato`: riesgo de invalidez de salida.

## 3) Teoria de juegos aplicada

- Juego repetido con informacion parcial.
- Actualizacion bayesiana de tipos de rival.
- Estrategias mixtas para evitar explotacion por patrones fijos.
- VOI (Value of Information) para decidir preguntar o actuar.

### Regla VOI

Preguntar solo si:

`EV(preguntar) - EV(no_preguntar) > costo_pregunta`

## 4) Filosofia de la mente aplicada

- **BDI**:
  - Beliefs: que creo sobre mundo y rival.
  - Desires: que quiero optimizar.
  - Intentions: que voy a ejecutar ahora.
- **Metacognicion**: auditoria interna de supuestos y contradicciones.
- **Disciplina epistemica**:
  - Hecho,
  - Inferencia,
  - Conjetura.

## 5) Arquitectura modular

1. `belief_state.py`
2. `opponent_model.py`
3. `voi_gate.py`
4. `planner.py`
5. `mixed_policy.py`
6. `mind_model.py`
7. `meta_reasoner.py`
8. `output_validator.py`
9. `agent.py`

## 6) Principios de robustez

- Salidas validadas y fallback seguro.
- Estocasticidad controlada por temperatura.
- Separacion estricta entre estimacion, decision y formato.
- Evaluacion adversarial continua.
