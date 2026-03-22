# Agente Purpura v2 (Shareable)

Implementacion base de un **Agente Purpura** orientado a competencia, con foco en:

- **Teoria de juegos**: juego repetido, informacion parcial, estrategia mixta, valor de informacion (VOI), minimizacion de explotabilidad.
- **Filosofia de la mente**: modelo BDI (Beliefs, Desires, Intentions), metacognicion, disciplina epistemica (hecho vs inferencia vs conjetura).

## Contenido

- `docs/`: diseno, roadmap y matriz de evaluacion.
- `src/purple_v2/`: modulos del agente.
- `tests/`: suite de pruebas (`unittest`).
- `scripts/demo_run.py`: demo local de decisiones por ronda.
- `scripts/package_share.sh`: crea zip listo para compartir.

## Quickstart

```bash
cd agente_purpura_v2_shareable
python3 -m unittest discover -s tests -p "test_*.py" -v
python3 scripts/demo_run.py
```

## Arquitectura (resumen)

1. `BeliefState`: actualiza creencias sobre tipo de oponente via Bayes.
2. `OpponentModel`: proyecta conducta probable del rival.
3. `VOIGate`: decide si conviene preguntar aclaracion.
4. `UtilityPlanner`: calcula utilidad esperada por accion.
5. `MixedStrategyPolicy`: selecciona accion estocastica controlada.
6. `MindModel (BDI)`: modela estado mental propio y del rival.
7. `MetaReasoner`: audita supuestos y contradicciones.
8. `OutputValidator`: evita respuestas invalidas.

## Como compartir

```bash
cd agente_purpura_v2_shareable
bash scripts/package_share.sh
```

Genera: `agente_purpura_v2_shareable.zip`

## Nota estrategica

"Imparable" no existe en entornos adversariales. El objetivo real es:

- alta adaptabilidad,
- baja explotabilidad,
- estabilidad de formato,
- y mejora continua bajo evaluacion adversarial.
