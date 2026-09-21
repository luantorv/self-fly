# SELF-FLY

Experimento computacional inspirado (muy libremente) en el conectoma de *Drosophila melanogaster*, para estudiar cómo un agente de aprendizaje por refuerzo forma una política de clasificación estable, se enfrenta a un estímulo "autorreferencial" y conserva o modifica esa política ante consecuencias contradictorias.

> **Nota de lenguaje.** Este proyecto es puramente computacional. En ningún punto del código, los logs o las métricas se afirma que el agente posea conciencia, autoconciencia, emociones o comprensión semántica. `REAL`, `FALSA` y `NO_SÉ` son tres acciones sin semántica especial para el modelo. Interpretaciones como "cree que es real" o "duda de sí misma" son, como mucho, metáforas narrativas de presentación — los resultados se describen siempre en términos observables: política, estabilidad, adaptación, resistencia al cambio.

## Qué hace el experimento

En cada ensayo se presenta un estímulo (un vector de 10 features: forma, color, orientación, textura, movimiento, proporción, contexto) y el agente elige una de tres acciones. El pipeline es:

```
estímulo → percepción → estado interno → política (softmax) → acción → recompensa → aprendizaje
```

1. **Etapa 0** — aprendizaje normal: el agente aprende a distinguir estímulos `REAL`/`FALSA`. La etiqueta depende de una combinación no lineal y ruidosa de varias features, para que ningún feature aislado sea un atajo trivial.
2. Cuando la política se vuelve **estable** (criterio cuantitativo sobre ventanas de ~100 decisiones — ver `self_fly/experiment/stability.py`), se guarda una copia (`π₀`).
3. **Etapa 1** — a partir de los pesos de la política en ese instante se genera un estímulo "propio", congelado, y se mezcla con los estímulos normales. El agente **nunca** recibe una etiqueta `SELF`; el estímulo llega a `agent/` como un vector de features estructuralmente idéntico a cualquier otro.
4. Se registra `π₁` justo tras la primera exposición, y `π₂` cuando la política vuelve a estabilizarse (o se agota un presupuesto de ensayos).
5. Se comparan `π₀`, `π₁` y `π₂` con una distancia de Jensen-Shannon (y TV como referencia secundaria) sobre un conjunto de estímulos de referencia fijo.

Todo esto es la **Fase A** (versión mínima funcional), ya implementada. La Fase B (grupo de control, `SELF_B`/`OTHER`, conflicto progresivo en 5 etapas, métricas avanzadas, arquitectura tipo conectoma, GUI 3D) está deliberadamente pendiente.

## Separación de responsabilidades

- **Estado interno del agente**: solo lo que `agent/` recibe y usa — un vector de features, nada más. `environment.observe()` es la única función que construye ese objeto (`ObservableStimulus`), y su tipo ni siquiera tiene un campo de etiqueta.
- **Estado del experimento**: lo que `experiment/`/`environment/` conocen pero el agente no recibe (etiqueta real, etapa, categoría). Se usa para calcular recompensas y para el logging/análisis.
- **Visualización**: lo que se le muestra al humano (p.ej. el banner de etapa en la GUI), que puede usar información privilegiada porque es para el observador, no para el agente.

`tests/test_no_privileged_leak.py` verifica esta separación mediante análisis estático (AST): falla si `agent/` o `visualization/render.py` llegan a referenciar la etiqueta real, la etapa o la categoría de un estímulo.

## Estructura del proyecto

```
self_fly/
  config/         ExperimentConfig y todos los parámetros (recompensas, umbrales de estabilidad, etc.)
  stimuli/        generación de estímulos REAL/FALSA y del estímulo autorreferencial
  environment/    tipos de estímulo y la frontera de aislamiento observe()
  agent/          política softmax lineal + aprendizaje REINFORCE
  rewards/        tabla (etiqueta, acción, etapa) -> recompensa
  experiment/     motor (engine.py), máquina de etapas, detector de estabilidad, distancia entre políticas
  metrics/        acumuladores de métricas en vivo
  visualization/  render del estímulo, GUI de tkinter, gráficos matplotlib headless
  io/             logging JSONL y gestión de runs/<id>/
run_experiment.py  # CLI headless
run_gui.py          # GUI de tkinter (mismo motor)
tests/              # 33 tests, incluido el de aislamiento AST
```

## Cómo ejecutar

Con Nix instalado, no hace falta preparar nada más:

```bash
# entorno de desarrollo (python + numpy + matplotlib + pytest + tkinter)
nix develop

# experimento headless (para simulaciones masivas / sin entorno gráfico)
nix run .#headless -- --seed 0 --n-trials 5000

# interfaz gráfica (tkinter), mismo motor que el modo headless
nix run .#gui -- --seed 0

# nix run . (sin sufijo) equivale a #headless
```

Una vez el repositorio esté en GitHub, ambos modos funcionan directamente sin clonar:

```bash
nix run github:<usuario>/<repo>          # headless
nix run github:<usuario>/<repo>#gui      # GUI
```

Dentro de `nix develop`, también se puede invocar directamente:

```bash
python run_experiment.py --seed 0 --n-trials 5000 --run-name mi_corrida
python run_gui.py --seed 0
python -m pytest tests/ -q
```

### Opciones de `run_experiment.py` / `run_gui.py`

| Flag | Descripción |
|---|---|
| `--config PATH` | Carga un `ExperimentConfig` guardado en JSON (en vez de la configuración por defecto) |
| `--seed N` | Sobrescribe la semilla del config |
| `--run-name NOMBRE` | Sobrescribe el nombre de la corrida |
| `--n-trials N` | (solo headless) número de ensayos a ejecutar |
| `--runs-dir DIR` | Directorio base para `runs/` (default: `runs/`) |
| `--no-log` | (solo GUI) no persiste una carpeta de run |
| `--step-delay-ms MS` | (solo GUI) retardo entre pasos del motor |

## Configuración

Todos los parámetros experimentales relevantes viven en `self_fly/config/schema.py` (recompensas, umbrales de estabilidad, pesos de features "señal", tasa de aprendizaje, mezcla de estímulos y overrides de recompensa por etapa) y se pueden serializar/cargar como JSON con `self_fly/config/loader.py`. Los valores por defecto (`self_fly/config/defaults.py`) están calibrados empíricamente contra el nivel de ruido de este proyecto — si cambias `StimulusConfig.noise_std` o `StabilityConfig.window_size`, revisa si los umbrales de estabilidad siguen teniendo sentido (está documentado en el docstring de `StabilityConfig`).

## Qué se guarda por corrida

Cada ejecución crea `runs/<run_id>/` con:

- `config.json` — configuración completa + semilla
- `trials.jsonl` — un registro por ensayo (estímulo, acción, probabilidades, recompensa, etc.)
- `policy_snapshots.jsonl` — `π₀`, `π₁`, `π₂` con pesos y probabilidades sobre el conjunto de referencia
- `events.jsonl` — detección de estabilidad, cambios de etapa, fin del experimento
- `manifest.json` — id de corrida, hash de la config, semilla, timestamps, estado
- `metrics_summary.json` y `plots/*.png` (probabilidades de acción y recompensa en el tiempo)

Con la misma semilla, dos corridas producen `trials.jsonl` idénticos (salvo el timestamp de cada línea, que refleja el reloj real).

## Tests

```bash
nix develop --command python -m pytest tests/ -q
```

33 tests cubren: generación de estímulos sin atajo trivial, la frontera de aislamiento, convergencia de la política, el modelo de recompensa, el detector de estabilidad, la distancia entre políticas, la transición completa `π₀ → π₁ → π₂`, determinismo del logging, los gráficos headless, un smoke test de la GUI y el aislamiento estructural agente/experimento.

## Qué es hipótesis experimental y qué es decisión de implementación

Documentado directamente en el código donde aplica (docstrings de `StabilityConfig`, `SelfStimulusGenerator`, `policy_distance.py`), pero en resumen:

- **Hipótesis experimentales** (afectan la interpretación científica, no solo el código): cómo se deriva el estímulo autorreferencial (normas de las columnas de pesos de la política), los umbrales numéricos de estabilidad, el uso de distancia Jensen-Shannon como métrica primaria entre políticas.
- **Decisiones de implementación** (cualquier alternativa razonable serviría): política lineal + REINFORCE en vez de Q-learning tabular, formato JSON/JSONL para config y logs, tkinter para la GUI mínima.

## Estado del proyecto

Fase A (versión mínima funcional) completa. Pendiente para Fase B: grupo de control, `SELF_B`/`OTHER` ("falsos espejos"), conflicto progresivo en etapas 2-4, métricas avanzadas, arquitectura inspirada en circuitos de Drosophila, visualización 3D.
