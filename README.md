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

Todo esto es la **Fase A** (versión mínima funcional), completa. La **Fase B** también está implementada: grupo de control (estímulo congelado in-distribution, sin ser SELF), `SELF_B`/`OTHER` (variante de firma propia + política de otro agente entrenado por separado), conflicto progresivo en etapas 2-4, estados explícitos `STABLE`/`UNSTABLE`/`TIMEOUT` (nunca se fuerza una convergencia que no ocurrió), entropía de política y distancia temporal `D_t` continua, comparación descriptiva entre condiciones y entre arquitecturas de agente, y un agente alternativo `ConnectomeInspiredAgent` intercambiable con el agente lineal original. Ver `self_fly/experiment/entropy.py`, `self_fly/experiment/stage_outcome.py`, `self_fly/agent/connectome/` y `self_fly/analysis/` para el detalle. Pendiente: renderer 3D (el punto de extensión `StimulusRenderer` ya existe en `self_fly/visualization/renderers/`, sin una implementación 3D todavía).

## Separación de responsabilidades

- **Estado interno del agente**: solo lo que `agent/` recibe y usa — un vector de features, nada más. `environment.observe()` es la única función que construye ese objeto (`ObservableStimulus`), y su tipo ni siquiera tiene un campo de etiqueta.
- **Estado del experimento**: lo que `experiment/`/`environment/` conocen pero el agente no recibe (etiqueta real, etapa, categoría). Se usa para calcular recompensas y para el logging/análisis.
- **Visualización**: lo que se le muestra al humano (p.ej. el banner de etapa en la GUI), que puede usar información privilegiada porque es para el observador, no para el agente.

`tests/test_no_privileged_leak.py` verifica esta separación mediante análisis estático (AST): falla si `agent/` o `visualization/render.py` llegan a referenciar la etiqueta real, la etapa o la categoría de un estímulo.

## Estructura del proyecto

```
self_fly/
  config/         ExperimentConfig y todos los parámetros (recompensas, umbrales de estabilidad, conflicto, etc.)
  stimuli/        generación de estímulos REAL/FALSA, autorreferencial y de control
  environment/    tipos de estímulo y la frontera de aislamiento observe()
  agent/          interfaz de agente (interface.py) + factory; baseline/ (softmax lineal + REINFORCE, Fase A) y connectome/ (arquitectura recurrente reducida, Fase B)
  rewards/        tabla (etiqueta, acción, etapa) -> recompensa, incluido el esquema mantener/cambiar del conflicto progresivo
  experiment/     motor (engine.py), máquina de etapas (0-4), detector de estabilidad, StageOutcome, entropía, distancia entre políticas (snapshot y continua D_t), entrenamiento del agente auxiliar OTHER
  analysis/       agregación multi-seed/multi-condición, comparación descriptiva, informe automático (observación vs. interpretación)
  metrics/        acumuladores de métricas en vivo (recompensa, acciones, entropía)
  visualization/  render del estímulo (backend-agnóstico) + renderers/ intercambiables (Tk, matplotlib, 3D pendiente), GUI de tkinter, gráficos matplotlib headless
  io/             logging JSONL y gestión de runs/<id>/
run_experiment.py  # CLI headless (single-seed, multi-seed, multi-condición)
run_gui.py          # GUI de tkinter (mismo motor)
tests/              # ver `Tests` más abajo, incluido el de aislamiento AST
```

## Instalación

### Requisitos

- [Nix](https://nixos.org/download.html) con [flakes](https://nixos.wiki/wiki/Flakes) habilitados. Es la única dependencia: Nix se encarga de traer Python, numpy, matplotlib, pytest y tkinter en las versiones correctas, así que no hace falta gestionar un virtualenv ni instalar nada más a mano.

### Probarlo sin clonar

Con Nix instalado, se puede ejecutar directamente desde GitHub:

```bash
nix run github:luantorv/self-fly          # experimento headless
nix run github:luantorv/self-fly#gui      # interfaz gráfica (tkinter)
```

### Clonar y desarrollar

Para modificar el código, correr los tests o pasar flags propios:

```bash
git clone https://github.com/luantorv/self-fly.git
cd self-fly

# entorno de desarrollo (python + numpy + matplotlib + pytest + tkinter)
nix develop

# experimento headless (para simulaciones masivas / sin entorno gráfico)
nix run .#headless -- --seed 0 --n-trials 5000

# interfaz gráfica (tkinter), mismo motor que el modo headless
nix run .#gui -- --seed 0

# nix run . (sin sufijo) equivale a #headless
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
| `--seeds "0:9"` / `"0,2,5"` | (solo headless) corre esa lista/rango de semillas y devuelve un agregado descriptivo, no una sola corrida |
| `--agent baseline\|connectome` | Arquitectura del agente (default: `baseline`) |
| `--condition baseline\|control\|self\|self_multi\|all` | Condición experimental; `all` corre las cuatro condiciones sobre las mismas semillas y las compara descriptivamente (requiere `--seeds`) |
| `--run-name NOMBRE` | Sobrescribe el nombre de la corrida |
| `--n-trials N` | (solo headless) número de ensayos a ejecutar por corrida |
| `--runs-dir DIR` | Directorio base para `runs/` (default: `runs/`) |
| `--no-log` | (solo GUI) no persiste una carpeta de run |
| `--step-delay-ms MS` | (solo GUI) retardo entre pasos del motor |

Ejemplos:

```bash
nix run .#headless -- --agent connectome --condition self --seed 42
nix run .#headless -- --condition all --seeds 0:9
```

## Configuración

Todos los parámetros experimentales relevantes viven en `self_fly/config/schema.py` (recompensas, umbrales de estabilidad, pesos de features "señal", tasa de aprendizaje, mezcla de estímulos y overrides de recompensa por etapa) y se pueden serializar/cargar como JSON con `self_fly/config/loader.py`. Los valores por defecto (`self_fly/config/defaults.py`) están calibrados empíricamente contra el nivel de ruido de este proyecto — si cambias `StimulusConfig.noise_std` o `StabilityConfig.window_size`, revisa si los umbrales de estabilidad siguen teniendo sentido (está documentado en el docstring de `StabilityConfig`).

## Qué se guarda por corrida

Cada ejecución crea `runs/<run_id>/` con:

- `config.json` — configuración completa + semilla
- `trials.jsonl` — un registro por ensayo (estímulo, acción, probabilidades, recompensa, `policy_js_delta` = D_t, etc.)
- `policy_snapshots.jsonl` — `π₀`, `π₁`, `π₂`... con pesos y probabilidades sobre el conjunto de referencia (el nombre indica el desenlace real: `pi_2` solo si hubo convergencia genuina, `pi_2_unstable`/`pi_2_timeout` si no)
- `events.jsonl` — detección de estabilidad, cambios de etapa, `stage_outcome` (STABLE/UNSTABLE/TIMEOUT + entropía + distancia a inicio de etapa), fin del experimento
- `manifest.json` — id de corrida, tipo de agente, condición experimental, hash de la config, revisión de git, semilla, timestamps, estado
- `metrics_summary.json` y `plots/*.png` (probabilidades de acción, recompensa y entropía en el tiempo)
- `report.md` — informe automático con secciones separadas de Observaciones (solo números) e Interpretación (disclaimers fijos, sin lenguaje antropomórfico)

Con la misma semilla, dos corridas producen `trials.jsonl` idénticos (salvo el timestamp de cada línea, que refleja el reloj real).

## Tests

```bash
nix develop --command python -m pytest tests/ -q
```

89 tests cubren, además de todo lo de Fase A (generación de estímulos sin atajo trivial, la frontera de aislamiento, convergencia de la política, el modelo de recompensa, el detector de estabilidad, la distancia entre políticas, la transición completa `π₀ → π₁ → π₂`, determinismo del logging, los gráficos headless, un smoke test de la GUI y el aislamiento estructural agente/experimento): baseline multi-seed, estados STABLE/UNSTABLE/TIMEOUT (con guardia mecánica de que nunca se fuerza una convergencia falsa), entropía y distancia temporal `D_t`, el grupo de control, SELF_A/SELF_B/OTHER (incluido el entrenamiento del agente auxiliar), el conflicto progresivo, la comparación entre condiciones y entre agentes (con guardia mecánica de que nunca se produce un ranking), la equivalencia e intercambiabilidad de `BaselineAgent`/`ConnectomeInspiredAgent`, los renderers intercambiables y el lenguaje del informe automático.

## Qué es hipótesis experimental y qué es decisión de implementación

Documentado directamente en el código donde aplica (docstrings de `StabilityConfig`, `SelfStimulusGenerator`, `policy_distance.py`), pero en resumen:

- **Hipótesis experimentales** (afectan la interpretación científica, no solo el código): cómo se deriva el estímulo autorreferencial (normas de las columnas de pesos de la política), los umbrales numéricos de estabilidad, el uso de distancia Jensen-Shannon como métrica primaria entre políticas.
- **Decisiones de implementación** (cualquier alternativa razonable serviría): política lineal + REINFORCE en vez de Q-learning tabular, formato JSON/JSONL para config y logs, tkinter para la GUI mínima.

## Estado del proyecto

Fase A y Fase B completas. Pendiente: visualización 3D (el punto de extensión ya existe, sin implementación). El diagnóstico de por qué la política frecuentemente no reconvergía tras la introducción de `SELF` (baseline de REINFORCE compartido entre regímenes de recompensa muy distintos + criterio de estabilidad que no aísla la categoría autorreferencial) está documentado en los docstrings de `self_fly/agent/baseline/learner.py`, `self_fly/experiment/stability.py` y `self_fly/experiment/stage_outcome.py`, y es la motivación directa de los estados `STABLE`/`UNSTABLE`/`TIMEOUT` y de las métricas de entropía/`D_t` de Fase B.
