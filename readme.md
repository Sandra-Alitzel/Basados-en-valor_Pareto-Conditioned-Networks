

# Pareto Conditioned Networks (PCN) - Control Continuo

## 1. Estructura del Proyecto

El sistema está organizado en módulos que separan la lógica de datos, la arquitectura neuronal y las herramientas de evaluación:

### Núcleo (src/pcn/)
* *core.py: Implementa el ExperienceBuffer (memoria de episodios completos), el cálculo de *returns-to-go y los algoritmos de ordenamiento no dominado para extraer el frente de Pareto.
* *networks.py: Contiene la clase ConditionedActor, una red neuronal gaussiana que recibe como entrada el estado y un comando de retorno objetivo (*target return).
* *wrappers.py*: Adaptadores para entornos MuJoCo (HalfCheetah-v5) que transforman la recompensa escalar en vectores multiobjetivo y aplican normalización.

### Evaluación y Benchmarking (src/evaluation/)
* *hypervolume.py*: Algoritmo para el cálculo del hipervolumen en 2D (barrido) y N-dimensiones.
* *compare_baselines.py*: Script principal de experimentación que ejecuta PCN y PGMORL bajo las mismas condiciones para realizar la comparativa estadística.

---

## 2. Instrucciones de Ejecución

### Instalación de Dependencias
Asegúrese de contar con Python 3.9+ y los siguientes paquetes:
bash
pip install torch numpy gymnasium mujoco scipy matplotlib mo-gymnasium morl-baselines


### Entrenamiento y Evaluación de PCN
Para ejecutar un entrenamiento estándar de PCN con los parámetros por defecto:
bash
python main.py --env-id HalfCheetah-v5 --total-iterations 500


### Ejecución del Benchmark Comparativo
Para realizar la comparación estadística contra el baseline (PGMORL) utilizando múltiples semillas aleatorias:
bash
python -m src.evaluation.compare_baselines --n-seeds 10 --base-seed 1000


### Resultados
Al finalizar el benchmark, los resultados se almacenarán en la carpeta results/, incluyendo:
* boxplot_hv.png: Comparativa de hipervolumen.
* pareto_fronts.png: Visualización de los frentes de Pareto alcanzados por ambos algoritmos.
* stats_report.txt: Resultados del test de Wilcoxon y medias de desempeño.
"""


<details>
  <summary> <b>Análisis PDF</b></summary>
  <br>
  <p align="center">
    <embed src="./PCN.pdf" type="/pdf" width="100%" height="600px" />
  </p>
  <p align="center">
    <i>Si no puedes ver el archivo, <a href="./PCN.pdf">haz clic aquí para descargarlo.</a></i>
  </p>
</details>