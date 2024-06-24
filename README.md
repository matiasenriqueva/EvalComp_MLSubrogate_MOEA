# Evaluación comparativa de modelos subrogantes en el algoritmo NSGA-II guiado por datos

Repositorio con el código fuente para la **"Evaluación comparativa de modelos subrogantes en el algoritmo NSGA-II guiado por datos"**, proyecto de tesis de pregrado del autor.

## Aspectos generales

El proyecto esta almacenado en la carpeta [Code](code/)

Dentro de dicha carpeta se encuentran los *pipelines* generados, con un total de 5.

* [**01_pipeline_NSGAII:**](code/01_pipeline_NSGAII.ipynb) Tiene el *pipeline* de trabajo que configura y llama los diferentes experimentos realizados.
* [**02_quality_indicator_summary_generator:**](code/02_quality_indicator_summary_generator.ipynb) Genera los archivos resumen para tiempo, épsilon e hipervolumen basado en los datos generados en los experimentos.
* [**03_plot_generator:**](code/03_plot_generator.ipynb) Funciones de automatización para generar todos los gráficos de frentes de Pareto y gráficos de pérdida de diversidad.
* [**04_statical_test_summary_generator:**](code/04_statical_test_summary_generator.ipynb) Genera los archivos y tablas usados en los test estadísticos no paramétricos. Test de Wilcoxon y test de Friedman.
* [**read_wilcoxon:**](code/read_wilcoxon.ipynb) Automatiza el proceso de creación de tablas en latex para tener un test de Wilcoxon de manera consolidada en una sola tabla, en vez de una tabla por problema.

También podemos encontrar la clase con el algoritmo intervenido.

* [**modnsgaii.py:**](code/modnsgaii.py) Tiene la clase con el algoritmo intervenido con modelos subrogantes y el algoritmos original intervenido para medir perdida de diversidad.

Dentro de los modelos subrogantes, podemos encontrarlos en [surrogate_models](code/surrogate_models/) se encuentran los siguientes.

* [iSOUPTreeRegressor](code/surrogate_models/iSOUPTreeRegressor_surrogate.py)
* [RegressorChain](code/surrogate_models/regressor_chain_surrogate.py)
* [MultiOutputLearnerMLP](code/surrogate_models/multioutput_learning_surrogate.py)
* [LSTM](code/surrogate_models/LSTM_surrogate.py)

Tambien se encuentra las clases modificadas en JMetalPy, para la correcta ejecución de los códigos (deben ser reemplazados).

## Requisitos

Para la ejecución sin inconvenientes y sin problema de instalación de dependencias se **debe utilizar Python 3.8.x**

Los requisitos de dependencias se encuentran en el archivo [requirements.txt](/requeriments.txt) que puede ser instalado en un entorno virtual siguiendo los siguientes pasos en windows.

- Abrimos un terminal y creamos un entonro virtual para luego activarlo.
```bash
python.exe -m venv venv
venv/Scripts/activate
```

- Instalamos las dependencias del archivo **requirements.txt**

```bash
pip install -r requirements.txt
```

Una vez instaladas las dependencias podremos ejecutar los pipelines sin problemas utilizando jupyter o VSCode.

## Ejecución

 En nuestro IDE preferido, seleccionar el entorno recien creado como interprete de código y lanzamos los archivos.