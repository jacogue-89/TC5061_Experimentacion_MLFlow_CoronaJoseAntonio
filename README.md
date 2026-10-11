# TC5061_Experimentacion_MLFlow_JoséAntonioCorona
# Clasificación de riesgo de diabetes y tracking con MLflow

Este proyecto implementa un flujo de MLOps para entrenar, evaluar y registrar modelos de clasificación de riesgo de diabetes, usando el dataset BRFSS2015 y Scikit-Learn y MLflow

El proyecto incluye preprocesamiento de datos (truncado de outliers y escalamiento estándar), parametrización por línea de comandos con `argparse`, y fijación de semillas aleatorias para garantizar reproducibilidad. También incluye el registro automatizado de experimentos, métricas, artefactos (matriz de confusión) y modelos.

---

# Estructura del Proyecto

"""text
.
├── train.py                                       # Script ejecutable principal para entrenamiento y registro
├── requirements.txt                               # Lista de dependencias del entorno Python
├── diabetes_binary_health_indicators_BRFSS2015.csv # Dataset de entrada
├── README.md                                      # Documentación del proyecto
└── mlruns/                                        # Carpeta generada automáticamente por MLflow
---

# Requisitos previos
1. Clonar o descargar el repositorio
Asegúrarse de estar en el directorio raíz del proyecto.

2. Crear y activar un entorno virtual (altamente recomendado)
En Linux/macOS:

python3 -m venv env
source env/bin/activate

En Windows (PowerShell):

python -m venv env
.\env\Scripts\Activate.ps1

3. Instalar dependencias
Instalar todas las librerías necesarias ejecutando:

pip install -r requirements.txt

# Instrucciones para la ejecución
El script train.py permite parametrizar hiperparámetros por línea de comandos. Si no se especifican argumentos, se ejecutará con los valores pro default

1. Ejecución Básica (valores default)

python train.py

2. Ejecución con otros hiper parámetros
Los parámetros que se pueden modificar son la fuerza de regularización (--C), el algoritmo de optimización (--solver), el balanceo de clases (--class_weight) y la semilla (--seed):

python train.py --C 0.1 --solver saga --class_weight balanced --seed 42

3. Ejemplo de 5 Corridas para Comparación en MLflow
Los ejemplos que se ejecutaron patra poner a prueba el script son estos:

Corrida 1: Configuración base
python train.py

Corrida 2: Sin balanceo de clases
python train.py --C 1.0 --solver lbfgs --class_weight None

Corrida 3: Alta regularización (C bajo)
python train.py --C 0.01 --solver lbfgs --class_weight balanced

Corrida 4: Baja regularización con solver liblinear
python train.py --C 10.0 --solver liblinear --class_weight balanced

Corrida 5: Regularización moderada con solver saga
python train.py --C 0.1 --solver saga --class_weight balanced

Nota: Para este ejercicio, la corrida 2 se ejecutó 2 veces para verificar repetibilidad del modelo bajo las mismas condiciones

# Visualización de resultados en MLflow
Para comparar las métricas, parámetros y matrices de confusión registradas:
1. Abrir la terminal e iniciar la interfaz gráfica de MLflow:

mlflow ui

2. Abrir el navegador e ingresar esta IP:
http://127.0.0.1:5000

3.Seleccionar el experimento "Clasificacion_Diabetes_MLOps"
4.Seleccionar las ejecuciones que se deseen evaluar y utilizar la herramienta "Compare" para mostrar en paralelo métricas como Recall, Precision y Accuracy.

# Parámetros disponibles en el scritp train.py
Argumento: --datapath
Tipo: str
Valor default: diabetes_binary_health_indicators_BRFSS2015
Descripción: ruta al archivo CSV con los datos.

Argumento: --experiment_name
Tipo: str
Valor default: Clasificacion_Diabetes_MLOps
Descripción: nombre del experimento en MLflow.

Argumento: --C
Tipo: float
Valor default: 1.0
Descripción: inverso de la fuerza de regularización.

Argumento: --max_iter
Tipo: int
Valor default: 1000
Descripción: número máximo de iteraciones.

Argumento: --solver
Tipo: str
Valor default: lbfgs
Descripción: algoritmo de optimización (lbfgs, liblinear, saga).

Argumento: --class_weight
Tipo: str
Valor default: balanced
Descripción: peso de clases (balanced, None).

Argumento: --seed
Tipo: int
Valor default: 42
Descripción: semilla aleatoria para reproducibilidad.

Argumento: --test_size
Tipo: float
Valor default: 0.20
Descripción: proporción del conjunto de prueba (80/20).