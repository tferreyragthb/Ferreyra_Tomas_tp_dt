Trabajo Práctico – Decision Transformer para Sistemas de Recomendación
Autor: Tomas Ferreyra – 2025

------------------------------------------------------------
DESCRIPCIÓN GENERAL
------------------------------------------------------------
Este proyecto implementa un Decision Transformer (DT) aplicado a un sistema 
de recomendación, siguiendo la estructura y consignas de la cátedra.

Incluye:
1) Exploración del dataset
2) Entrenamiento del Decision Transformer
3) Evaluación del modelo y comparación con un baseline
4) Return Conditioning (efecto del return objetivo en recomendaciones)

El dataset provisto por la cátedra consta de:
- netflix8_train.df
- netflix8_test.json
- mu_netflix.csv (groups)

------------------------------------------------------------
EJECUCIÓN
------------------------------------------------------------
1. Instalar dependencias:
   pip install -r requirements.txt

2. Ejecutar notebooks en este orden:
   - 01_exploracion_dataset.ipynb
   - 02_training.ipynb
   - 03_evaluation.ipynb
   - 04_return_conditioning.ipynb

El modelo entrenado se carga automáticamente desde:
results/checkpoints/dt_model.pth

------------------------------------------------------------
LOGS DE ENTRENAMIENTO
------------------------------------------------------------
Se almacenan en:
results/logs/training_log.txt

Generados automáticamente desde 02_training.ipynb

------------------------------------------------------------
LICENCIA / USO
------------------------------------------------------------
Uso estrictamente académico.
