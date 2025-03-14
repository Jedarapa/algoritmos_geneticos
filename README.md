# Generación de Datos de Prueba Estructurales con Algoritmos Genéticos

Este repositorio contiene dos enfoques distintos para la **generación de datos de prueba** en el contexto de pruebas estructurales de software. Ambos se basan en **algoritmos genéticos (AG)**, pero difieren en la forma de representar los datos y medir la cobertura o activación de ramas.

## 1. Estructura del Repositorio

algoritmos_geneticos/ ├── primera_opcion/ │ ├── src/ │ │ ├── binary_encoding.py │ │ ├── code_analyzer.py │ │ ├── config.py │ │ ├── fitness_calculator.py │ │ ├── genetic_algorithm.py │ │ ├── string_test_data_generator.py │ │ └── test_data_generator.py │ ├── test/ │ │ └── test_programs/ │ │ └── triangle.py │ ├── main.py │ └── results/ │ └── ... (gráficas, logs, CSV) ├── segunda_opcion/ │ └── ag.py └── README.md (este archivo)


### `primera_opcion/`

- **main.py**: Archivo principal para ejecutar en modo `triangle` o `string`.
- **src/**: Contiene archivos esenciales del GA, la instrumentación de cobertura y la lógica de fitness.
- **test/test_programs/triangle.py**: Programa bajo prueba para la clasificación de triángulos.
- **results/**: Carpeta donde se guardan logs, CSV y gráficas.

### `segunda_opcion/`

- **ag.py**: Implementa un GA genérico que:
  - Inicializa individuos según un template.
  - Evalúa funciones objetivo (real o complex).
  - Mide la activación de ramas sin usar cobertura.

## 2. Requisitos e Instalación

### 2.1 Requisitos Generales

- **Python 3.7+**
- **pip** para instalar librerías.

### 2.2 Dependencias Principales

Para la **primera opción** (primera_opcion/):

- coverage
- matplotlib
- seaborn
- numpy
- pandas

Para la **segunda opción** (segunda_opcion/):

- matplotlib
- numpy

### 2.3 Instalación

1. Clonar el repositorio:

    ```bash
    git clone https://github.com/usuario/algoritmos_geneticos.git
    cd algoritmos_geneticos
    ```

2. Instalar dependencias:

    Para la **primera opción**:

    ```bash
    pip install coverage matplotlib seaborn numpy pandas
    ```

    Si únicamente vas a usar la **segunda opción**, con `ag.py`, basta con:

    ```bash
    pip install matplotlib numpy
    ```

## 3. Primera Opción: Triángulos y Correos

En la carpeta `primera_opcion/` encontrarás:

- **main.py**: Permite ejecutar el modo `triangle` o `string`.
- **src/**: Varios archivos que implementan la lógica del GA, el análisis de código (`code_analyzer.py`), la codificación binaria (`binary_encoding.py`), etc.
- **test/test_programs/triangle.py**: Función `classify_triangle(a, b, c)` que clasifica triángulos.

### 3.1 Ejecución

1. **Entrar a `primera_opcion`**:

    ```bash
    cd primera_opcion
    ```

2. **Modo triángulo**:

    ```bash
    python main.py triangle
    ```

   - Instrumenta `triangle.py` con cobertura.
   - Genera individuos `[a,b,c]` y evalúa su fitness con base en la cobertura y la clasificación resultante.
   - Muestra la evolución en consola y genera gráficas en `results/`.

3. **Modo string**:

    ```bash
    python main.py string
    ```

   - Genera individuos de tipo cadena simulando correos electrónicos.
   - Muestra la evolución del fitness (qué tan “válido” es el correo) y al final, el mejor correo.

### 3.2 Resultados y Gráficas

En la carpeta `results/` se generan los siguientes archivos:

- **Logs**: `genetic_testing.log`
- **CSV de evolución**: `genetic_results_*.csv`
- **Gráficas**: `fitness_evolution_*.png`, `generation_stats_*.png`, etc.

### 3.3 Adaptación

Para probar otro archivo en vez de `triangle.py`, modifica `test_data_generator.py` y/o `fitness_calculator.py` para instrumentarlo y calcular el fitness sobre tu propio programa bajo prueba.

## 4. Segunda Opción: ag.py (Plantilla Genérica)

En la carpeta `segunda_opcion/` se encuentra:

- **ag.py**: Archivo único que:
  - Define una plantilla (`template`) con parámetros como `int`, `float`, `choice`, `string`, `nested`, etc.
  - Ofrece dos funciones objetivo: `target_function_real(...)` y `target_function_complex(...)`.
  - Implementa la clase `Chromosome` (con métodos de mutación, cruce, reparación, etc.).
  - Realiza un GA con selección por torneo, cruce, mutación dinámica y medición de la cobertura de ramas (internamente).

### 4.1 Ejecución

1. **Entrar a `segunda_opcion`**:

    ```bash
    cd ../segunda_opcion
    ```

2. **Ejecutar `ag.py`**:

    ```bash
    python ag.py
    ```

   - El script solicitará:
     - Función objetivo (`real` o `complex`).
     - Parámetros: tamaño de población, generaciones, tasa de cruce y mutación.

   - Cada ~10 generaciones, se mostrará información sobre la evolución:

     ```
     Generación X: Mejor fitness = 0.XXXX, Promedio = 0.XXXX, Diversidad = 0.XXXX, MutRate = 0.XXX
     ```

   - Al terminar, se imprimirá el mejor individuo (sus genes), las ramas cubiertas y la cobertura (porcentaje de ramas activadas).
   - Se abrirá una gráfica con la evolución del mejor fitness, fitness promedio y la diversidad global.

### 4.2 Adaptación

- Para cambiar la estructura de los datos de prueba, edita la variable `template` al inicio de `ag.py`.
- Si quieres nuevas funciones objetivo, crea otra función similar a `target_function_complex` y asígnala a `TARGET_FUNCTION`.
- Ajusta los parámetros del GA (p. ej., `pop_size`, `generations`) en la llamada a `genetic_algorithm(...)`.

## 5. Recomendaciones y Notas

### Cobertura vs. Ramas:

- En la **primera opción** se usa `coverage` para medir líneas y ramas ejecutadas en un archivo Python.
- En la **segunda opción**, la cobertura se mide manualmente a través de condiciones (`branches.add(...)`).

### Diversidad:

Ambos enfoques implementan mecanismos para evitar la convergencia prematura, ya sea calculando distancias binarias (primer enfoque) o distancias en la plantilla (segundo enfoque).

### Logs y Depuración:

- La **primera opción** genera `genetic_testing.log` en `results/`.
- La **segunda opción** imprime en consola cada ~10 generaciones. Puedes redirigir la salida a un archivo si deseas.

## 6. Posibles Extensiones

- **Cambiar la Función Bajo Prueba**: En la primera opción, reemplaza `triangle.py` con otro archivo, ajustando `test_data_generator.py` para instrumentarlo.
- **Más Tipos de Genes**: En la segunda opción, añade más tipos (p. ej., booleanos) en `template` y define su mutación.
- **Algoritmos de Selección Alternativos**: Tanto en `genetic_algorithm.py` como en `ag.py`, podrías cambiar la selección por torneo a ruleta, rank-based, etc.
- **Parámetros Evolutivos**: Ajustar población, generaciones, crossover, mutación, etc., para ver mejoras en cobertura o convergencia.

## 7. Licencia y Créditos

Este repositorio forma parte de un trabajo de grado sobre optimización de pruebas estructurales con algoritmos genéticos.

## 8. Contacto

Para dudas o sugerencias:

- **Autor**: Jesus David Ramirez Pareja
- **Email**: jedramirezpa@unal.edu.co

¡Gracias por tu interés en este proyecto!
