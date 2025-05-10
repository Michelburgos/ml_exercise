# Análisis y Entrenamiento de Modelos para el Conjunto de Datos de MercadoLibre

Este README proporciona una visión general completa del análisis exploratorio de datos (EDA) y el entrenamiento de modelos realizados sobre el conjunto de datos de MercadoLibre (`MLA_100k.jsonlines`), documentados en los cuadernos Jupyter `001-EDA.ipynb` y `002-model_training.ipynb`. El conjunto de datos contiene 100,000 listados de productos de MercadoLibre Argentina, con el objetivo de analizar los datos y construir modelos de aprendizaje automático para predecir la `condición` de los artículos (nuevo o usado).

## Visión General del Conjunto de Datos
El conjunto de datos original (`MLA_100k.jsonlines`) consta de 100,000 listados de productos con 48 características, incluyendo precio, información del vendedor, detalles de envío y condición del producto. La variable objetivo, `condición`, es binaria (`nuevo` o `usado`). El proceso de EDA limpió y preprocesó el conjunto de datos, resultando en un conjunto refinado (`clean_dataset.csv`) con 97,181 filas y 50 características, listo para modelado.

## Análisis Exploratorio de Datos (EDA)

### Hallazgos Clave del EDA
1. **Variable Objetivo (`condición`)**:
   - **Distribución**: Balanceada con 53.76% `nuevo` y 46.24% `usado`, adecuada para clasificación sin preocupaciones significativas por desequilibrio de clases.
2. **Valores Faltantes**:
   - **Alta Tasa de Faltantes (>90%)**: Columnas como `differential_pricing`, `subtitle`, `catalog_product_id`, `original_price`, `official_store_id`, `seller_contact` y `video_id` fueron eliminadas por su baja informatividad.
   - **Tasa Moderada de Faltantes**: `warranty` (60.9% faltante) y `parent_item_id` (23% faltante) fueron eliminadas por su complejidad o irrelevancia.
3. **Análisis de Características**:
   - **Características Numéricas**: `price`, `initial_quantity`, `sold_quantity`, `available_quantity` mostraron rangos diversos, con la mayoría de los listados teniendo cantidades bajas.
   - **Características Categóricas**: `listing_type_id`, `buying_mode`, `status`, `seller_state` fueron codificadas con one-hot encoding.
   - **Características Booleanas**: `accepts_mercadopago`, `automatic_relist`, `free_shipping`, `local_pick_up` fueron convertidas a binario (0/1). `accepts_mercadopago` fue eliminada por ser constante.
   - **Características Derivadas**:
     - `pixels` y `max_pixels`: Calculadas a partir de las resoluciones de imágenes para cuantificar calidad.
     - `num_payment_methods`: Conteo de métodos de pago no MercadoPago.
     - `title_length`: Longitud del título del producto.
     - `num_pictures`: Número de imágenes en el listado.
     - `seller_state`: Extraída de `seller_address` para información geográfica.
4. **Transformaciones de Datos**:
   - **Columnas Eliminadas**: 37 columnas (e.g., `id`, `permalink`, `thumbnail`, `warranty`) fueron removidas por alta tasa de faltantes o irrelevancia.
   - **Duplicados**: Eliminados 549 duplicados, reduciendo el conjunto a 97,985 entradas.
   - **Codificación**: Aplicado one-hot encoding a variables categóricas, conversiones de booleanos a enteros, y codificación de `condición` (`usado`=0, `nuevo`=1).
   - **Faltantes en Pixels**: Eliminadas filas con `pixels` o `max_pixels` faltantes, resultando en 97,231 filas.
   - **Salida**: Guardado del conjunto limpio como `data/clean_dataset.csv` con 50 columnas.
5. **Observaciones**:
   - La mayoría de los vendedores son de Buenos Aires y Capital Federal, reflejando la base de usuarios de MercadoLibre.
   - Características como `price`, `free_shipping`, `local_pick_up` y `seller_state` son probablemente predictivas de `condición`.
   - La calidad de imagen (`pixels`, `max_pixels`) y detalles del listado (`num_pictures`, `title_length`) pueden correlacionarse con listados profesionales (a menudo nuevos).

### Conclusiones del EDA
- El conjunto de datos fue limpiado exitosamente, eliminando características irrelevantes o ruidosas y generando características predictivas como `title_length` y `num_pictures`. La distribución balanceada de `condición` soporta una clasificación robusta.

## Entrenamiento de Modelos

### Modelos Utilizados
Se entrenaron cuatro modelos de aprendizaje automático para predecir `condición`:
1. **Regresión Logística**: Modelo lineal base por su simplicidad e interpretabilidad.
2. **Clasificador Random Forest**: Método de ensamblaje de árboles de decisión para capturar relaciones no lineales.
3. **Clasificador Gradient Boosting**: Ensamblaje secuencial de árboles para corregir errores previos.
4. **Clasificador XGBoost**: Algoritmo optimizado de gradient boosting para alto rendimiento.

### Metodología
1. **Carga de Datos**: Carga de `clean_dataset.csv` en un DataFrame de Pandas.
2. **Separación de Características y Objetivo**:
   - Características (`X`): Todas las columnas excepto `condición`.
   - Objetivo (`y`): `condición` (0 para usado, 1 para nuevo).
3. **Preprocesamiento**:
   - Estandarización de columnas numéricas (e.g., `price`, `initial_quantity`, `pixels`) usando `StandardScaler`.
   - Uso de `ColumnTransformer` para preservar columnas codificadas y booleanas.
4. **División Entrenamiento-Prueba**:
   - División en 80% entrenamiento y 20% prueba con estratificación para mantener el balance de clases.
   - Semilla aleatoria establecida en 42 para reproducibilidad.
5. **Entrenamiento de Modelos**:
   - Entrenados con hiperparámetros predeterminados (excepto `LogisticRegression`: `max_iter=1000`; `XGBoost`: `use_label_encoder=False`, `eval_metric='logloss'`).
6. **Evaluación**:
   - Evaluación en el conjunto de prueba usando precisión, recall, F1-Score y exactitud.
7. **Importancia de Características**:
   - Análisis de importancia de características para el mejor modelo (XGBoost).

### Métricas Usadas para la Evaluación
- **Exactitud (Accuracy)**: Proporción de predicciones correctas: `(TP + TN) / (TP + TN + FP + FN)`.
- **Precisión (Precision)**: Proporción de verdaderos positivos entre predicciones positivas: `TP / (TP + FP)`.
- **Recall (Sensibilidad)**: Proporción de verdaderos positivos entre instancias positivas reales: `TP / (TP + FN)`.
- **F1-Score**: Media armónica de precisión y recall: `2 * (Precisión * Recall) / (Precisión + Recall)`.

Estas métricas fueron elegidas para:
- Evaluar el rendimiento general (Exactitud).
- Minimizar falsos positivos (Precisión).
- Asegurar cobertura de artículos nuevos (Recall).
- Balancear precisión y recall (F1-Score), especialmente para un conjunto balanceado.

### Resultados del Rendimiento del Modelo
| Modelo              | Exactitud | Precisión | Recall | F1-Score |
|---------------------|-----------|-----------|--------|----------|
| XGBoost             | 0.8586    | 0.8971    | 0.8339 | **0.8643** |
| Random Forest       | 0.8527    | 0.8738    | 0.8500 | 0.8617   |
| Gradient Boosting   | 0.8466    | 0.8784    | 0.8311 | 0.8541   |
| Regresión Logística | 0.7770    | 0.7776    | 0.8222 | 0.7993   |

#### Observaciones Clave:
- **XGBoost** superó a los demás con un F1-Score de 0.8643, mostrando alta exactitud (85.86%) y precisión (0.8971).
- **Random Forest** y **Gradient Boosting** se desempeñaron cerca de XGBoost, con F1-Scores de 0.8617 y 0.8541.
- **Regresión Logística** tuvo el menor rendimiento (F1-Score: 0.7993), probablemente debido a sus supuestos lineales.
- La alta precisión de XGBoost minimiza falsos positivos, y su recall asegura buena cobertura de artículos nuevos.

#### Razón para Escoger XGBoost
XGBoost fue seleccionado como el modelo preferido debido a su rendimiento superior en todas las métricas evaluadas, con un F1-Score de 0.8643, exactitud de 85.86%, precisión de 0.8971 y recall de 0.8339, superando a los otros modelos. Este modelo destaca por su capacidad para manejar datos estructurados como los de MercadoLibre, que incluyen una mezcla de características numéricas, categóricas y derivadas. 

### Características Más Relevantes
Importancia de características desde XGBoost, basada en la gráfica actualizada:
| Característica                     | Importancia |
|------------------------------------|-------------|
| `initial_quantity`                 | 0.369       |
| `listing_listing_type_id_free`     | 0.312       |
| `buying_buying_mode_buy_it_now`    | 0.038       |
| `sold_quantity`                    | 0.033       |
| `free_shipping`                    | 0.027       |
| `state_seller_state_Chubut`        | 0.018       |
| `price`                            | 0.017       |
| `listing_listing_type_id_gold_special` | 0.013  |
| `max_pixels`                       | 0.013       |
| `buying_buying_mode_auction`       | 0.012       |

#### Perspectivas:
- **initial_quantity (0.369)**: Cantidades más altas probablemente indican artículos nuevos de vendedores profesionales.
- **listing_listing_type_id_free (0.312)**: Listados gratuitos suelen asociarse con artículos usados por vendedores individuales.
- **buying_buying_mode_buy_it_now (0.038)**: Opciones de compra inmediata pueden correlacionarse con artículos nuevos.
- **sold_quantity (0.033)** y **free_shipping (0.027)**: Volumen de ventas y envío gratuito pueden reflejar el tipo de vendedor o condición del artículo.
- **state_seller_state_Chubut (0.018)**: La ubicación geográfica (Chubut) tiene influencia moderada, posiblemente por patrones regionales de venta.
- **price (0.017)**: Los artículos nuevos suelen tener precios más altos.
- **listing_listing_type_id_gold_special (0.013)** y **max_pixels (0.013)**: Listados premium y calidad de imagen pueden indicar artículos profesionales (nuevos).
- **buying_buying_mode_auction (0.012)**: El modo subasta podría estar vinculado a artículos usados.

## Conclusiones
- **EDA**: El conjunto de datos fue limpiado exitosamente, eliminando características irrelevantes o ruidosas y generando características predictivas como `title_length` y `num_pictures`. La distribución balanceada de `condición` soporta una clasificación robusta.
- **Rendimiento del Modelo**: XGBoost logró el mejor rendimiento (F1-Score: 0.8643), lo que lo hace ideal para clasificar condiciones de artículos debido a su capacidad para manejar relaciones complejas.
- **Importancia de Características**: `initial_quantity` y `listing_listing_type_id_free` son los principales impulsores, destacando la importancia de las características del listado. Otras como `buying_buying_mode_buy_it_now`, `sold_quantity` y `free_shipping` también contribuyen significativamente.
- **Implicaciones Prácticas**: El modelo XGBoost puede automatizar la clasificación de condiciones en MercadoLibre, mejorando la eficiencia de la plataforma. Los vendedores pueden optimizar listados (e.g., cantidad, opciones de envío) para aclarar la condición del artículo.
- **Limitaciones**:
  - Los hiperparámetros predeterminados pueden no ser óptimos.
  - Características eliminadas como `warranty` y texto de `title` podrían proporcionar más información con procesamiento avanzado (e.g., NLP).
  - El modelo es específico de MercadoLibre Argentina y puede requerir retrainamiento para otros contextos.
- **Adecuación de Métricas**: La combinación de exactitud, precisión, recall y F1-Score proporcionó una evaluación integral, con F1-Score balanceando efectivamente precisión y recall.

## Próximos Pasos
1. **Ajuste de Hiperparámetros**: Optimizar XGBoost, Random Forest y Gradient Boosting usando búsqueda en cuadrícula o aleatoria.
2. **Ingeniería de Características**: Incorporar análisis de texto para `title` o `warranty` y explorar interacciones entre características.
3. **Validación Cruzada**: Usar validación cruzada k-fold para estimaciones robustas de rendimiento.
4. **Selección de Características**: Eliminar características de baja importancia para reducir complejidad.
5. **Despliegue del Modelo**: Integrar el modelo XGBoost en la plataforma de MercadoLibre para clasificación en tiempo real.

## Cómo Usar
1. **Requisitos**: Instalar dependencias (`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`).
2. **Ejecutar EDA**:
   - Ejecutar `001-EDA.ipynb` para preprocesar el conjunto de datos y generar `clean_dataset.csv`.
3. **Ejecutar Entrenamiento de Modelos**:
   - Ejecutar `002-model_training.ipynb` para entrenar modelos, evaluar rendimiento y analizar importancia de características.
4. **Datos de Entrada**:
   - Asegurarse de que `MLA_100k.jsonlines` esté disponible para EDA.
   - Usar `clean_dataset.csv` para el entrenamiento del modelo.
5. **Salidas**:
   - **EDA**: `clean_dataset.csv` y visualizaciones de distribuciones de datos.
   - **Entrenamiento de Modelos**: Tabla de métricas de rendimiento y una gráfica de barras de las 10 características más importantes para XGBoost.
