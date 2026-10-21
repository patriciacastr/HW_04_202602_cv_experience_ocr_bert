# Predicción de años de experiencia a partir de imágenes de CV con PaddleOCR y BERT

**Curso:** Machine Learning / NLP / Applied AI · **Tarea:** HW_04_202602
**Autoras:** Patricia Castro Hilario, Carla Bocanegra Valentin

Pipeline de IA documental de punta a punta: lee la imagen de un CV con PaddleOCR y predice los años de experiencia laboral del candidato (de 0 a 20). Se comparan cuatro modelos sobre la misma partición: una predicción por la mediana, TF-IDF + Ridge, embeddings de oraciones congelados + Ridge y un DistilBERT ajustado como regresor. Se agrega una regla que suma las fechas de los empleos como referencia honesta.

## Resultados principales

| Modelo | MAE (años) | Acc ±2 | R² |
|---|---|---|---|
| Mediana | 3.42 | 0.365 | -0.03 |
| TF-IDF + Ridge | 0.99 | 0.893 | 0.90 |
| Embeddings + Ridge | 1.26 | 0.823 | 0.85 |
| **BERT ajustado** | **0.20** | **1.000** | **1.00** |
| *Regla: suma de fechas de empleos (bonus)* | *0.02* | – | – |

BERT reduce el MAE entre 80% y 94% frente a los baselines aprendidos, y la ganancia es estadísticamente significativa. Pero una regla simple que suma las duraciones de los empleos es diez veces más precisa que BERT en este formato sintético, y la prueba de +5 años en las fechas muestra que BERT depende en parte de los años absolutos en lugar de calcular duraciones. Ambos hallazgos se discuten abajo.

## 1. ¿Qué hace el proyecto y para quién?

Los años de experiencia casi nunca están escritos en un CV. Un reclutador los deduce conectando las fechas de cada empleo, el número de cargos y el año en que terminó la educación. Este proyecto aprende a hacerlo a partir del texto, empezando desde la *imagen* del CV.

Los usuarios potenciales son agencias de reclutamiento, bolsas de empleo, servicios públicos de empleo y equipos de RR. HH. que revisan miles de CVs. El uso sensato es pre-llenar un campo del perfil que una persona confirma. Nunca debe decidir quién recibe una entrevista (ver la pregunta 4).

## 2. Muestreo

- Dataset: PopResume (Yu, Park & Moon, licencia CC BY 4.0), CVs sintéticos de 5 ocupaciones. Solo se usaron imágenes **sin foto** (`has_photo == False`).
- **800 CVs por ocupación × 5 ocupaciones = 4,000 CVs**, semilla `42`, leídos en streaming una ocupación a la vez con un buffer de mezcla (`--buffer-size 500`). Los IDs muestreados están en `data/raw/sampled_ids.txt`.
- Variable objetivo: `exp_year`, recortada a 0–20.
- **Revisión del tope:** solo 25 de 4,000 CVs (0.62%) están en 20, muy por debajo del límite de 15%, así que el tope se mantiene en 20 para todos los modelos.

Distribución de `exp_year` (4,000 CVs): media 6.49, mediana 5.70, desviación estándar 4.36, rango de 0.01 a 20.

| Ocupación | Media | Mediana |
|---|---|---|
| Contadores y auditores | 7.32 | 6.57 |
| Obreros de construcción | 5.87 | 4.69 |
| Docentes de primaria y secundaria | 6.65 | 5.97 |
| Enfermeros/as registrados | 6.43 | 5.81 |
| Desarrolladores de software | 6.20 | 5.42 |

Peso de cada banda: 0–2 años 18.7%, 3–5 años 29.8%, 6–10 años 34.0%, 11–20 años 17.5%.

## 3. PaddleOCR

- **Modelo de reconocimiento:** `PP-OCRv5_mobile_rec` (por defecto `--model mobile`; también existe `--model server`). El modelo mobile es mucho más rápido, y como el OCR es la etapa más lenta y se corre sobre 4,000 CVs, la velocidad importó. Los CVs son renderizados digitales limpios donde no hacía falta la mayor precisión del modelo server (el CER ya es 0.86%).
- **Configuración:** `lang="en"`, con la clasificación de orientación del documento, la corrección de deformación y la orientación de líneas de texto desactivadas (los CVs son renderizados planos y derechos).
- **Caché:** un JSON por CV en `data/ocr/` con el texto, la confianza y el cuadro delimitador de cada línea. La etapa se puede reanudar.
- **Orden de lectura** (`reading_order` en `src/ocr_extraction.py`): PaddleOCR devuelve los cuadros en el orden de detección, que puede separar el cargo de sus fechas. Los cuadros se reordenan en tres pasos. (1) Un histograma de cobertura horizontal busca, en el 25%–75% central del ancho de la página, un hueco vertical vacío de al menos 3% del ancho; si existe (y la página tiene al menos 8 líneas) la página se divide en columna izquierda y derecha, y si no se trata como una sola columna. (2) Dentro de cada columna, las líneas se agrupan en filas según el centro vertical de sus cuadros (con una tolerancia relativa a la altura mediana de línea). (3) Las filas van de arriba abajo y las líneas de cada fila de izquierda a derecha; la columna izquierda se lee completa antes que la derecha.
- **Reporte de la corrida:** 4,000 CVs procesados, confianza media por línea 0.9825, 0 imágenes fallidas; tiempo medio de OCR por CV 0.48 s (mediana 0.47 s) en una GPU Tesla T4 de Google Colab.

## 4. Calidad del OCR

Medida contra el texto de referencia de los mismos 4,000 CVs, después de normalizar ambos textos de la misma forma (`src/ocr_quality.py`).

| Métrica | Valor |
|---|---|
| Tasa de error de caracteres (CER), media / mediana / p95 | 0.86% / 0.82% / 1.46% |
| Tasa de error de palabras (WER), media / mediana / p95 | 9.95% / 9.52% / 16.3% |
| WER solo con letras y dígitos (`wer_alnum`) | 0.11% |
| Años de referencia encontrados correctamente en el OCR | 99.99% |
| Rangos de fechas completos encontrados correctamente | 99.62% |

Por ocupación (CER / WER): contadores 0.80% / 9.6%, obreros de construcción 0.82% / 10.1%, docentes 0.80% / 8.8%, enfermeros/as 0.93% / 10.2%, desarrolladores 0.97% / 11.2%.

El WER parece alto, pero es casi todo formato: contando solo letras y dígitos baja a 0.11%. Los errores sistemáticos que quedan son una I mayúscula leída como l minúscula (`lmplemented`, `apls`) y, rara vez, un dígito perdido en un mes. Los 10 CVs con mayor CER están en `output/tables/ocr_worst_cvs.csv`.

## 5. Limpieza y enmascarado

`src/preprocessing.py` construye tres variantes de texto a partir del OCR en caché (con el orden de lectura ya aplicado):

- `text_raw`: sin limpieza de formato, pero con los datos personales eliminados y las respuestas directas enmascaradas (esos dos pasos son obligatorios en toda variante).
- `text_clean` (**la que usan los modelos**): unicode y guiones normalizados, guiones de fin de línea y líneas partidas corregidos, y datos personales eliminados.
- `text_clean_lowconf`: igual que `text_clean` pero sin las líneas con confianza menor a 0.6. Ninguna línea cayó por debajo de 0.6 (proporción media de líneas descartadas: 0.0%), así que esta variante es idéntica a `text_clean` y no se entrenó por separado.

**Datos personales eliminados:** nombre (nombre y apellido, también cuando el OCR los pegó, y la línea del nombre en mayúsculas al inicio), correo, teléfono, URLs, dirección y ciudad/estado/código postal. Los nombres usados para borrar salen de un archivo que solo sirve para eliminarlos, nunca como variable. La edad, el sexo, la raza, el nombre y la ubicación nunca son entradas del modelo; el sexo y la raza solo se usan en la revisión de equidad.

**Enmascarado de respuestas directas:** frases como "12 years of experience", la edad y la fecha de nacimiento se reemplazan con un marcador usando expresiones regulares. Antes se leyeron 50 textos OCR al azar (`python src/preprocessing.py --review 50`). **0 de 4,000 CVs** indican la experiencia total, la edad o la fecha de nacimiento, por lo que ningún CV necesitó enmascarado. Las fechas de empleos y de graduación se conservan a propósito: son la evidencia con la que el modelo debe razonar.

## 6. Partición

70% / 15% / 15% por ID de CV, estratificada por ocupación × banda de experiencia (0–2, 3–5, 6–10, 11–20), semilla `42`. Los IDs y las etiquetas están en `data/splits/{train,val,test}.csv` y todos los modelos usan los mismos. El conjunto de prueba no se usó para ajustar nada.

| Ocupación | Train | Val | Test |
|---|---|---|---|
| Contadores y auditores | 560 | 120 | 120 |
| Obreros de construcción | 559 | 120 | 121 |
| Docentes de primaria y secundaria | 559 | 121 | 120 |
| Enfermeros/as registrados | 560 | 120 | 120 |
| Desarrolladores de software | 562 | 119 | 119 |
| **Total** | **2,800** | **600** | **600** |

## 7. Baselines

Todos se ajustan solo con el conjunto de validación. Cada fuerza de regularización elegida queda dentro de su cuadrícula de búsqueda (no en el borde).

- **Mediana:** siempre predice la mediana de entrenamiento, 5.68 años.
- **TF-IDF + Ridge:** n-gramas de palabras 1–2, `min_df=3`, 9,685 variables, **con los dígitos y los años conservados** como tokens porque las fechas son la señal (quitarlos se prueba en la pregunta 3). Ridge con alpha 0.1.
- **Embeddings + Ridge:** `all-MiniLM-L6-v2` congelado. El codificador lee como máximo 256 tokens, así que cada CV se divide en fragmentos de 254 tokens, se codifica cada fragmento y se promedian los vectores. Ridge con alpha 0.01. El codificador no se ajusta.
- **Regla bonus** (`src/baseline_rules.py`): extrae los rangos de fechas con una expresión regular y suma las duraciones de los empleos, fusionando traslapes (si no encuentra fechas usa la mediana). MAE de prueba 0.021.

## 8. BERT como regresor

- **Modelo:** `distilbert-base-uncased` (67M de parámetros), `num_labels=1` (una sola salida de regresión), pérdida de error cuadrático medio (también se probó Huber con delta de 2 años).
- **Variable objetivo:** dividida entre 20 para entrenar; las predicciones se convierten de vuelta a años y se recortan a 0–20.
- **CVs largos:** la estrategia es inicio + final (primeros 128 tokens más el final) con `max_len` 512. En la práctica **0% de los CVs supera 512 tokens y el 100% de los años se conserva**, así que con la longitud por defecto no hay truncamiento. Para comparar de todos modos las estrategias de texto largo, también se corrieron forzadas a 128 tokens (pregunta 3).
- **Entrenamiento:** tasa de aprendizaje 5e-5, tamaño de lote 16, hasta 8 épocas, parada temprana sobre el MAE de validación con paciencia 2 (mejor época: 7), warmup de 6%, weight decay 0.01, semilla 42, GPU Tesla T4, 118 segundos de entrenamiento.

## 9. Resultados

Conjunto de prueba, 600 CVs (`output/tables/model_comparison.csv`):

| Modelo | MAE | RMSE | Acc ±1 | Acc ±2 | Acc ±3 | Acc. de banda | F1 macro de banda | R² | Tiempo de entrenamiento | Inferencia / CV |
|---|---|---|---|---|---|---|---|---|---|---|
| Mediana | 3.42 | 4.44 | 0.190 | 0.365 | 0.525 | 0.340 | 0.127 | -0.033 | 0.0004 s | ~0 |
| TF-IDF + Ridge | 0.99 | 1.36 | 0.618 | 0.893 | 0.967 | 0.815 | 0.823 | 0.904 | 0.4 s | 0.1 ms |
| Embeddings + Ridge | 1.26 | 1.72 | 0.505 | 0.823 | 0.932 | 0.808 | 0.815 | 0.846 | 5.7 s | 2.0 ms |
| BERT ajustado | 0.20 | 0.26 | 0.998 | 1.000 | 1.000 | 0.957 | 0.959 | 0.996 | 118 s (GPU) | 1.4 ms (GPU) |

**Ganancia de BERT** (reducción del MAE, intervalo bootstrap pareado de 95%, 2,000 remuestreos):

| Frente a | MAE del baseline | Ganancia (años) | IC 95% | Ganancia relativa | ¿Significativa? |
|---|---|---|---|---|---|
| Mediana | 3.42 | 3.22 | 3.02 a 3.45 | 94% | Sí |
| TF-IDF + Ridge | 0.99 | 0.79 | 0.72 a 0.87 | 80% | Sí |
| Embeddings + Ridge | 1.26 | 1.06 | 0.97 a 1.16 | 84% | Sí |

Todos los intervalos están lejos de cero. Las figuras (predicho vs real con la recta de 45°, MAE por banda y por ocupación, matrices de confusión por banda) están en `output/figures/`, y la app de Streamlit las muestra de forma interactiva. MAE por banda y por ocupación: `output/tables/mae_by_band.csv` y `mae_by_occupation.csv`.

**Los 10 mayores errores de BERT** (`output/tables/bert_top10_errors.csv`) son todos pequeños, entre 0.71 y 1.09 años. Por causa: **7 tienen un hueco de carrera entre empleos** y **3 son un solo empleo corto** que BERT predice en aproximadamente la mitad. En ninguno el OCR leyó mal una fecha, así que estos errores son del modelo.

**Mejoras posibles:** darle a BERT la aritmética de fechas de forma explícita (por ejemplo, agregar como entrada la estimación de la regla o predecir una corrección sobre ella), aumentar los datos de entrenamiento con fechas desplazadas para que deje de depender de los años absolutos, entrenar con más semillas y reportar la dispersión, y probar con CVs escaneados reales.

## 10. Preguntas analíticas

### Pregunta 1 — ¿Qué aprende BERT que los baselines no pueden?

**BERT supera a todos los baselines aprendidos, pero lo que aprendió es aritmética de fechas, y la hace de forma imperfecta.**

En los 600 CVs de prueba, BERT ajustado llega a un MAE de 0.20 años, frente a 0.99 de TF-IDF + Ridge, 1.26 de los embeddings congelados + Ridge y 3.42 de la mediana. Las ganancias frente a cada baseline son grandes y sus intervalos bootstrap pareados de 95% están lejos de cero (por ejemplo, 0.79 años sobre TF-IDF, intervalo de 0.72 a 0.87). El error de BERT también es plano entre bandas de experiencia (de 0.17 a 0.22 años), mientras que el baseline de la mediana llega a 8.3 años de error en la banda 11–20.

La prueba de +5 años en las fechas muestra *cómo* lo logra. Si un modelo calculara duraciones de verdad, desplazar todas las fechas cinco años no cambiaría nada. La predicción de BERT se movió 2.2 años en promedio y solo el 15% de los CVs quedó a menos de medio año, así que depende en parte de los años absolutos vistos en el entrenamiento (todos los CVs terminan en 2023) y no de las duraciones. Al borrar un empleo, BERT bajó 2.20 años frente a un ideal de 2.59, y bajó en el 95% de los CVs; el tamaño de la caída se correlaciona 0.59 con la duración del empleo borrado. O sea que sí lee las entradas de empleo, pero solo de forma aproximada. TF-IDF casi no reacciona a un empleo borrado (-1.09) y los embeddings reaccionan de más (-2.79).

Una regla que suma las duraciones de los empleos a partir de las fechas tiene un MAE de 0.021 y no se altera con el desplazamiento de fechas. En este formato sintético, la regla es la mejor herramienta; BERT sirve como respaldo cuando faltan fechas o no se pueden leer.

Sus mayores errores son pequeños (el peor es de 1.09 años). Siete de los diez tienen un hueco de carrera entre empleos, y tres son un solo empleo corto que BERT predice en aproximadamente la mitad. En todo el conjunto de prueba, los CVs con hueco tienen un MAE de 0.226 frente a 0.179 de los que no tienen, y los CVs de un solo empleo 0.229 frente a 0.167 de los que tienen cuatro o más. En los diez el OCR leyó bien las fechas, así que estos errores vienen del modelo, no del OCR.

### Pregunta 2 — ¿Cuánto cuesta el ruido del OCR?

**El ruido del OCR casi no cuesta nada aquí, porque los CVs de PopResume son renderizados digitales limpios.**

PaddleOCR lee los 4,000 CVs con una tasa de error de caracteres de 0.86% y una tasa de error de palabras de cerca de 10%. El WER parece alto, pero es casi todo formato: contando solo letras y dígitos (sin puntuación ni símbolos), baja a 0.11%. Lo que queda son unas pocas confusiones sistemáticas, como una I mayúscula leída como l minúscula ("lmplemented", "apls") y la pérdida de un dígito en un mes. Los años se leen bien en el 99.99% de los casos y los rangos de fechas completos en el 99.6%, que es lo que importa para esta tarea.

Evaluar los mismos CVs de prueba con el texto del OCR y con el texto de referencia perfecto da casi el mismo MAE: BERT 0.196 frente a 0.217, TF-IDF 0.986 frente a 0.987, embeddings 1.258 frente a 1.241. Las diferencias están dentro de unos 0.02 años. BERT es ligeramente *mejor* con el texto del OCR, muy probablemente porque se entrenó con texto del OCR y el texto de referencia es un pequeño cambio de distribución, no porque el OCR ayude.

Limitación: estos CVs son sintéticos y están renderizados sin artefactos de escaneo, inclinación ni baja resolución. El resultado dice que el OCR no es el cuello de botella con estos datos; con CVs escaneados reales casi seguro costaría más.

### Pregunta 3 — Sensibilidad metodológica

**Los resultados son más sensibles a que el modelo vea los dígitos que a los detalles de entrenamiento.**

- **Los dígitos importan.** TF-IDF sin dígitos pasa de 0.99 a 1.70 de MAE. La señal está en las fechas.
- **Estrategia de texto largo.** Ningún CV supera 512 tokens, así que con la longitud por defecto no hay truncamiento. Para comparar las estrategias de todos modos, se forzaron a 128 tokens: inicio 0.214, inicio+final 0.305, fragmentos 0.483. Conservar el inicio del CV (donde está el historial laboral) funciona mejor; las otras estrategias descartan o diluyen esa parte.
- **Pérdida y limpieza.** La pérdida de Huber (0.180) y usar el texto sin limpiar en lugar del limpio (0.182) quedan entre 0.015 y 0.02 por debajo de la corrida principal (0.196). Cada variante se entrenó una sola vez, así que diferencias de este tamaño probablemente son ruido de la semilla y no deben leerse como mejoras reales.
- **Regla.** La regla de suma de fechas (0.021) le gana a todos los modelos entrenados, lo que muestra que el error restante no es cuestión del tamaño del modelo sino de si calcula las duraciones con exactitud.

### Pregunta 4 — Valor de negocio, equidad y ética

**El modelo no muestra una brecha de desempeño significativa por sexo ni por raza en estos datos, pero ese resultado tiene límites, y el caso de uso necesita supervisión humana.**

El MAE de BERT es 0.193 para mujeres y 0.200 para hombres; por raza, 0.200 para asiáticos/isleños del Pacífico, 0.188 para personas negras y 0.197 para personas blancas. Las brechas son de 0.012 años como máximo. El error medio con signo es ligeramente positivo en todos los grupos (el modelo sobreestima un poco), con el valor más bajo para personas blancas (0.028 frente a 0.051 y 0.054), que sigue siendo de unas pocas semanas como máximo. Los grupos de personas negras y de asiáticos/isleños del Pacífico tienen solo 62 y 81 CVs de prueba y no se calcularon intervalos de confianza, así que las diferencias pequeñas entre grupos no se pueden separar del ruido.

Por qué la evidencia es limitada: los CVs son sintéticos y la etiqueta es una suma mecánica de fechas, así que el modelo no tiene motivo para usar datos demográficos. Los nombres, correos, teléfonos y direcciones se eliminaron antes de entrenar y los CVs no tienen foto, lo que cierra los sustitutos más obvios. Los CVs reales tienen otros más sutiles (huecos por cuidado de familiares, nombres de colegios, redacción).

Quién podría usarlo y para qué: agencias de reclutamiento y equipos de RR. HH. que revisan miles de CVs (para pre-llenar un perfil u ordenar una pila), bolsas de empleo (para completar perfiles y alimentar filtros de "años de experiencia") y servicios públicos de empleo (para emparejar personas con vacantes y marcar perfiles inconsistentes). Pre-llenar un campo que luego una persona confirma es un uso de bajo riesgo; decidir quién recibe una entrevista es de alto riesgo, y este sistema nunca debe usarse para eso.

Cuánto cuesta un error: nuestro BERT queda a menos de 2 años en el 100% de los CVs de prueba, así que errores de ±2 o ±5 años son hipotéticos para él, pero no para los baselines (TF-IDF se equivoca por más de 2 años en el 11% de los CVs y los embeddings en el 18%). Un error de ±2 años puede mover a un candidato de una banda a otra (3–5 frente a 6–10) y a través de un umbral de "5+ años requeridos"; un error de ±5 años confunde a un junior con un senior. Sobreestimar deja pasar a un candidato poco calificado y gasta entrevistas. Subestimar es peor para el candidato: una persona calificada queda filtrada y, como nadie revisa los CVs rechazados, el error nunca se descubre. En una selección de alto volumen, subestimar es entonces el error más dañino.

Otros riesgos: los CVs contienen datos personales protegidos por la ley peruana de protección de datos personales (Ley 29733). El uso sensato es como apoyo a la decisión, con una persona revisando los casos en que faltan fechas o la regla y el modelo no coinciden, nunca como filtro automático.

## 11. Limitaciones y aplicaciones de negocio

- **Datos sintéticos:** los CVs de PopResume se generan a partir de estadísticas de EE. UU., en inglés, para solo 5 ocupaciones y con una familia de diseños limpios. La etiqueta es básicamente la suma de las duraciones de los empleos (la regla de fechas llega a 0.021 de MAE), así que la tarea es mucho más fácil que con CVs reales y no debe esperarse el puntaje casi perfecto de BERT con datos reales.
- **OCR sobre renderizados limpios:** escaneos reales, fotos o archivos de baja resolución costarían más que la brecha casi nula medida aquí.
- **Una sola semilla:** cada modelo se entrenó una vez; las diferencias de cerca de 0.02 años entre variantes de BERT probablemente son ruido.
- **Revisión de equidad con atributos sintéticos** y subgrupos pequeños (62 y 81 CVs), sin intervalos; no muestra una brecha grande pero no puede probar equidad.
- **Uso:** pre-llenado de perfiles, filtros de búsqueda, alertas de consistencia y ordenamiento para revisión humana. **Nunca** decisiones automáticas de contratación o rechazo.

## 12. Cómo reproducir el proyecto, paso a paso

Las etapas pesadas (OCR y BERT) se corrieron en Google Colab con una Tesla T4: `notebooks/colab_ocr_pipeline.ipynb` (PaddleOCR, que requiere PaddlePaddle) y `notebooks/colab_models_pipeline.ipynb` (modelos de PyTorch). Son cuadernos separados porque PaddlePaddle y PyTorch entran en conflicto en un mismo entorno. Los datos, los modelos y las salidas viven en una carpeta de Drive que no se sube al repositorio.

```bash
# 0. Entorno. PaddlePaddle y PyTorch chocan en un mismo entorno: usa dos entornos (o dos
#    cuadernos de Colab), uno para el OCR y otro para los modelos. Ver los bloques de requirements.txt.
python -m venv .venv && source .venv/bin/activate
pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/   # GPU
# (en CPU: paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/)
pip install -r requirements.txt

# 1. Hugging Face: aceptar las condiciones de sumin-yu/PopResume e iniciar sesión
huggingface-cli login

# 2. Muestreo estratificado y descarga (imágenes, metadata, texto de referencia)
python src/download_data.py --n-per-occ 800 --seed 42 --buffer-size 500

# 3. OCR (lento: usa una GPU; con caché y se puede reanudar)
python src/ocr_extraction.py

# 4. Calidad del OCR, limpieza/enmascarado y partición
python src/ocr_quality.py
python src/preprocessing.py --review 50      # leer 50 textos OCR al azar
python src/preprocessing.py
python src/split.py

# 5. Baselines
python src/baseline_median.py
python src/baseline_tfidf.py
python src/baseline_tfidf.py --keep-digits 0
python src/baseline_embeddings.py
python src/baseline_rules.py

# 6. BERT (corrida principal y variantes de sensibilidad)
python src/train_bert.py                                                    # principal: inicio+final, MSE
python src/train_bert.py --loss huber --name bert_huber
python src/train_bert.py --text-col text_raw --name bert_raw
python src/train_bert.py --strategy head --max-len 128 --name bert_head128
python src/train_bert.py --strategy headtail --max-len 128 --name bert_headtail128
python src/train_bert.py --strategy chunks --max-len 128 --name bert_chunks128

# 7. Evaluación, prueba de fechas y análisis de errores
python src/evaluation.py
python src/probe_dates.py --n 20
python src/error_analysis.py --model bert
```

Las imágenes, los JSON del OCR y los pesos de los modelos **no** se suben al repositorio (ver `.gitignore`); los comandos de arriba los regeneran. Los resultados livianos (`output/tables`, `output/figures`, `output/metrics`, `data/splits`) sí están en el repositorio para poder revisar la app y este README sin volver a correr nada.

## 13. Cómo correr la app de Streamlit

```bash
pip install -r requirements-app.txt
streamlit run app.py
```

La app tiene exactamente cuatro pestañas: Problem & Methodology, OCR & Exploratory Analysis, Model Results e Interactive Experience Estimator (la interfaz está en inglés, igual que el enunciado). El estimador (subir una imagen o PDF de un CV y correr los modelos) necesita los modelos entrenados y PaddleOCR; sin ellos, la app muestra los resultados guardados e indica qué falta. Las respuestas escritas de la pestaña Model Results se leen de `docs/answers.md`.

## Video

El pitch de 5 minutos está enlazado en `video/link.txt`.

## Flujo de trabajo en GitHub

Todo el trabajo se hizo en ramas `feature/...` y se integró a `main` mediante Pull Requests.

## Datos y licencia

PopResume se publica bajo CC BY 4.0 (sumin-yu/PopResume en Hugging Face); cada CV es sintético y no contiene datos de ninguna persona real.
