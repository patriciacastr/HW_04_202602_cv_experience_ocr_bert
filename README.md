# cv_experience_ocr_bert

Predicción de los **años de experiencia laboral** (0–20) a partir de la **imagen de un CV**:
PaddleOCR extrae el texto y se comparan cuatro modelos (mediana, TF-IDF + Ridge,
embeddings + Ridge y BERT ajustado como regresor) sobre el dataset sintético PopResume.

> Estado: **borrador inicial** (se irá completando con los resultados de cada etapa).
> Las secciones marcadas con `TODO` se llenan con números propios al terminar cada etapa.

## 1. ¿Qué hace el proyecto y para quién?
TODO: pipeline imagen → OCR → limpieza → 4 modelos → evaluación. Usuarios potenciales
(agencias de reclutamiento, bolsas de empleo, servicios públicos de empleo, equipos de RR. HH.).

## 2. Muestreo de CVs
- Dataset: PopResume (CC BY 4.0), solo imágenes **sin foto** (`has_photo == False`).
- 800 CVs por ocupación × 5 ocupaciones = 4 000 (mínimo aceptado: 400 por ocupación).
- Semilla fija: `42`. Los IDs muestreados quedan en `data/raw/sampled_ids.txt`.
- Objetivo: `exp_year` recortado a [0, 20].
- TODO: distribución de experiencia y cuántos CVs están en el tope de 20.

## 3. PaddleOCR
TODO: modelo de reconocimiento usado (PP-OCRv5 mobile/server) y por qué; reconstrucción del
orden de lectura (columnas, filas); tiempo por CV, hardware, confianza media, imágenes fallidas.

## 4. Calidad del OCR
TODO: CER, WER (media por ocupación) y % de años leídos correctamente.

## 5. Limpieza y enmascarado
TODO: reglas de limpieza, umbral de confianza, frases enmascaradas (cuántos CVs afectados).

## 6. Split
TODO: 70/15/15 estratificado por ocupación y banda de experiencia, semilla 42.

## 7. Baselines
TODO: mediana, TF-IDF + Ridge, embeddings + Ridge.

## 8. BERT como regresor
TODO: modelo, hiperparámetros, estrategia para CVs largos.

## 9. Resultados
TODO: tabla `output/tables/model_comparison.csv` y ganancia de BERT con IC 95 %.

## 10. Preguntas analíticas
TODO: Q1 (qué aprende BERT), Q2 (costo del ruido OCR), Q3 (sensibilidad), Q4 (negocio, equidad y ética).

## 11. Limitaciones y aplicaciones
TODO.

## 12. Cómo reproducir (paso a paso)

```bash
# 0. Entorno (instalar PaddlePaddle primero; ver comentarios en requirements.txt)
python -m venv .venv && source .venv/bin/activate
pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
pip install -r requirements.txt

# 1. Hugging Face: aceptar condiciones del dataset sumin-yu/PopResume y loguearse
huggingface-cli login

# 2. Muestreo y descarga (imágenes, metadata, texto de referencia)
python src/download_data.py --n-per-occ 800 --seed 42

# 3. OCR (lento: conviene GPU / Colab; usa caché, se puede reanudar)
python src/ocr_extraction.py

# 4+. (se agregan a medida que se implementan)
#   python src/ocr_quality.py
#   python src/preprocessing.py
#   python src/split.py
#   python src/baseline_median.py && python src/baseline_tfidf.py && python src/baseline_embeddings.py
#   python src/train_bert.py
#   python src/evaluation.py
```

Las imágenes, los JSON de OCR y los pesos de modelos **no se suben a GitHub** (`.gitignore`);
se regeneran con los comandos de arriba.

## 13. Cómo correr la app

```bash
streamlit run app.py
```

## Flujo de trabajo en GitHub
Ramas `feature/...` + Pull Requests hacia `main` (no se trabaja directo en `main`).
