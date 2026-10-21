"""Utilidades para leer y modificar las fechas laborales de un CV (texto limpio).

Formato de PopResume:  "Cargo, Empresa, 2018.05-2020.05"  (año.mes - año.mes) y la educación
con un solo año ("..., 2014"). Aquí se usa para:
  - la referencia basada en reglas (suma de duraciones de los cargos),
  - la prueba de la Pregunta 1 (mover todas las fechas +5 años o borrar un cargo).
"""
from __future__ import annotations

import re

RANGE = re.compile(r"\b((?:19|20)\d{2})(?:\.(\d{1,2}))?\s*-\s*((?:19|20)\d{2})(?:\.(\d{1,2}))?\b")
YEAR = re.compile(r"\b(19|20)(\d{2})\b")


def _months(year: str, month: str | None) -> int:
    m = int(month) if month else 1
    return int(year) * 12 + (min(max(m, 1), 12) - 1)


def parse_jobs(text: str) -> list[dict]:
    """Cargos = líneas con un rango de fechas. Devuelve [{line, start, end}] (meses absolutos)."""
    jobs = []
    for i, line in enumerate(str(text).splitlines()):
        m = RANGE.search(line)
        if m:
            s, e = _months(m.group(1), m.group(2)), _months(m.group(3), m.group(4))
            if e >= s:
                jobs.append({"line": i, "start": s, "end": e})
    return jobs


def years_from_jobs(jobs: list[dict]) -> float | None:
    """Años trabajados = unión de los intervalos (los solapes no se cuentan dos veces)."""
    if not jobs:
        return None
    total, cur_s, cur_e = 0, None, None
    for j in sorted(jobs, key=lambda j: j["start"]):
        if cur_e is None or j["start"] > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = j["start"], j["end"]
        else:
            cur_e = max(cur_e, j["end"])
    total += cur_e - cur_s
    return total / 12.0


def years_from_text(text: str) -> float | None:
    return years_from_jobs(parse_jobs(text))


def shift_years(text: str, k: int = 5) -> str:
    """Suma k años a TODOS los años del texto (cargos y educación)."""
    return YEAR.sub(lambda m: str(int(m.group(0)) + k), str(text))


def delete_job(text: str, job_line: int) -> str:
    """Borra la línea del cargo y sus viñetas (hasta el siguiente cargo o encabezado)."""
    lines = str(text).splitlines()
    job_lines = {j["line"] for j in parse_jobs(text)}
    out, skipping = [], False
    for i, ln in enumerate(lines):
        if i == job_line:
            skipping = True
            continue
        if skipping:
            is_header = ln.strip().isupper() and len(ln.strip()) > 3
            if i in job_lines or is_header:
                skipping = False
            else:
                continue
        out.append(ln)
    return "\n".join(out)
