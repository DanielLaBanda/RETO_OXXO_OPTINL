"""
========================================================================
 MODELO NO LINEAL (random-key) + COMPARACION CONTRA EL GA  (Modelo 1)
 Reto MA2008B - OXXO Planogramacion
========================================================================

Resuelve EXACTAMENTE el mismo problema del Entregable 2 (acomodo de planograma,
Ec. 1) pero con un enfoque de OPTIMIZACION NO LINEAL en vez de la metaheuristica
GA, para poder compararlos sobre la misma instancia.

FORMULACION (identica al documento del modelo matematico)
  Conjuntos:  P={1..N} productos (bloques),  S={1..M} charolas (M=6*TAMANO).
  Parametros: a_j=ANCHO_j*NUM_FRENTES_j, h_j=ALTO_j, b_j=marca, W, w1,w2,w3, rho.
  Variables (exactas): x_ij in {0,1} (producto j -> charola i), p_ij posicion.
  Objetivo:   max Z = w1*U - w2*Ctam + w3*Bmarca - rho*max(0, M_usadas - M)
              (no lineal por el termino de varianza Ctam).
  Restricciones: R1 capacidad de ancho, R2 asignacion unica, R3 posiciones
                 unicas/secuenciales, R4 naturaleza de variables.

ENFOQUE NO LINEAL (random-key)
  En vez de variables binarias x_ij, se usan N variables CONTINUAS y_j in [0,1]
  (random keys). El orden (argsort) de y define una permutacion; esa permutacion
  se decodifica con el MISMO first-fit del GA (Sec. 8.1), garantizando R1-R4 por
  construccion. Asi el problema combinatorio se convierte en una optimizacion
  no lineal continua en [0,1]^N, que se resuelve con un optimizador global no
  lineal (Differential Evolution, sin derivadas). La funcion objetivo Z(y) es
  exactamente la Ec. 1, evaluada sobre el acomodo decodificado.

  -> GA y NLP comparten decoder y funcion objetivo: la comparacion aisla el
     METODO de busqueda (metaheuristica poblacional vs. optimizador no lineal).

USO
  python nlp_model.py [FORMATO ...]
  (por defecto BCO_CF_4.0_Refrescos)
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

# helpers de reporte ya existentes en el repo
from math_model import (
    write_csv,
    write_latex_tables,
    write_location_report,
    draw_line_chart,
    draw_grouped_bar_chart,
)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEP = 0.5     # separador entre productos (cm)
PEN = 50.0    # rho: penalizacion por charola excedente (igual que ga1.py)


# ----------------------------------------------------------------------
# INSTANCIA
# ----------------------------------------------------------------------
@dataclass
class Instance:
    formato: str
    prods: pd.DataFrame
    n_shelves: int
    W: float
    w1: float
    w2: float
    w3: float
    a: np.ndarray          # ancho de bloque = ANCHO*NUM_FRENTES
    h: np.ndarray          # alto
    marca: np.ndarray
    E: np.ndarray          # E_CHAROLA_NORM (ubicacion esperada historica)
    charola_hist: np.ndarray   # charola real (modal) del historico

    @property
    def N(self) -> int:
        return len(self.prods)


def load_instance(formato: str) -> Instance:
    prods = pd.read_csv(DATA_DIR / f"oxxo_instance_{formato}.csv").reset_index(drop=True)
    with open(DATA_DIR / f"oxxo_weights_{formato}.json") as f:
        wj = json.load(f)
    with open(DATA_DIR / f"oxxo_locdist_{formato}.json") as f:
        n_shelves = json.load(f)["n_shelves"]
    return Instance(
        formato=formato, prods=prods, n_shelves=int(n_shelves),
        W=float(wj.get("W_shelf", 55.0)),
        w1=float(wj["w1"]), w2=float(wj["w2"]), w3=float(wj["w3"]),
        a=(prods["ANCHO"] * prods["NUM_FRENTES"]).to_numpy(float),
        h=prods["ALTO"].to_numpy(float),
        marca=prods["MARCA"].to_numpy(),
        E=prods["E_CHAROLA_NORM"].to_numpy(float),
        charola_hist=prods["CHAROLA"].to_numpy(int),
    )


# ----------------------------------------------------------------------
# DECODER (first-fit, Sec. 8.1) + FUNCION OBJETIVO (Ec. 1)  -- igual a ga1
# ----------------------------------------------------------------------
def decode(inst: Instance, perm) -> list[list[int]]:
    sh, cur, used = [], [], 0.0
    for j in perm:
        extra = SEP if cur else 0.0
        if used + extra + inst.a[j] <= inst.W + 1e-9:
            cur.append(int(j)); used += extra + inst.a[j]
        else:
            sh.append(cur); cur = [int(j)]; used = inst.a[j]
    if cur:
        sh.append(cur)
    return sh


def _used_w(inst: Instance, s) -> float:
    return sum(inst.a[j] for j in s) + SEP * max(0, len(s) - 1)


def evaluate_shelves(inst: Instance, sh: list[list[int]]) -> dict:
    """Metricas de la Ec. 1 sobre un acomodo (lista de charolas)."""
    util = float(np.mean([_used_w(inst, s) / inst.W for s in sh]))
    incoh = float(np.mean([np.std([inst.h[j] for j in s]) if len(s) > 1 else 0.0 for s in sh]))
    same, tot = 0, 0
    for s in sh:
        for x, y in zip(s[:-1], s[1:]):
            tot += 1
            if inst.marca[x] == inst.marca[y]:
                same += 1
    block = same / tot if tot > 0 else 0.0
    # consistencia con E[ubicacion] historica (charola normalizada del acomodo)
    n = len(sh)
    loc, cnt = 0.0, 0
    for i, s in enumerate(sh):
        chn = (i + 1) / n
        for j in s:
            loc += 1.0 - abs(chn - inst.E[j]); cnt += 1
    loc = loc / cnt if cnt else 0.0
    over = max(0, len(sh) - inst.n_shelves)
    fitness = inst.w1 * util - inst.w2 * (incoh / 10.0) + inst.w3 * block - PEN * over
    waste = float(np.mean([1 - _used_w(inst, s) / inst.W for s in sh]))
    return dict(fitness=fitness, aprovechamiento_pct=100 * util, desperdicio_pct=100 * waste,
                incoherencia_cm=incoh, agrupacion_marca_pct=100 * block,
                valor_esperado_ubicacion=loc, n_shelves_used=len(sh), overflow=over)


def fitness_perm(inst: Instance, perm) -> float:
    return evaluate_shelves(inst, decode(inst, perm))["fitness"]


# ----------------------------------------------------------------------
# METODO 1: GA  (misma configuracion que genetic_algorithm/ga1.py)
# ----------------------------------------------------------------------
def run_ga(inst: Instance, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    N = inst.N

    def ox(p1, p2):
        n = len(p1); a, b = sorted(rng.choice(n, 2, replace=False))
        c = [-1] * n; c[a:b + 1] = p1[a:b + 1]
        sset = set(p1[a:b + 1]); fill = [g for g in p2 if g not in sset]; k = 0
        for i in list(range(b + 1, n)) + list(range(0, a)):
            c[i] = fill[k]; k += 1
        return c

    def mut(p, r=0.2):
        p = p[:]
        if rng.random() < r:
            i, j = rng.choice(len(p), 2, replace=False); p[i], p[j] = p[j], p[i]
        if rng.random() < r:
            i, j = rng.choice(len(p), 2, replace=False); g = p.pop(i); p.insert(j, g)
        return p

    def tourn(pop, fits, k=3):
        idx = rng.choice(len(pop), k, replace=False)
        return pop[idx[np.argmax([fits[i] for i in idx])]][:]

    POP, GENS, ELITE = 150, 400, 8
    obs = list(np.argsort(inst.h)); pop = []
    for _ in range(POP // 2):
        p = obs[:]
        for _ in range(rng.integers(2, 8)):
            i, j = rng.choice(N, 2, replace=False); p[i], p[j] = p[j], p[i]
        pop.append(p)
    pop += [list(rng.permutation(N)) for _ in range(POP - len(pop))]

    t0 = time.time()
    best_hist, best, bf = [], None, -1e9
    for _ in range(GENS):
        fits = [fitness_perm(inst, p) for p in pop]
        order = np.argsort(fits)[::-1]
        if fits[order[0]] > bf:
            bf = fits[order[0]]; best = pop[order[0]][:]
        best_hist.append(bf)
        npop = [pop[order[i]][:] for i in range(ELITE)]
        while len(npop) < POP:
            npop.append(mut(ox(tourn(pop, fits), tourn(pop, fits))))
        pop = npop
    dt = time.time() - t0

    m = evaluate_shelves(inst, decode(inst, best))
    m.update(algoritmo="GA main", formato=inst.formato, tiempo_s=round(dt, 2))
    m["_perm"] = best
    m["_hist"] = best_hist
    return m


# ----------------------------------------------------------------------
# METODO 2: NLP random-key  (optimizador no lineal global)
# ----------------------------------------------------------------------
def run_nlp(inst: Instance, seed: int = 7, maxiter: int = 150, popsize: int = 8) -> dict:
    N = inst.N
    running_best = {"val": -1e9}
    hist = []

    def neg_obj(y):
        perm = np.argsort(y)
        f = fitness_perm(inst, perm)
        if f > running_best["val"]:
            running_best["val"] = f
        hist.append(running_best["val"])
        return -f

    t0 = time.time()
    res = differential_evolution(
        neg_obj, bounds=[(0.0, 1.0)] * N, seed=seed, maxiter=maxiter, popsize=popsize,
        strategy="best1bin", mutation=(0.5, 1.0), recombination=0.7, tol=1e-7,
        init="sobol", polish=False, updating="deferred",
    )
    dt = time.time() - t0

    best_perm = np.argsort(res.x)
    m = evaluate_shelves(inst, decode(inst, best_perm))
    m.update(algoritmo="NLP random-key", formato=inst.formato, tiempo_s=round(dt, 2))
    m["_perm"] = best_perm
    # convergencia: mejor-hasta-ahora submuestreado a ~200 puntos
    if hist:
        step = max(1, len(hist) // 200)
        m["_hist"] = hist[::step]
    else:
        m["_hist"] = [m["fitness"]]
    m["_nfev"] = int(res.nfev)
    return m


# ----------------------------------------------------------------------
# BENCHMARK: acomodo HISTORICO real
# ----------------------------------------------------------------------
def run_historico(inst: Instance) -> dict:
    # agrupar productos por su charola real (modal) del historico
    shelves: dict[int, list[int]] = {}
    for j, ch in enumerate(inst.charola_hist):
        shelves.setdefault(int(ch), []).append(j)
    sh = [shelves[k] for k in sorted(shelves)]
    m = evaluate_shelves(inst, sh)
    m.update(algoritmo="Historico", formato=inst.formato, tiempo_s=0.0)
    return m


# ----------------------------------------------------------------------
# TAMANO DEL PROBLEMA (nodos/arcos/variables/parametros)
# ----------------------------------------------------------------------
def problem_size(inst: Instance) -> dict:
    N, M = inst.N, inst.n_shelves
    # Grafo equivalente = asignacion bipartita Productos-Charolas:
    #   nodos = N+M (un nodo por producto y por charola)
    #   arcos = N*M (cada posible asignacion producto->charola es un arco = x_ij)
    # NLP random-key: N variables continuas (una clave por producto)
    # Parametros: a_j,h_j,b_j por producto (3N) + W,w1,w2,w3,rho,M (6 escalares)
    return dict(
        formato=inst.formato,
        nodos=N + M,
        arcos=N * M,
        variables_random_key=N,
        variables_grafo_equiv=N * M,
        parametros=3 * N + 6,
    )


# ----------------------------------------------------------------------
# SENSIBILIDAD (sobre el formato principal)
# ----------------------------------------------------------------------
def run_sensibilidad(inst: Instance) -> list[dict]:
    escenarios = [
        ("base", inst.w1, inst.w2, inst.w3),
        ("+util (w1*1.5)", inst.w1 * 1.5, inst.w2, inst.w3),
        ("+coherencia (w2*3)", inst.w1, inst.w2 * 3.0, inst.w3),
        ("+marca (w3*3)", inst.w1, inst.w2, inst.w3 * 3.0),
    ]
    rows = []
    for nombre, w1, w2, w3 in escenarios:
        variant = Instance(**{**inst.__dict__, "w1": w1, "w2": w2, "w3": w3})
        nlp = run_nlp(variant, maxiter=120, popsize=8)
        rows.append(dict(escenario=nombre, fitness=nlp["fitness"],
                         aprovechamiento_pct=nlp["aprovechamiento_pct"],
                         desperdicio_pct=nlp["desperdicio_pct"],
                         incoherencia_cm=nlp["incoherencia_cm"],
                         agrupacion_marca_pct=nlp["agrupacion_marca_pct"]))
    return rows


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main(formatos: list[str]) -> None:
    results: list[dict] = []
    size_rows: list[dict] = []
    sens_rows: list[dict] = []
    main_inst = None

    for fmt in formatos:
        print(f"\n==================== {fmt} ====================")
        inst = load_instance(fmt)
        size = problem_size(inst)
        print(f"  N={inst.N} bloques | M={inst.n_shelves} charolas | "
              f"nodos={size['nodos']} arcos={size['arcos']} "
              f"var_nlp={size['variables_random_key']} params={size['parametros']}")

        hist = run_historico(inst)
        print(f"  Historico : fit={hist['fitness']:.3f} aprov={hist['aprovechamiento_pct']:.1f}% "
              f"incoh={hist['incoherencia_cm']:.2f} marca={hist['agrupacion_marca_pct']:.1f}% "
              f"char={hist['n_shelves_used']}")

        ga = run_ga(inst)
        print(f"  GA main   : fit={ga['fitness']:.3f} aprov={ga['aprovechamiento_pct']:.1f}% "
              f"incoh={ga['incoherencia_cm']:.2f} marca={ga['agrupacion_marca_pct']:.1f}% "
              f"char={ga['n_shelves_used']} | t={ga['tiempo_s']}s")

        nlp = run_nlp(inst)
        print(f"  NLP rndkey: fit={nlp['fitness']:.3f} aprov={nlp['aprovechamiento_pct']:.1f}% "
              f"incoh={nlp['incoherencia_cm']:.2f} marca={nlp['agrupacion_marca_pct']:.1f}% "
              f"char={nlp['n_shelves_used']} | t={nlp['tiempo_s']}s nfev={nlp.get('_nfev')}")

        for row in (hist, ga, nlp):
            merged = {**row, **size}            # adjunta tamano del problema a cada fila
            merged.pop("_perm", None); merged.pop("_hist", None); merged.pop("_nfev", None)
            results.append(merged)
        size_rows.append(size)

        # graficas + reporte de ubicacion para el formato principal
        if main_inst is None:
            main_inst = inst
            draw_line_chart(
                OUT_DIR / f"convergencia_{fmt}.png",
                f"Convergencia: GA vs NLP random-key  ({fmt})",
                [("GA main", ga["_hist"], (29, 158, 117)),
                 ("NLP random-key", nlp["_hist"], (124, 92, 191))],
            )
            draw_grouped_bar_chart(
                OUT_DIR / f"criterios_{fmt}.png",
                f"Comparacion por criterio  ({fmt})",
                ["Aprov.%", "Desp.%", "Incoh.cm", "Marca%", "E[ubic]"],
                [("Historico", [hist["aprovechamiento_pct"], hist["desperdicio_pct"],
                                hist["incoherencia_cm"], hist["agrupacion_marca_pct"],
                                100 * hist["valor_esperado_ubicacion"]], (150, 180, 210)),
                 ("GA main", [ga["aprovechamiento_pct"], ga["desperdicio_pct"],
                              ga["incoherencia_cm"], ga["agrupacion_marca_pct"],
                              100 * ga["valor_esperado_ubicacion"]], (29, 158, 117)),
                 ("NLP", [nlp["aprovechamiento_pct"], nlp["desperdicio_pct"],
                          nlp["incoherencia_cm"], nlp["agrupacion_marca_pct"],
                          100 * nlp["valor_esperado_ubicacion"]], (124, 92, 191))],
                y_label="valor",
            )
            write_location_report(inst)
            print(f"  Sensibilidad sobre {fmt} (NLP)...")
            sens_rows = run_sensibilidad(inst)

    # tablas LaTeX + CSV de resultados
    write_latex_tables(results, sens_rows)
    clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
    write_csv(OUT_DIR / "comparison_results.csv", clean)
    print(f"\nListo. Tablas -> outputs/comparison_tables.tex | CSV -> outputs/comparison_results.csv")
    print("Graficas -> outputs/convergencia_*.png, outputs/criterios_*.png")


if __name__ == "__main__":
    args = sys.argv[1:] or ["BCO_CF_4.0_Refrescos"]
    main(args)
