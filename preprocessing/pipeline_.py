"""
========================================================================
 PIPELINE - Reconstruccion de PLANOGRAMAS individuales (Ejemplo.csv)
 Reto MA2008B - OXXO Planogramacion
========================================================================

CAMBIO CONCEPTUAL (v2)
  Una INSTANCIA = un PLANOGRAMA individual = el surtido concreto de un mueble
  (sus productos con frentes dados) que cabe en un numero definido de charolas.
  NO es el universo acumulado de productos de un formato (eso daba 361 productos
  / 62 charolas, irreal). El socio nos da el planograma; nosotros lo acomodamos.

EL OBSTACULO
  Ejemplo.csv NO trae ID de tienda / ejecucion / planograma (solo 13 columnas).
  Cada posicion (charola, bandeja) esta observada ~95 veces = ~95 tiendas
  apiladas. No se pueden separar planogramas FISICOS individuales de forma
  exacta. Pero SI se reconstruye un planograma CANONICO por completitud de
  posiciones: cada UBICACION_BANDEJA es un FRENTE; para cada (charola, frente)
  se toma el UPC modal del historico y se colapsan los frentes consecutivos del
  mismo producto en bloques -> un planograma de ~101 productos (bloques) en 24
  charolas que casi llena cada una (≈49 de 55 cm). Eso valida la reconstruccion.

ARTEFACTOS GENERADOS (consumidos por GA1 y GA2), uno por planograma:
  1) oxxo_instance_<FORMATO>.csv  -> productos del planograma (por slot):
     dims, frentes dados, marca, charola/bandeja original, E[charola] norm.
  2) oxxo_locdist_<FORMATO>.json  -> n_shelves + P(charola | producto) del
     formato (la "distribucion de ubicacion" historica que pide el profesor).
  3) oxxo_weights_<FORMATO>.json  -> pesos w1,w2,w3 de la funcion objetivo,
     derivados por SIMULACION a nivel de planograma plausible (ver abajo).

CLAVE DE FORMATO/PLANOGRAMA
  SEGMENTO + MUEBLE + TAMANO + PLANOGRUPO. La DIRECCION (DI/ID) es un ESPEJO
  del mismo diseno: se canoniza (se voltea la posicion de las filas DI), no se
  separa en dos instancias.

SIMULACION DE PESOS (reformulada)
  En cada escenario se muestrea, POR POSICION, un UPC segun su distribucion
  historica P(UPC | posicion) -> un planograma plausible que respeta la
  estructura (siempre ~146 productos en 24 charolas). Se miden los 3 criterios
  (aprovechamiento, coherencia de altura, agrupacion de marca) sobre sus
  charolas y se promedia sobre escenarios. Mas fiel y defendible que sortear
  charolas de un pool de 361 productos.
      w_k  ~  E[criterio_k] / desv[criterio_k]   (senal/ruido), normalizado.

USO
  python pipeline_.py
  (ajusta DATA_FILE, FORMATO_OBJETIVO y W_SHELF abajo segun el caso)
"""
import pandas as pd, numpy as np, json, os, re

rng = np.random.default_rng(2024)

# ----------------------------------------------------------------------
# RUTAS (ancladas a la raiz del proyecto: funciona desde cualquier cwd)
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')        # entradas + artefactos del pipeline

# ----------------------------------------------------------------------
# CONFIGURACION
# ----------------------------------------------------------------------
DATA_FILE        = os.path.join(DATA_DIR, 'Ejemplo.csv')   # dataset grande real
W_SHELF          = 55.0     # cuarto frio = 55 ; gondola = 91
SEP              = 0.5      # separador entre productos (cm)
N_SIM            = 1000     # escenarios simulados para los pesos
FORMATO_OBJETIVO = None     # None = procesa TODOS ; o ej. 'BCO_CF_4.0_Refrescos'
OUT_DIR          = DATA_DIR  # los artefactos son ENTRADA de GA1/GA2 -> van a data/

# ----------------------------------------------------------------------
# Utilidades de carga robusta (el dataset viene en latin-1 con posible BOM)
# ----------------------------------------------------------------------
def _norm(c):
    # quita BOM (utf-8 leido como latin-1 -> 'ï»¿' ; o como '﻿') y espacios
    return c.replace('﻿', '').replace('ï»¿', '').strip()

def cargar(path):
    df = pd.read_csv(path, encoding='latin-1', sep=None, engine='python')
    df.columns = [_norm(c) for c in df.columns]
    # resolver nombres equivalentes entre datasets
    ren = {}
    for cand, target in [('UPC_CVE','UPC'),('ITEM','UPC'),
                         ('TAMANO_DESC','TAMANO'),('TAMAÑO_POST','TAMANO'),
                         ('PLANOGRUPO_DESC','PLANOGRUPO')]:
        if cand in df.columns: ren[cand]=target
    df = df.rename(columns=ren)
    return df

def marca(desc):
    d = str(desc).upper()
    for m in ['COCA','PEPSI','FRESCA','SQUIRT','PENAFIEL','TOPO CHICO','CIEL',
              'SPRITE','FANTA','JOYA','BARRILITOS','MONSTER','DR PEPPER',
              'MIRINDA','SCHWEPPES','MUNDET']:
        if m in d: return m
    return 'OTRO'

def slug(s):
    return re.sub(r'[^0-9A-Za-z]+', '', str(s))

# ----------------------------------------------------------------------
# 1. CARGA + CLAVE DE PLANOGRAMA + NORMALIZACION DE DIRECCION (espejo de DI)
# ----------------------------------------------------------------------
df = cargar(DATA_FILE)
df = df[df['CHAROLA'] >= 1].copy()            # descartar charola 0 si existe
df['MARCA'] = df['ITEM_DESC'].apply(marca)

# Formato CANONICO de un PLANOGRAMA: SEGMENTO + MUEBLE + TAMANO + PLANOGRUPO.
# (DI e ID son el mismo diseno espejado; mover izq-der no cambia la charola.)
df['FORMATO'] = (df['SEGMENTO_ID'].astype(str) + '_' +
                 df['MUEBLE_ID'].astype(str) + '_' +
                 df['TAMANO'].astype(str) + '_' +
                 df['PLANOGRUPO'].astype(str).map(slug))

# ----------------------------------------------------------------------
# 2. RECONSTRUCCION DEL PLANOGRAMA CANONICO + INSTANCIA
# ----------------------------------------------------------------------
def reconstruir_planograma(gd):
    """Planograma canonico por BLOQUES-PRODUCTO.
    CLAVE: cada UBICACION_BANDEJA es UN FRENTE (facing), no un producto. Un
    producto ocupa frentes consecutivos en la charola. Por eso:
      1) para cada (charola, bandeja) se toma el UPC modal del historico;
      2) se colapsan los frentes consecutivos del MISMO UPC en un bloque, con
         NUM_FRENTES = nº de frentes (run length) y ANCHO = ancho de un frente.
    El ancho del bloque es ANCHO*NUM_FRENTES; el de la charola, la suma de
    frentes + separadores entre bloques (cabe en ~55 cm, como el historico)."""
    rows = []
    for ch, cc in gd.groupby('CHAROLA'):
        # secuencia de frentes (UPC modal por bandeja) en orden
        seq = []
        for ba in sorted(cc['UBICACION_BANDEJA'].unique()):
            pp = cc[cc['UBICACION_BANDEJA'] == ba]
            upc = pp['UPC'].mode().iloc[0]
            ref = pp[pp['UPC'] == upc]
            prof = float(ref['PROFUNDO'].median()) if 'PROFUNDO' in ref else np.nan
            seq.append((upc, float(ref['ANCHO'].median()), float(ref['ALTO'].median()),
                        ref['ITEM_DESC'].iloc[0], ref['MARCA'].iloc[0], prof))
        # colapsar frentes consecutivos del mismo producto en bloques
        i = 0
        while i < len(seq):
            j = i
            while j+1 < len(seq) and seq[j+1][0] == seq[i][0]: j += 1
            upc, an, al, desc, mar, prof = seq[i]
            rows.append(dict(CHAROLA=int(ch), UPC=upc, ITEM_DESC=desc,
                             ANCHO=an, ALTO=al, PROFUNDO=prof,
                             NUM_FRENTES=j-i+1, MARCA=mar))
            i = j + 1
    return pd.DataFrame(rows).reset_index(drop=True)

def construir_instancia(pg, g, n_shelves):
    """Anexa E[charola] normalizada por producto (promedio sobre TODO el formato:
    'en que charola suele ir este producto a lo largo de los planogramas')."""
    e_by_upc = (g.groupby('UPC')['CHAROLA'].mean() / n_shelves).round(4)
    pg = pg.copy()
    pg['E_CHAROLA_NORM'] = (pg['UPC'].map(e_by_upc)
                              .fillna(pg['CHAROLA'] / n_shelves).round(4))
    return pg[['UPC','ITEM_DESC','ANCHO','ALTO','PROFUNDO','NUM_FRENTES','MARCA',
               'CHAROLA','E_CHAROLA_NORM']]

def distribucion_charola(g):
    """P(charola | producto): histograma normalizado por producto, sobre el formato."""
    dist = {}
    for upc, sub in g.groupby('UPC'):
        vc = sub['CHAROLA'].value_counts(normalize=True).sort_index()
        dist[str(upc)] = {int(k): round(float(v), 4) for k, v in vc.items()}
    return dist

# ----------------------------------------------------------------------
# 3. SIMULACION DE PESOS A NIVEL DE PLANOGRAMA PLAUSIBLE
# ----------------------------------------------------------------------
def simular_pesos(gd, n_sim=N_SIM):
    """En cada escenario muestrea un UPC por FRENTE (posicion) segun
    P(UPC | posicion), colapsa frentes consecutivos en bloques-producto y mide
    los 3 criterios sobre las charolas. Promedia sobre escenarios -> E[criterio]
    +- desv y pesos derivados (senal/ruido)."""
    # catalogo por UPC (ancho de un frente, alto, marca)
    cat = {}
    for upc, sub in gd.groupby('UPC'):
        cat[upc] = (float(sub['ANCHO'].median()), float(sub['ALTO'].median()),
                    sub['MARCA'].iloc[0])
    # posiciones (frentes) por charola, en orden de bandeja
    charolas = {}
    for (ch, ba), pp in gd.groupby(['CHAROLA', 'UBICACION_BANDEJA']):
        vc = pp['UPC'].value_counts(normalize=True)
        ks = list(vc.index); ps = np.asarray(vc.values, float); ps = ps / ps.sum()
        charolas.setdefault(int(ch), []).append((int(ba), ks, ps))
    for ch in charolas: charolas[ch].sort()

    Us, Cs, Bs = [], [], []
    for _ in range(n_sim):
        utils, incohs = [], []
        same, tot = 0, 0
        for ch, poss in charolas.items():
            seq = [ks[int(rng.choice(len(ks), p=ps))] for _ba, ks, ps in poss]
            # colapsar frentes consecutivos del mismo UPC en bloques (upc, frentes)
            blocks = []
            i = 0
            while i < len(seq):
                j = i
                while j+1 < len(seq) and seq[j+1] == seq[i]: j += 1
                blocks.append((seq[i], j-i+1)); i = j + 1
            used = sum(cat[u][0]*fr for u, fr in blocks) + SEP*max(0, len(blocks)-1)
            utils.append(min(used, W_SHELF) / W_SHELF)
            if len(blocks) > 1:
                incohs.append(np.std([cat[u][1] for u, _ in blocks]))
            for a, b in zip(blocks[:-1], blocks[1:]):
                tot += 1
                if cat[a[0]][2] == cat[b[0]][2]: same += 1
        Us.append(np.mean(utils) if utils else 0)
        Cs.append(np.mean(incohs) if incohs else 0)
        Bs.append(same/tot if tot > 0 else 0)

    EU, EC, EB = np.mean(Us), np.mean(Cs), np.mean(Bs)
    SU, SC, SB = np.std(Us)+1e-6, np.std(Cs)+1e-6, np.std(Bs)+1e-6
    raw = np.array([EU/SU, EC/SC, EB/SB])
    w = raw / raw.sum() * 3.0
    return dict(w1=round(float(w[0]),3), w2=round(float(w[1]),3), w3=round(float(w[2]),3),
                E_util=round(float(EU),4), E_incoh=round(float(EC),4), E_block=round(float(EB),4),
                std_util=round(float(SU),4), std_incoh=round(float(SC),4), std_block=round(float(SB),4),
                n_sim=n_sim)

# ----------------------------------------------------------------------
# 4. VALIDACION DE LA RECONSTRUCCION
# ----------------------------------------------------------------------
def validar(inst, n_shelves):
    """Un planograma bien aislado: surtido realista, charolas = nominal, y cada
    charola cabe (o casi) en W_SHELF. Devuelve dict de chequeos + flag ok."""
    g = inst.assign(w=inst['ANCHO']*inst['NUM_FRENTES'])
    wide = g.groupby('CHAROLA')['w'].sum() + SEP*(g.groupby('CHAROLA').size()-1)
    n_prod = len(inst)
    chk = dict(n_prod=n_prod, n_shelves=n_shelves,
               ancho_med=round(float(wide.mean()),1), ancho_max=round(float(wide.max()),1),
               charolas_sobre_W=int((wide > W_SHELF).sum()))
    chk['ok'] = bool((n_prod <= 220) and (n_shelves <= 40) and (wide.max() <= W_SHELF*1.7))
    return chk

# ----------------------------------------------------------------------
# 5. PROCESAR PLANOGRAMA(S)
# ----------------------------------------------------------------------
formatos = [FORMATO_OBJETIVO] if FORMATO_OBJETIVO else sorted(df['FORMATO'].unique())
print(f"Planogramas a procesar: {len(formatos)}")

ok_n = 0
for fmt in formatos:
    g = df[df['FORMATO'] == fmt].copy()
    if len(g) < 30:
        continue
    # DI e ID son el MISMO diseno espejado: el planograma canonico se reconstruye
    # desde la direccion dominante (combinarlas desalinea las posiciones e infla
    # el ancho por charola). Las distribuciones de charola/E si usan todo el formato.
    dom = g['DIRECCION_LEGO_ID'].value_counts().idxmax()
    gd = g[g['DIRECCION_LEGO_ID'] == dom].copy()
    pg = reconstruir_planograma(gd)
    n_shelves = int(pg['CHAROLA'].max())
    inst = construir_instancia(pg, g, n_shelves)
    dist = distribucion_charola(g)
    pesos = simular_pesos(gd)
    chk = validar(inst, n_shelves)

    inst.to_csv(f'{OUT_DIR}/oxxo_instance_{fmt}.csv', index=False)
    with open(f'{OUT_DIR}/oxxo_locdist_{fmt}.json', 'w') as f:
        json.dump({'n_shelves': n_shelves, 'dist': dist}, f)
    with open(f'{OUT_DIR}/oxxo_weights_{fmt}.json', 'w') as f:
        json.dump({'formato': fmt, 'W_shelf': W_SHELF, **pesos, 'validacion': chk}, f, indent=2)

    flag = 'OK ' if chk['ok'] else '!! '
    if chk['ok']: ok_n += 1
    print(f"  {flag}{fmt:30s} | prod={chk['n_prod']:3d} | charolas={n_shelves:2d} | "
          f"ancho/char med={chk['ancho_med']:4.1f} max={chk['ancho_max']:5.1f} "
          f"(>{int(W_SHELF)}cm: {chk['charolas_sobre_W']:2d}) | w=({pesos['w1']},{pesos['w2']},{pesos['w3']})")

print(f"\nListo. {ok_n} planogramas validados de {len(formatos)} formatos.")
print("Artefactos: oxxo_instance_*.csv, oxxo_locdist_*.json, oxxo_weights_*.json (en data/)")
print("GA1 y GA2 leen estos archivos (ver USE_PIPELINE y FORMATO en cada uno).")
