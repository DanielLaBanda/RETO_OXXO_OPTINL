"""
Modelo 1 - Algoritmo Genetico para acomodo de planograma OXXO
Reto MA2008B - Optimizacion No Lineal

Problema: dados N productos (con su ancho, alto y numero de frentes ya decididos)
y un mueble con S charolas de 55 cm de ancho, decidir en que charola va cada
producto y en que orden, MAXIMIZANDO la calidad del acomodo (sin usar ventas).

Fitness = w1*Aprovechamiento - w2*Incoherencia_tamano + w3*Agrupacion_marca
          - penalizacion por infactibilidad
"""
import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# ----------------------------------------------------------------------
# 1. Cargar una instancia real (un planograma historico)
# ----------------------------------------------------------------------
df = pd.read_csv('/mnt/user-data/uploads/ejemplo_planograma.csv', encoding='latin-1', sep=',', engine='python')
df.columns = [c.replace('ï»¿','').replace('Ã\x91','Ñ') for c in df.columns]

def marca(d):
    d = str(d).upper()
    for m in ['COCA','PEPSI','FRESCA','SQUIRT','PENAFIEL','TOPO CHICO','CIEL','SPRITE',
              'FANTA','JOYA','BARRILITOS','MONSTER','DR PEPPER','MIRINDA','SCHWEPPES','MUNDET']:
        if m in d: return m
    return 'OTRO'
df['MARCA'] = df['ITEM_DESC'].apply(marca)

# Instancia: HRN, tamano 3.0, direccion DI (18 charolas) -> mas chica para demo clara
inst = df[(df['SEGMENTO_ID']=='HRN') & (df['TAMAÑO_POST']==3.0) &
          (df['DIRECCION_LEGO_ID']=='DI') & (df['CONJUNTO_ID']=='10MON')].copy()

W_SHELF = 55.0                          # ancho de charola (cm)
N_SHELVES = int(inst['CHAROLA'].max())  # charolas disponibles

prods = inst[['ITEM','ITEM_DESC','ANCHO','ALTO','NUM_FRENTES','MARCA']].reset_index(drop=True)
prods['W'] = prods['ANCHO'] * prods['NUM_FRENTES']   # ancho ocupado total
N = len(prods)
ancho = prods['W'].values
alto  = prods['ALTO'].values
marca_arr = prods['MARCA'].values

print(f"Instancia: HRN_3.0_DI | productos={N} | charolas={N_SHELVES} | "
      f"ancho total productos={ancho.sum():.0f}cm vs capacidad={N_SHELVES*W_SHELF:.0f}cm")

# ----------------------------------------------------------------------
# 2. Decodificador: permutacion -> asignacion a charolas (first-fit)
# ----------------------------------------------------------------------
def decode(perm):
    """Recorre productos en el orden de la permutacion y los va colocando en la
    charola actual mientras quepan (ancho<=55); si no, abre una nueva charola."""
    shelves, cur, used = [], [], 0.0
    for j in perm:
        if used + ancho[j] <= W_SHELF + 1e-9:
            cur.append(j); used += ancho[j]
        else:
            shelves.append(cur); cur = [j]; used = ancho[j]
    if cur: shelves.append(cur)
    return shelves

# ----------------------------------------------------------------------
# 3. Funcion objetivo (fitness)
# ----------------------------------------------------------------------
W1, W2, W3, PEN = 1.0, 1.5, 0.6, 50.0

def fitness(perm, detail=False):
    shelves = decode(perm)
    # (a) Aprovechamiento: ocupacion promedio de las charolas usadas
    util = np.mean([sum(ancho[j] for j in s)/W_SHELF for s in shelves])
    # (b) Incoherencia de tamano: desv. est. de ALTO dentro de cada charola (penaliza)
    incoh = np.mean([np.std([alto[j] for j in s]) if len(s)>1 else 0.0 for s in shelves])
    # (c) Agrupacion de marca: fraccion de vecinos contiguos con misma marca
    same, tot = 0, 0
    for s in shelves:
        for a, b in zip(s[:-1], s[1:]):
            tot += 1
            if marca_arr[a]==marca_arr[b]: same += 1
    block = same/tot if tot>0 else 0.0
    # (d) Penalizacion por usar mas charolas de las disponibles
    overflow = max(0, len(shelves)-N_SHELVES)
    f = W1*util - W2*(incoh/10.0) + W3*block - PEN*overflow
    if detail:
        return dict(fitness=f, util=util, incoh=incoh, block=block,
                    n_shelves=len(shelves), overflow=overflow, shelves=shelves)
    return f

# ----------------------------------------------------------------------
# 4. Operadores geneticos
# ----------------------------------------------------------------------
def ox_crossover(p1, p2):
    """Order Crossover (OX) - estandar para permutaciones."""
    n = len(p1); a, b = sorted(rng.choice(n, 2, replace=False))
    child = [-1]*n; child[a:b+1] = p1[a:b+1]
    fill = [g for g in p2 if g not in set(p1[a:b+1])]
    k = 0
    for i in list(range(b+1, n)) + list(range(0, a)):
        child[i] = fill[k]; k += 1
    return child

def mutate(perm, rate=0.2):
    p = perm[:]
    if rng.random() < rate:                 # swap
        i, j = rng.choice(len(p), 2, replace=False); p[i], p[j] = p[j], p[i]
    if rng.random() < rate:                 # insertion
        i, j = rng.choice(len(p), 2, replace=False); g = p.pop(i); p.insert(j, g)
    return p

def tournament(pop, fits, k=3):
    idx = rng.choice(len(pop), k, replace=False)
    return pop[idx[np.argmax([fits[i] for i in idx])]][:]

# ----------------------------------------------------------------------
# 5. Bucle del GA
# ----------------------------------------------------------------------
POP, GENS, ELITE = 150, 400, 8
# Inicializacion heuristica: la mitad de la poblacion se siembra ordenando los
# productos por ALTO (con ruido), lo que acerca productos de tamano similar y
# da ventaja inicial al criterio de coherencia. La otra mitad es aleatoria.
order_by_size = list(np.argsort(alto))
pop = []
for _ in range(POP//2):
    p = order_by_size[:]
    for _ in range(rng.integers(2, 8)):           # ruido: algunos swaps
        i, j = rng.choice(N, 2, replace=False); p[i], p[j] = p[j], p[i]
    pop.append(p)
pop += [list(rng.permutation(N)) for _ in range(POP - len(pop))]
best_hist, avg_hist = [], []
best, best_f = None, -1e9

for g in range(GENS):
    fits = [fitness(p) for p in pop]
    order = np.argsort(fits)[::-1]
    if fits[order[0]] > best_f:
        best_f = fits[order[0]]; best = pop[order[0]][:]
    best_hist.append(best_f); avg_hist.append(np.mean(fits))
    newpop = [pop[order[i]][:] for i in range(ELITE)]   # elitismo
    while len(newpop) < POP:
        c = ox_crossover(tournament(pop, fits), tournament(pop, fits))
        newpop.append(mutate(c))
    pop = newpop

# ----------------------------------------------------------------------
# 6. Comparacion: GA vs acomodo HISTORICO de OXXO
# ----------------------------------------------------------------------
# Reconstruir la permutacion historica (orden charola, ubicacion_bandeja)
hist_sorted = inst.sort_values(['CHAROLA','UBICACION_BANDEJA']).reset_index(drop=True)
item_to_idx = {row.ITEM: i for i, row in prods.iterrows()}
# Mapear de forma robusta por posicion
hist_perm = []
seen = {}
for _, r in hist_sorted.iterrows():
    cands = prods.index[(prods['ITEM']==r['ITEM'])].tolist()
    cands = [c for c in cands if c not in seen]
    if cands:
        hist_perm.append(cands[0]); seen[cands[0]] = True
for i in range(N):
    if i not in seen: hist_perm.append(i)

ga = fitness(best, detail=True)
hi = fitness(hist_perm, detail=True)

print("\n=== RESULTADOS PRELIMINARES ===")
print(f"{'Metrica':<28}{'Historico OXXO':>16}{'GA (Modelo 1)':>16}")
print(f"{'Fitness':<28}{hi['fitness']:>16.3f}{ga['fitness']:>16.3f}")
print(f"{'Aprovechamiento (%)':<28}{100*hi['util']:>16.1f}{100*ga['util']:>16.1f}")
print(f"{'Incoherencia tamano (cm)':<28}{hi['incoh']:>16.2f}{ga['incoh']:>16.2f}")
print(f"{'Agrupacion marca (%)':<28}{100*hi['block']:>16.1f}{100*ga['block']:>16.1f}")
print(f"{'Charolas usadas':<28}{hi['n_shelves']:>16d}{ga['n_shelves']:>16d}")

# ----------------------------------------------------------------------
# 7. Graficas
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

axes[0].plot(best_hist, label='Mejor individuo', lw=2.2, color='#1D9E75')
axes[0].axhline(hi['fitness'], ls='--', lw=1.4, color='#D85A30',
                label=f"Histórico OXXO ({hi['fitness']:.2f})")
axes[0].set_xlabel('Generación'); axes[0].set_ylabel('Fitness')
lo = min(best_hist); axes[0].set_ylim(lo-0.05, max(best_hist)+0.05)
axes[0].set_title('Convergencia del GA (mejor solución)')
axes[0].legend(loc='lower right'); axes[0].grid(alpha=0.25)

mets = ['Aprovecham.\n(%)','Coherencia tam.\n(menor=mejor, cm)','Agrupacion\nmarca (%)']
hv = [100*hi['util'], hi['incoh'], 100*hi['block']]
gv = [100*ga['util'], ga['incoh'], 100*ga['block']]
x = np.arange(3); wbar = 0.35
axes[1].bar(x-wbar/2, hv, wbar, label='Historico OXXO', color='#85B7EB')
axes[1].bar(x+wbar/2, gv, wbar, label='GA (Modelo 1)', color='#1D9E75')
axes[1].set_xticks(x); axes[1].set_xticklabels(mets, fontsize=9)
axes[1].set_title('GA vs acomodo historico'); axes[1].legend(); axes[1].grid(alpha=0.25, axis='y')

plt.tight_layout()
plt.savefig('/home/claude/resultados_ga.png', dpi=130, bbox_inches='tight')
print("\nGrafica guardada.")
