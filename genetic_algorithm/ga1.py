"""
========================================================================
 GA1 - MODELO NO EXTENDIDO (version SOCIO FORMADOR)
 Lee los artefactos del pipeline (instancia + pesos por valor esperado).
========================================================================

ENTRADA (generada por pipeline_ejemplo.py):
  oxxo_instance_<FORMATO>.csv : productos con dims, frentes dados, marca, E[charola]
  oxxo_weights_<FORMATO>.json : pesos w1,w2,w3 (valor esperado de la simulacion)

REGLAS: frentes dados (no se optimizan) | espaciado 0.5 cm ENTRE productos |
  W=55 cuarto frio (param.) | direccion ya normalizada por el pipeline.

Si no hay artefactos del pipeline, USE_PIPELINE=False usa el dataset chico
'ejemplo_planograma.csv' como demostracion con pesos fijos.
"""
import pandas as pd, numpy as np, json, os
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
rng = np.random.default_rng(42)

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
USE_PIPELINE = True
FORMATO      = os.environ.get('OXXO_FORMATO', 'BCO_CF_4.0_Refrescos')   # planograma (override por env OXXO_FORMATO)
W_SHELF      = 55.0
SEP          = 0.5

# ----------------------------------------------------------------------
# RUTAS (ancladas a la raiz del proyecto: funciona desde cualquier cwd)
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')      # entradas + artefactos del pipeline
OUT_DIR  = os.path.join(BASE_DIR, 'outputs')   # resultados del modelo (graficas)
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# CARGA DE LA INSTANCIA
# ----------------------------------------------------------------------

if USE_PIPELINE and os.path.exists(f'{DATA_DIR}/oxxo_instance_{FORMATO}.csv'):
    prods = pd.read_csv(f'{DATA_DIR}/oxxo_instance_{FORMATO}.csv')
    with open(f'{DATA_DIR}/oxxo_weights_{FORMATO}.json') as f: wj = json.load(f)
    W1, W2, W3 = wj['w1'], wj['w2'], wj['w3']
    W_SHELF = wj.get('W_shelf', W_SHELF)
    print(f"GA1 | {FORMATO} (pipeline) | pesos E[w]=({W1},{W2},{W3})")
else:
    # ---- modo demostracion con dataset chico ----
    df = pd.read_csv(f'{DATA_DIR}/ejemplo_planograma.csv',
                     encoding='latin-1', sep=',', engine='python')
    df.columns = [c.replace('\ufeff','').replace('ï»¿','').replace('Ã\x91','Ñ').strip() for c in df.columns]
    def mk(d):
        d=str(d).upper()
        for m in ['COCA','PEPSI','FRESCA','SQUIRT','PENAFIEL','TOPO CHICO','CIEL','SPRITE',
                  'FANTA','JOYA','BARRILITOS','MONSTER','DR PEPPER','MIRINDA','SCHWEPPES','MUNDET']:
            if m in d: return m
        return 'OTRO'
    df['MARCA']=df['ITEM_DESC'].apply(mk)
    inst=df[(df['SEGMENTO_ID']=='HRN')&(df['TAMAÑO_POST']==3.0)&
            (df['DIRECCION_LEGO_ID']=='DI')&(df['CONJUNTO_ID']=='10MON')].copy()
    inst=inst.sort_values(['CHAROLA','UBICACION_BANDEJA']).reset_index(drop=True)
    prods=inst[['ITEM','ITEM_DESC','ANCHO','ALTO','NUM_FRENTES','MARCA']].rename(columns={'ITEM':'UPC'})
    W1,W2,W3=1.0,1.5,0.6
    print(f"GA1 | demo dataset chico | pesos fijos=({W1},{W2},{W3})")

# vectores de trabajo
prods = prods.reset_index(drop=True)
ancho = (prods['ANCHO']*prods['NUM_FRENTES']).values
alto  = prods['ALTO'].values
marca_arr = prods['MARCA'].values
N = len(prods)

# charolas del planograma (dato del socio: no se inventa holgura)
if USE_PIPELINE and os.path.exists(f'{DATA_DIR}/oxxo_locdist_{FORMATO}.json'):
    with open(f'{DATA_DIR}/oxxo_locdist_{FORMATO}.json') as f: N_SHELVES_NOM = json.load(f)['n_shelves']
else:
    N_SHELVES_NOM = 18
N_SHELVES = N_SHELVES_NOM
print(f"     productos={N} | charolas={N_SHELVES} | W={W_SHELF}")

# ----------------------------------------------------------------------
# DECODER (espaciado entre productos) + FITNESS
# ----------------------------------------------------------------------
def decode(perm):
    sh,cur,used=[],[],0.0
    for j in perm:
        extra=(SEP if cur else 0.0)
        if used+extra+ancho[j]<=W_SHELF+1e-9: cur.append(j); used+=extra+ancho[j]
        else: sh.append(cur); cur=[j]; used=ancho[j]
    if cur: sh.append(cur)
    return sh
def used_w(s): return sum(ancho[j] for j in s)+SEP*max(0,len(s)-1)

PEN=50.0
def fitness(perm, detail=False):
    sh=decode(perm)
    util=np.mean([used_w(s)/W_SHELF for s in sh])
    incoh=np.mean([np.std([alto[j] for j in s]) if len(s)>1 else 0.0 for s in sh])
    same,tot=0,0
    for s in sh:
        for a,b in zip(s[:-1],s[1:]):
            tot+=1
            if marca_arr[a]==marca_arr[b]: same+=1
    block=same/tot if tot>0 else 0.0
    over=max(0,len(sh)-N_SHELVES)
    f=W1*util-W2*(incoh/10.0)+W3*block-PEN*over
    if detail:
        waste=np.mean([1-used_w(s)/W_SHELF for s in sh])
        return dict(fitness=f,util=util,incoh=incoh,block=block,n_shelves=len(sh),
                    overflow=over,waste=waste,shelves=sh)
    return f

# ----------------------------------------------------------------------
# OPERADORES + GA
# ----------------------------------------------------------------------
def ox(p1,p2):
    n=len(p1); a,b=sorted(rng.choice(n,2,replace=False))
    c=[-1]*n; c[a:b+1]=p1[a:b+1]; fill=[g for g in p2 if g not in set(p1[a:b+1])]; k=0
    for i in list(range(b+1,n))+list(range(0,a)): c[i]=fill[k]; k+=1
    return c
def mut(p,r=0.2):
    p=p[:]
    if rng.random()<r: i,j=rng.choice(len(p),2,replace=False); p[i],p[j]=p[j],p[i]
    if rng.random()<r: i,j=rng.choice(len(p),2,replace=False); g=p.pop(i); p.insert(j,g)
    return p
def tourn(pop,fits,k=3):
    idx=rng.choice(len(pop),k,replace=False); return pop[idx[np.argmax([fits[i] for i in idx])]][:]

POP,GENS,ELITE=150,400,8
obs=list(np.argsort(alto)); pop=[]
for _ in range(POP//2):
    p=obs[:]
    for _ in range(rng.integers(2,8)): i,j=rng.choice(N,2,replace=False); p[i],p[j]=p[j],p[i]
    pop.append(p)
pop+=[list(rng.permutation(N)) for _ in range(POP-len(pop))]

best_hist=[]; best=None; bf=-1e9
for g in range(GENS):
    fits=[fitness(p) for p in pop]; order=np.argsort(fits)[::-1]
    if fits[order[0]]>bf: bf=fits[order[0]]; best=pop[order[0]][:]
    best_hist.append(bf)
    npop=[pop[order[i]][:] for i in range(ELITE)]
    while len(npop)<POP: npop.append(mut(ox(tourn(pop,fits),tourn(pop,fits))))
    pop=npop

# ----------------------------------------------------------------------
# RESULTADOS
# ----------------------------------------------------------------------
ga=fitness(best,detail=True)
print("\n=== RESULTADOS GA1 ===")
print(f"Fitness={ga['fitness']:.3f} | Aprovech={100*ga['util']:.1f}% | "
      f"Desperdicio={100*ga['waste']:.1f}% | Incoher={ga['incoh']:.2f}cm | "
      f"Marca={100*ga['block']:.1f}% | Charolas={ga['n_shelves']}")

fig,ax=plt.subplots(figsize=(7,4.4))
ax.plot(best_hist,lw=2.2,color='#1D9E75')
ax.set_xlabel('Generacion'); ax.set_ylabel('Fitness'); ax.set_title(f'GA1 - {FORMATO}')
ax.grid(alpha=0.25); plt.tight_layout()
out_png = os.path.join(OUT_DIR, 'resultados_ga1.png')
plt.savefig(out_png, dpi=130, bbox_inches='tight')
print(f"Grafica: {out_png}")
