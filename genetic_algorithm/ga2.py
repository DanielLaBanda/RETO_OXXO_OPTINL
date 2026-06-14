"""
========================================================================
 GA2 - MODELO EXTENDIDO (version REPORTE)
 = GA1 + termino de consistencia con E[ubicacion] historica
       + parametros extra TOMADOS RANDOM (placeholder de datos futuros)
========================================================================

ENTRADA (pipeline_ejemplo.py):
  oxxo_instance_<FORMATO>.csv, oxxo_weights_<FORMATO>.json,
  oxxo_locdist_<FORMATO>.json  (distribucion de charola por producto)

EXTENSIONES SOBRE GA1:
  1. Termino W4 * consistencia(asignacion, E[ubicacion] historica). El producto
     cuya charola asignada se acerca a su charola esperada historica suma.
  2. Parametros EXTRA RANDOM: un multiplicador de "importancia/demanda" por
     producto, sorteado al azar, que representa datos reales que aun no tenemos
     (margen, ventas). Marcado claramente como placeholder.

NO LINEALIDAD: termino de varianza (incoherencia) + naturaleza estocastica.
Frentes dados | espaciado 0.5 cm entre productos | direccion normalizada.
"""
import pandas as pd, numpy as np, json, os
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
rng = np.random.default_rng(7)

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
USE_PIPELINE = True
FORMATO      = 'BCO_CF_4.0_Refrescos'   # planograma; debe existir como artefacto del pipeline
W_SHELF      = 55.0
SEP          = 0.5
W4           = 0.4     # peso del termino de consistencia con E[ubicacion]

# ----------------------------------------------------------------------
# RUTAS (ancladas a la raiz del proyecto: funciona desde cualquier cwd)
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')      # entradas + artefactos del pipeline
OUT_DIR  = os.path.join(BASE_DIR, 'outputs')   # resultados del modelo (graficas)
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# CARGA
# ----------------------------------------------------------------------
if USE_PIPELINE and os.path.exists(f'{DATA_DIR}/oxxo_instance_{FORMATO}.csv'):
    prods = pd.read_csv(f'{DATA_DIR}/oxxo_instance_{FORMATO}.csv')
    with open(f'{DATA_DIR}/oxxo_weights_{FORMATO}.json') as f: wj = json.load(f)
    W1,W2,W3 = wj['w1'], wj['w2'], wj['w3']; W_SHELF = wj.get('W_shelf', W_SHELF)
    with open(f'{DATA_DIR}/oxxo_locdist_{FORMATO}.json') as f: locj = json.load(f)
    N_SHELVES_NOM = locj['n_shelves']
    E_loc = prods['E_CHAROLA_NORM'].values
    print(f"GA2 | {FORMATO} (pipeline) | pesos E[w]=({W1},{W2},{W3}) | +W4={W4}")
else:
    df = pd.read_csv(f'{DATA_DIR}/ejemplo_planograma.csv',
                     encoding='latin-1', sep=',', engine='python')
    df.columns=[c.replace('\ufeff','').replace('ï»¿','').replace('Ã\x91','Ñ').strip() for c in df.columns]
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
    N_SHELVES_NOM=int(inst['CHAROLA'].max())
    E_loc=(inst['CHAROLA'].values/N_SHELVES_NOM)
    W1,W2,W3=1.0,1.5,0.6
    print(f"GA2 | demo dataset chico | pesos fijos=({W1},{W2},{W3}) | +W4={W4}")

prods=prods.reset_index(drop=True)
ancho=(prods['ANCHO']*prods['NUM_FRENTES']).values
alto=prods['ALTO'].values; marca_arr=prods['MARCA'].values
N=len(prods)

# ----- PARAMETROS EXTRA RANDOM (placeholder de demanda/margen reales) -----
importancia = rng.uniform(0.8, 1.2, size=N)   # multiplicador random por producto

N_SHELVES=N_SHELVES_NOM   # charolas del planograma (dato del socio)
print(f"     productos={N} | charolas={N_SHELVES} | W={W_SHELF} | params extra: RANDOM")

# ----------------------------------------------------------------------
# DECODER + FITNESS EXTENDIDO
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
    # consistencia con E[ubicacion], ponderada por la importancia RANDOM
    loc=0.0
    for i,s in enumerate(sh):
        chn=(i+1)/len(sh)
        for j in s: loc += importancia[j]*(1-abs(chn-E_loc[j]))
    loc/=N
    over=max(0,len(sh)-N_SHELVES)
    f=W1*util-W2*(incoh/10.0)+W3*block+W4*loc-PEN*over
    if detail:
        waste=np.mean([1-used_w(s)/W_SHELF for s in sh])
        return dict(fitness=f,util=util,incoh=incoh,block=block,loc=loc,
                    n_shelves=len(sh),overflow=over,waste=waste,shelves=sh)
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
print("\n=== RESULTADOS GA2 ===")
print(f"Fitness={ga['fitness']:.3f} | Aprovech={100*ga['util']:.1f}% | "
      f"Desperdicio={100*ga['waste']:.1f}% | Incoher={ga['incoh']:.2f}cm | "
      f"Marca={100*ga['block']:.1f}% | Consist.E[ubic]={ga['loc']:.3f} | Charolas={ga['n_shelves']}")

fig,ax=plt.subplots(figsize=(7,4.4))
ax.plot(best_hist,lw=2.2,color='#7C5CBF')
ax.set_xlabel('Generacion'); ax.set_ylabel('Fitness'); ax.set_title(f'GA2 (extendido) - {FORMATO}')
ax.grid(alpha=0.25); plt.tight_layout()
out_png = os.path.join(OUT_DIR, 'resultados_ga2.png')
plt.savefig(out_png, dpi=130, bbox_inches='tight')
print(f"Grafica: {out_png}")
