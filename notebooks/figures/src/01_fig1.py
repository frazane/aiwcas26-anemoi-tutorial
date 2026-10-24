# Generates notebooks/figures/01_data_array_anatomy.svg (hand-authored geometry, emitted by script for exact coordinates)
OUT="/leonardo/home/usertrain/a08trd25/aiwcas26-anemoi-tutorial/notebooks/figures/01_data_array_anatomy.svg"
W,H=1060,600
BLUE="#2a78d6"; BLUE_L="#dbe8f8"; INK="#0b0b0b"; INK2="#52514e"; MUTED="#8a8984"; GRAY_F="#f1f2f4"; GRAY_S="#a7adb5"
ORANGE="#eb6834"; AQUA="#1baf7a"; VIOLET="#6250d6"
s=[]
a=s.append
a(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Inter, Segoe UI, Helvetica, Arial, sans-serif">')
a('<title>Anatomy of an anemoi dataset: the 4-D data array and its chunking</title>')
a('<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="%s"/></marker></defs>'%INK2)
a(f'<rect x="1" y="1" width="{W-2}" height="{H-2}" rx="14" fill="#ffffff" stroke="#e3e3e0"/>')
a(f'<text x="28" y="40" font-size="20" font-weight="700" fill="{INK}">The data array: shape (dates, variables, ensembles, grid points)</text>')
a(f'<text x="28" y="64" font-size="14" fill="{INK2}">Example: ERA5 O96 test set, shape (39, 101, 1, 40 320), Zarr chunks (1, 101, 1, 40 320)</text>')
fw,fh=330,210   # front face
t,t2=12,9       # slab thickness (ensemble axis)
dx,dy=36,-27    # step between dates
X0,Y0=160,270
N=6
for k in reversed(range(N)):
    x0=X0+k*dx; y0=Y0+k*dy
    front=(k==0)
    fill=BLUE_L if front else GRAY_F
    stroke=BLUE if front else GRAY_S
    sw=2.2 if front else 1.2
    top=f'{x0},{y0} {x0+t},{y0-t2} {x0+fw+t},{y0-t2} {x0+fw},{y0}'
    right=f'{x0+fw},{y0} {x0+fw+t},{y0-t2} {x0+fw+t},{y0+fh-t2} {x0+fw},{y0+fh}'
    a(f'<polygon points="{top}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" stroke-linejoin="round"/>')
    a(f'<polygon points="{right}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" stroke-linejoin="round"/>')
    a(f'<rect x="{x0}" y="{y0}" width="{fw}" height="{fh}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
    if front:
        # variable rows
        rows=["10u","10v","2d","2t","…","msl","…","t_850","…","z_925"]
        rh=fh/len(rows)
        for i,r in enumerate(rows):
            y=y0+i*rh
            if i: a(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+fw}" y2="{y:.1f}" stroke="{BLUE}" stroke-opacity="0.28" stroke-width="1"/>')
            a(f'<text x="{x0-8}" y="{y+rh/2+4.5:.1f}" font-size="12.5" text-anchor="end" fill="{INK2}" font-family="Menlo, Consolas, monospace">{r}</text>')
        # highlight one row = one field
        y=y0+3*rh
        a(f'<rect x="{x0+1.5}" y="{y+1:.1f}" width="{fw-3}" height="{rh-2:.1f}" fill="{BLUE}" fill-opacity="0.22"/>')
        # grid-point ticks along the highlighted row
        for j in range(0,fw-8,11):
            a(f'<circle cx="{x0+7+j}" cy="{y+rh/2:.1f}" r="2.1" fill="{BLUE}"/>')
        # chunk label (on a white pill for legibility)
        cx=x0+fw/2; cy=y0+fh*0.72
        a(f'<rect x="{cx-156}" y="{cy-30}" width="312" height="58" rx="8" fill="#ffffff" fill-opacity="0.92" stroke="{BLUE}" stroke-width="1.2"/>')
        a(f'<text x="{cx}" y="{cy-8}" font-size="15" font-weight="700" text-anchor="middle" fill="{INK}">ds[i]  =  one chunk  =  one sample</text>')
        a(f'<text x="{cx}" y="{cy+13}" font-size="12.5" text-anchor="middle" fill="{INK2}">date i · all variables · 1 member · all points</text>')
    else:
        pass
# date labels on two back slabs
a(f'<text x="{X0+fw+t+6}" y="{Y0-t2+4}" font-size="12" fill="{BLUE}" font-weight="600">date i</text>')
kb=N-1
a(f'<text x="{X0+kb*dx+t}" y="{Y0+kb*dy-t2-8}" font-size="12" fill="{MUTED}">date i+5 …</text>')
# axes arrows
# variables (vertical, left)
ax=X0-62
a(f'<line x1="{ax}" y1="{Y0+fh}" x2="{ax}" y2="{Y0+4}" stroke="{INK2}" stroke-width="1.6" marker-end="url(#ah)"/>')
a(f'<text transform="translate({ax-10},{Y0+fh/2}) rotate(-90)" font-size="14" text-anchor="middle" fill="{INK}" font-weight="600">variables (101)</text>')
# grid points (horizontal, bottom)
by=Y0+fh+24
a(f'<line x1="{X0}" y1="{by}" x2="{X0+fw}" y2="{by}" stroke="{INK2}" stroke-width="1.6" marker-end="url(#ah)"/>')
a(f'<text x="{X0+fw/2}" y="{by+22}" font-size="14" text-anchor="middle" fill="{INK}" font-weight="600">grid points (40 320): one flat 1-D axis</text>')
a(f'<text x="{X0+fw/2}" y="{by+41}" font-size="12.5" text-anchor="middle" fill="{INK2}">position of point g: latitudes[g], longitudes[g]</text>')
# dates (diagonal)
sx,sy=X0+fw+t+18, Y0+fh-t2+8
ex,ey=sx+(N-1)*dx+10, sy+(N-1)*dy-8
a(f'<line x1="{sx}" y1="{sy}" x2="{ex}" y2="{ey}" stroke="{INK2}" stroke-width="1.6" marker-end="url(#ah)"/>')
import math
ang=math.degrees(math.atan2(ey-sy,ex-sx))
mx,my=(sx+ex)/2+14,(sy+ey)/2+12
a(f'<text transform="translate({mx:.1f},{my:.1f}) rotate({ang:.1f})" font-size="14" text-anchor="middle" fill="{INK}" font-weight="600">dates (39)</text>')
# ensemble (thickness of each slab)
ex0,ey0=X0+t*0.5+2,Y0-t2*0.5
a(f'<line x1="{X0-40}" y1="{Y0-46}" x2="{ex0}" y2="{ey0-2}" stroke="{INK2}" stroke-width="1.3" marker-end="url(#ah)"/>')
a(f'<text x="{X0-44}" y="{Y0-66}" font-size="14" font-weight="600" text-anchor="middle" fill="{INK}">ensemble</text>')
a(f'<text x="{X0-44}" y="{Y0-50}" font-size="14" font-weight="600" text-anchor="middle" fill="{INK}">members (1)</text>')
# right panel: other arrays
px=740; py=100
a(f'<text x="{px}" y="{py}" font-size="15" font-weight="700" fill="{INK}">Other arrays in the same store</text>')
items=[
 (AQUA,"dates","(39,)","datetime64, one per sample"),
 (AQUA,"latitudes, longitudes","(40 320,)","flat coordinates of each point"),
 (ORANGE,"mean, stdev, minimum, maximum","(101,)","one value per variable"),
 (ORANGE,"sums, squares, count, has_nans","(101,)","raw accumulators"),
 (ORANGE,"statistics_tendencies_6h_*","(101,)","stats of x(t+6h) − x(t), also 12h, 1d"),
 (VIOLET,".zattrs (metadata)","JSON","variables, recipe, provenance, …"),
]
y=py+18
for col,name,shape,desc in items:
    a(f'<rect x="{px}" y="{y}" width="292" height="52" rx="7" fill="#ffffff" stroke="{col}" stroke-width="1.5"/>')
    a(f'<rect x="{px}" y="{y}" width="6" height="52" rx="3" fill="{col}"/>')
    a(f'<text x="{px+16}" y="{y+21}" font-size="13" font-weight="600" fill="{INK}" font-family="Menlo, Consolas, monospace">{name}</text>')
    a(f'<text x="{px+16}" y="{y+40}" font-size="12" fill="{INK2}">{shape}  {desc}</text>')
    y+=62
# legend for colours
ly=H-24
a(f'<text x="28" y="{ly}" font-size="12.5" fill="{INK2}">Default chunking when a dataset is created: output.chunking = {{dates: 1, ensembles: 1}}. Very large grids are split into ≤ 4 grid chunks (each ≥ 32 MiB).</text>')
a('</svg>')
open(OUT,"w").write("\n".join(s))
print("written")
