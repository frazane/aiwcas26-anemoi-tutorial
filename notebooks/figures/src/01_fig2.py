import math, random
OUT="/leonardo/home/usertrain/a08trd25/aiwcas26-anemoi-tutorial/notebooks/figures/01_flat_grid.svg"
W,H=1060,400
BLUE="#2a78d6"; ORANGE="#eb6834"; AQUA="#1baf7a"; VIOLET="#6250d6"; INK="#0b0b0b"; INK2="#52514e"; GRAY="#a7adb5"
s=[];a=s.append
a(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Inter, Segoe UI, Helvetica, Arial, sans-serif">')
a('<title>Any grid becomes a flat list of points</title>')
a(f'<defs><marker id="ah2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{INK2}"/></marker></defs>')
a(f'<rect x="1" y="1" width="{W-2}" height="{H-2}" rx="14" fill="#ffffff" stroke="#e3e3e0"/>')
a(f'<text x="28" y="40" font-size="20" font-weight="700" fill="{INK}">Any grid becomes one flat axis of points</text>')
a(f'<text x="28" y="64" font-size="14" fill="{INK2}">The grid structure is not stored in the array layout, only the coordinates of each point.</text>')
# three source panels
px=28; pw=200; ph=84
panels=[("Regular lat/lon (2-D)",BLUE),("Reduced Gaussian (O96)",ORANGE),("Regional / stations",AQUA)]
for i,(name,col) in enumerate(panels):
    y=88+i*100
    a(f'<rect x="{px}" y="{y}" width="{pw}" height="{ph}" rx="8" fill="#ffffff" stroke="{col}" stroke-width="1.5"/>')
    a(f'<text x="{px+10}" y="{y+18}" font-size="12.5" font-weight="600" fill="{INK}">{name}</text>')
    if i==0:
        for r in range(4):
            for c in range(12):
                a(f'<circle cx="{px+16+c*15.5}" cy="{y+32+r*13}" r="2.6" fill="{col}"/>')
    elif i==1:
        rows=[4,8,12,12,8,4]
        for r,n in enumerate([5,9,13]):
            for c in range(n):
                xx=px+pw/2+(c-(n-1)/2)*13.5
                a(f'<circle cx="{xx:.1f}" cy="{y+32+r*16}" r="2.6" fill="{col}"/>')
    else:
        random.seed(3)
        a(f'<path d="M{px+30},{y+70} C{px+60},{y+28} {px+120},{y+30} {px+170},{y+44} L{px+176},{y+74} Z" fill="{col}" fill-opacity="0.12" stroke="none"/>')
        for k in range(26):
            xx=px+18+random.random()*168; yy=y+28+random.random()*48
            a(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="2.6" fill="{col}"/>')
# arrows to strip
sx=250
for i in range(3):
    y=88+i*100+ph/2
    a(f'<path d="M{sx},{y} C{sx+40},{y} {sx+40},{200} {sx+78},{200}" fill="none" stroke="{INK2}" stroke-width="1.5" marker-end="url(#ah2)"/>')
a(f'<text x="{sx+42}" y="{120}" font-size="12.5" fill="{INK2}" font-style="italic">flatten</text>')
# strip of cells
x0=332; cw=17; n=20; y0=186
cols=[BLUE]*7+[ORANGE]*9+[AQUA]*6
labels_y=y0-34
a(f'<text x="{x0}" y="{labels_y}" font-size="14" font-weight="600" fill="{INK}">data[date, variable, member, :]</text>')
for k in range(n):
    c=cols[k] if False else BLUE
    a(f'<rect x="{x0+k*cw}" y="{y0}" width="{cw-2}" height="28" rx="3" fill="{BLUE}" fill-opacity="{0.25+0.6*abs(math.sin(k*0.7))}"/>')
a(f'<text x="{x0+n*cw+6}" y="{y0+19}" font-size="14" fill="{INK2}">…</text>')
for k,lab in [(0,"g = 0"),(n-1,"N−1")]:
    a(f'<text x="{x0+k*cw+cw/2}" y="{y0+46}" font-size="11.5" text-anchor="middle" fill="{INK2}">{lab}</text>')
# lat/lon arrays
for j,(name) in enumerate(["latitudes[:]","longitudes[:]"]):
    yy=y0+70+j*42
    a(f'<text x="{x0}" y="{yy-6}" font-size="12.5" fill="{INK2}" font-family="Menlo, Consolas, monospace">{name}</text>')
    for k in range(n):
        a(f'<rect x="{x0+k*cw}" y="{yy}" width="{cw-2}" height="16" rx="3" fill="{AQUA if j==0 else AQUA}" fill-opacity="0.35"/>')
a(f'<text x="{x0}" y="{y0+178}" font-size="12.5" fill="{INK2}">one value per point; same layout for every variable and date</text>')
# arrow to graph
gx=790
a(f'<line x1="{x0+n*cw+30}" y1="{y0+14}" x2="{gx-12}" y2="{y0+14}" stroke="{INK2}" stroke-width="1.5" marker-end="url(#ah2)"/>')
a(f'<text x="{(x0+n*cw+30+gx)/2-6}" y="{y0-4}" font-size="12.5" text-anchor="middle" fill="{INK2}" font-style="italic">points = nodes</text>')
# graph
random.seed(5)
pts=[]
for r in range(4):
    for c in range(4):
        pts.append((gx+30+c*62+random.uniform(-14,14)+(r%2)*18, 104+r*58+random.uniform(-12,12)))
def knn(i,k=3):
    p=pts[i]; return sorted(range(len(pts)), key=lambda j:(pts[j][0]-p[0])**2+(pts[j][1]-p[1])**2)[1:k+1]
E=set()
for i in range(len(pts)):
    for j in knn(i): E.add((min(i,j),max(i,j)))
for i,j in sorted(E):
    a(f'<line x1="{pts[i][0]:.1f}" y1="{pts[i][1]:.1f}" x2="{pts[j][0]:.1f}" y2="{pts[j][1]:.1f}" stroke="{GRAY}" stroke-width="1.3"/>')
for p in pts:
    a(f'<circle cx="{p[0]:.1f}" cy="{p[1]:.1f}" r="5" fill="{BLUE}" stroke="#ffffff" stroke-width="2"/>')
a(f'<text x="{gx+118}" y="{H-58}" font-size="13" font-weight="600" text-anchor="middle" fill="{INK}">Graph neural network input</text>')
a(f'<text x="{gx+118}" y="{H-40}" font-size="12" text-anchor="middle" fill="{INK2}">edges built from lat/lon by anemoi-graphs</text>')
a('</svg>')
open(OUT,"w").write("\n".join(s)); print("ok")
