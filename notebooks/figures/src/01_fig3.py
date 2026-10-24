OUT="/leonardo/home/usertrain/a08trd25/aiwcas26-anemoi-tutorial/notebooks/figures/01_open_dataset_views.svg"
W,H=1180,500
BLUE="#2a78d6"; ORANGE="#eb6834"; AQUA="#1baf7a"; VIOLET="#6250d6"; INK="#0b0b0b"; INK2="#52514e"
s=[];a=s.append
a(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Inter, Segoe UI, Helvetica, Arial, sans-serif">')
a('<title>open_dataset builds a tree of lazy views</title>')
a(f'<defs><marker id="ah3" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{INK2}"/></marker>'
  f'<marker id="ahb" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{BLUE}"/></marker>'
  f'<marker id="aho" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{ORANGE}"/></marker></defs>')
a(f'<rect x="1" y="1" width="{W-2}" height="{H-2}" rx="14" fill="#ffffff" stroke="#e3e3e0"/>')
a(f'<text x="28" y="40" font-size="20" font-weight="700" fill="{INK}">open_dataset(...) builds a tree of lazy views</text>')
a(f'<text x="28" y="64" font-size="14" fill="{INK2}">The cutout example from this notebook. Building the tree copies no data; reading ds[i] walks it.</text>')
def box(x,y,w,h,col,title,sub,fill_op=0.10, mono_title=False):
    a(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{col}" fill-opacity="{fill_op}" stroke="{col}" stroke-width="1.8"/>')
    ff=' font-family="Menlo, Consolas, monospace"' if mono_title else ''
    a(f'<text x="{x+w/2}" y="{y+24}" font-size="15" font-weight="700" text-anchor="middle" fill="{INK}"{ff}>{title}</text>')
    for k,line in enumerate(sub):
        a(f'<text x="{x+w/2}" y="{y+44+k*16}" font-size="12" text-anchor="middle" fill="{INK2}" font-family="Menlo, Consolas, monospace">{line}</text>')
def arrow(x1,y1,x2,y2):
    a(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{INK2}" stroke-width="1.6" marker-end="url(#ah3)"/>')
bw,bh=140,74; gap=22
yU,yL=130,300
xs=[28+i*(bw+gap) for i in range(5)]
# upper chain (regional, 1 deg)
box(xs[0],yU,bw,bh,AQUA,"Zarr store",["keisler … 1p0d","1° lat/lon, 3-hourly"],0.14)
box(xs[1],yU,bw,bh,BLUE,"Subset",["frequency='6h'"])
box(xs[2],yU,bw,bh,VIOLET,"Select",["4 variables"])
box(xs[3],yU,bw,bh,VIOLET,"Rename",["temperature_850","→ t_850, …"])
box(xs[4],yU,bw,bh,ORANGE,"Cropping",["area=(72,-25,30,45)"])
for i in range(4): arrow(xs[i]+bw,yU+bh/2,xs[i+1]-3,yU+bh/2)
# lower chain (global, O48)
box(xs[0],yL,bw,bh,AQUA,"Zarr store",["… o48 …","~2° global, 6-hourly"],0.14)
box(xs[1],yL,bw,bh,BLUE,"Subset",["end='1979-01-01T06'"])
box(xs[2],yL,bw,bh,VIOLET,"Select",["same 4 variables"])
for i in range(2): arrow(xs[i]+bw,yL+bh/2,xs[i+1]-3,yL+bh/2)
# cutout
cx=xs[4]+bw+52; cy=(yU+yL)/2+bh/2; cw,ch=160,92
box(cx,cy-ch/2,cw,ch,INK,"Cutout",["regional points +","global points outside"],0.06)
a(f'<path d="M{xs[4]+bw},{yU+bh/2} C{cx-25},{yU+bh/2} {cx-40},{cy-20} {cx-3},{cy-20}" fill="none" stroke="{INK2}" stroke-width="1.6" marker-end="url(#ah3)"/>')
a(f'<path d="M{xs[2]+bw},{yL+bh/2} L{xs[4]+bw-10},{yL+bh/2} C{cx-25},{yL+bh/2} {cx-40},{cy+20} {cx-3},{cy+20}" fill="none" stroke="{INK2}" stroke-width="1.6" marker-end="url(#ah3)"/>')
# root
rx=cx+cw+40; rw=92
box(rx,cy-36,rw,72,INK,"ds",["what you get"],0.0, mono_title=True)
arrow(cx+cw,cy,rx-3,cy)
# request/response bands
ty=100
a(f'<line x1="{rx+rw}" y1="{ty}" x2="{xs[0]+40}" y2="{ty}" stroke="{BLUE}" stroke-width="1.6" stroke-dasharray="6 5" marker-end="url(#ahb)"/>')
a(f'<text x="{(rx+rw+xs[0])/2}" y="{ty-8}" font-size="13" text-anchor="middle" fill="{INK}">ds[i]: the request travels root → leaves: each view translates the index (date, variables, points)</text>')
by=410
a(f'<line x1="{xs[0]+40}" y1="{by}" x2="{rx+rw}" y2="{by}" stroke="{ORANGE}" stroke-width="1.6" stroke-dasharray="6 5" marker-end="url(#aho)"/>')
a(f'<text x="{(rx+rw+xs[0])/2}" y="{by+22}" font-size="13" text-anchor="middle" fill="{INK}">arrays flow leaves → root: only the chunks needed for date i are read, then sliced, stacked and merged</text>')
# legend
ly=H-28; lx=28
for col,lab in [(AQUA,"leaf: Zarr store"),(BLUE,"dates (start, end, frequency)"),(VIOLET,"variables (select, rename, …)"),(ORANGE,"grid points (area, thinning)"),(INK,"combine (join, concat, cutout, …)")]:
    a(f'<rect x="{lx}" y="{ly-11}" width="14" height="14" rx="3" fill="{col}" fill-opacity="0.18" stroke="{col}" stroke-width="1.5"/>')
    a(f'<text x="{lx+20}" y="{ly+1}" font-size="12.5" fill="{INK2}">{lab}</text>')
    lx+=20+len(lab)*6.6+30
a('</svg>')
open(OUT,"w").write("\n".join(s)); print("ok")
