"""Fırça darbeli boyama (Hertzmann 1998, "Painterly Rendering with Curved Brush Strokes").
Referans görüntüyü (ör. CycleGAN'ın Monet renkli çıktısı) büyük fırçadan küçüğe katman katman,
gerçek fırça darbeleriyle yeniden boyar. Darbeler görüntüdeki kenarlara dik yönde kıvrılır."""
import numpy as np, random, sys
from PIL import Image, ImageDraw, ImageFilter

def paint(ref, radii=(12,6,3), T=18, max_len=10, min_len=2, fc=0.7, seed=0):
    random.seed(seed); np.random.seed(seed)
    W,H=ref.size
    canvas=Image.new("RGB",(W,H),(245,240,230))
    for R in radii:
        rb=ref.filter(ImageFilter.GaussianBlur(R*0.5))
        A=np.asarray(rb).astype(np.float32)
        L=0.3*A[...,0]+0.59*A[...,1]+0.11*A[...,2]
        gy,gx=np.gradient(L)
        C=np.asarray(canvas).astype(np.float32)
        D=np.sqrt(((C-A)**2).sum(-1))
        strokes=[]
        g=max(1,int(R*0.9))
        for y in range(0,H,g):
            for x in range(0,W,g):
                win=D[y:y+g, x:x+g]
                if win.mean()>T or R==radii[0]:
                    iy,ix=np.unravel_index(win.argmax(),win.shape)
                    strokes.append((x+ix,y+iy))
        random.shuffle(strokes)
        draw=ImageDraw.Draw(canvas)
        Cn=np.asarray(canvas).astype(np.float32)
        for (x0,y0) in strokes:
            col=tuple(int(v) for v in A[y0,x0])
            pts=[(x0,y0)]; x,y=float(x0),float(y0); ldx,ldy=0.0,0.0
            for i in range(max_len):
                xi,yi=int(x),int(y)
                if i>min_len and np.abs(A[yi,xi]-Cn[yi,xi]).sum() < np.abs(A[yi,xi]-np.array(col)).sum(): break
                gxv,gyv=gx[yi,xi],gy[yi,xi]
                if gxv*gxv+gyv*gyv<1e-2: break
                dx,dy=-gyv,gxv          # gradyana dik yön = kenar boyunca
                if ldx*dx+ldy*dy<0: dx,dy=-dx,-dy
                dx,dy=fc*dx+(1-fc)*ldx, fc*dy+(1-fc)*ldy
                n=(dx*dx+dy*dy)**0.5+1e-6; dx,dy=dx/n,dy/n
                x+=R*dx; y+=R*dy; ldx,ldy=dx,dy
                if not(0<=x<W and 0<=y<H): break
                pts.append((x,y))
            # hafif renk oynaması: elle boyanmış hissi
            j=np.random.randint(-8,9,3)
            c=tuple(int(np.clip(col[k]+j[k],0,255)) for k in range(3))
            if len(pts)>1: draw.line(pts,fill=c,width=R*2,joint="curve")
            for p in (pts[0],pts[-1]): draw.ellipse([p[0]-R,p[1]-R,p[0]+R,p[1]+R],fill=c)
    return canvas

if __name__=="__main__":
    for name in sys.argv[1:]:
        ref=Image.open(name).convert("RGB")
        out=paint(ref)
        out.save(name.replace(".jpg","_FIRCA.jpg"),quality=94)
        print("ok",name)
