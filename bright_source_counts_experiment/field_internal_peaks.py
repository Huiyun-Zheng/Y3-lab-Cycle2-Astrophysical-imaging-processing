"""Image-wide diagnostics for internal multiple peaks in fixed apertures.

Pixel S/N and saddle depths are descriptive statistics, not false-alarm
probabilities. A multi-peaked galaxy can be real; a flag does not count it twice.
"""
import numpy as np
from astropy.table import Table
from scipy import ndimage as ndi
from aperture_neighbour_model import aperture_grid


def scan_internal_peaks(table, res, rms, labels):
    rows=[]
    for i,r in enumerate(table):
     bounds,xx,yy,ap,edge=aperture_grid(r,res.shape)
     # Pad for smoothing so aperture edges do not create spurious maxima.
     x0=max(0,bounds[1].start-5);x1=min(res.shape[1],bounds[1].stop+5);y0=max(0,bounds[0].start-5);y1=min(res.shape[0],bounds[0].stop+5)
     yy,xx=np.mgrid[y0:y1,x0:x1];data=res[y0:y1,x0:x1].astype(float);noise=rms[y0:y1,x0:x1]
     theta=np.radians(r['aperture_theta_deg']);dx=xx-r['x'];dy=yy-r['y'];ap=((dx*np.cos(theta)+dy*np.sin(theta))/r['aperture_a_pix'])**2+((-dx*np.sin(theta)+dy*np.cos(theta))/r['aperture_b_pix'])**2<=1
     own=labels[y0:y1,x0:x1]==r['segment_id'];mask=ap&own
     surfaces=[ndi.gaussian_filter(data,s) for s in [.5,.7]];positions=[]
     for surf in surfaces:
      peaks=(surf==ndi.maximum_filter(surf,3))&(surf>5*noise)&mask
      groups,n=ndi.label(peaks);pos=ndi.maximum_position(surf,groups,np.arange(1,n+1)) if n else []
      positions.append(sorted(pos,key=lambda z:surf[z],reverse=True))
     if len(positions[0])<2 or len(positions[1])<2:continue
     primary=positions[0][0];sec=[]
     for secondary in positions[0][1:]:
      separation=np.linalg.norm(np.subtract(secondary,primary))
      if separation<3:continue
      if not any(np.linalg.norm(np.subtract(secondary,p))<=1.5 for p in positions[1][1:]):continue
      surf=surfaces[0];lo=0.;hi=float(surf[secondary]);seed=np.zeros(data.shape,bool);seed[secondary]=True
      for _ in range(20):
       mid=(lo+hi)/2;connected=ndi.binary_propagation(seed,structure=np.ones((3,3)),mask=own&(surf>=mid))
       if connected[primary]:lo=mid
       else:hi=mid
      dip=(surf[secondary]-lo)/noise[secondary]
      small=mask&(np.hypot(xx-xx[secondary],yy-yy[secondary])<=2.)
      fraction=np.maximum(data[small],0).sum()/np.maximum(data[ap],0).sum()
      sec.append((dip,secondary,separation,fraction))
     if not sec:continue
     dip,secondary,sep,frac=max(sec,key=lambda z:z[0]);surf=surfaces[0]
     rows.append((int(r['id']),i,bool(r['usable']),float(r['x']),float(r['y']),xx[primary],yy[primary],xx[secondary],yy[secondary],sep,float(surf[secondary]/noise[secondary]),float(surf[secondary]/surf[primary]),float(dip),float(frac)))
    a=Table(rows=rows,names=['id','row_index','usable','x','y','primary_x','primary_y','secondary_x','secondary_y','separation','secondary_snr','peak_ratio','dip_snr','secondary_core_fraction'])
    return a
