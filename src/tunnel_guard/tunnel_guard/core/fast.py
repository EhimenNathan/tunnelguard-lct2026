"""Allocation-light point kernels. IEEE arithmetic; no fast-math or point removal."""
import numpy as np
try:
    from numba import njit
    HAVE_FAST = True
except Exception:  # the numpy reference stays available if Numba cannot load
    HAVE_FAST = False
    def njit(*args, **kwargs):
        return lambda f: f


@njit(cache=True)
def track_coordinates(pts, yc, zr, head, step, roll):
    n=len(pts);l=np.empty(n,np.float64);h=np.empty(n,np.float64)
    for j in range(n):
        u=min(max(float(pts[j,0])/step,0.),len(yc)-1.000001)
        i=int(u);f=u-i;g=1.-f
        y=yc[i]*g+yc[i+1]*f;z=zr[i]*g+zr[i+1]*f
        hd=head[i]*g+head[i+1]*f
        l[j]=(pts[j,1]-y)/np.sqrt(1.+hd*hd)
        h[j]=pts[j,2]-z-roll*l[j]
    return l,h


@njit(cache=True)
def scatter_sorted(order, keys, ranges, encoded, depth, maximum):
    """Exact linear equivalent of the two sorted searchsorted calls."""
    n=len(order);flag=np.zeros(n,np.bool_);up=0;dn=0
    for i in range(n):
        while up<n and encoded[up]<encoded[i]+depth:up+=1
        while dn<n and encoded[dn]<=encoded[i]-depth:dn+=1
        deeper=up<n and keys[up]==keys[i] and ranges[up]-ranges[i]<maximum
        shallower=dn>0 and keys[dn-1]==keys[i] and ranges[i]-ranges[dn-1]<maximum
        flag[order[i]]=deeper or shallower
    return flag


@njit(cache=True)
def prepare_forward(xyz, valid, columns, signs, level, near_x, decimate, tangent):
    """Fuse signed-axis mapping, levelling, FOV and existing decimation.

    Preserves the original order and near-field decimation rank; far points
    are never sampled or removed beyond the existing geometric FOV.
    """
    n=len(xyz);out=np.empty((n,3),np.float32);selected=np.empty(n,np.int64)
    rank=-1;count=0
    for j in range(n):
        if not valid[j]:continue
        rank+=1
        a=xyz[j,columns[0]]*signs[0];b=xyz[j,columns[1]]*signs[1];c=xyz[j,columns[2]]*signs[2]
        x=level[0,0]*a+level[0,1]*b+level[0,2]*c
        y=level[1,0]*a+level[1,1]*b+level[1,2]*c
        z=level[2,0]*a+level[2,1]*b+level[2,2]*c
        if x<=1. or abs(y)>=tangent*x+3. or abs(z)>=12.:continue
        if decimate>1 and x<near_x and rank%decimate!=0:continue
        out[count,0]=x;out[count,1]=y;out[count,2]=z;selected[count]=j;count+=1
    return out[:count],selected[:count]


def warmup():
    if HAVE_FAST:
        track_coordinates(np.zeros((2,3),np.float32),np.zeros(3),np.zeros(3),np.zeros(3),1.,0.)
        track_coordinates(np.zeros((2,3),np.float64),np.zeros(3),np.zeros(3),np.zeros(3),1.,0.)
        scatter_sorted(np.arange(2),np.zeros(2,np.int64),np.zeros(2),np.zeros(2),.5,5.)
        scatter_sorted(np.arange(2),np.zeros(2,np.int64),np.zeros(2,np.float32),np.zeros(2),.5,5.)
        prepare_forward(np.zeros((2,3),np.float32),np.ones(2,np.bool_),np.arange(3),np.ones(3,np.float32),np.eye(3,dtype=np.float32),25.,3,2.7)
