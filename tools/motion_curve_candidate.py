"""Opt-in motion/curve research candidate. Frozen production defaults are unchanged.

SE(2)+grade transport from the previous measured near-track tangent; no ground
truth pose enters this estimator. Temporal evidence is reprojected scan by scan,
never counted twice. Evaluate before promoting this candidate to production.
"""
import numpy as np
from tunnel_guard.core.geometry import GeometryEstimator,TrackGeometry
from tunnel_guard.core.farfield import FarField


def motion_from_track(g,travel):
    if g is None or not g.ok or not np.isfinite(travel) or travel<=1e-5 or travel>3.:
        return (0.,0.,0.,0.)
    # A bounded local curvature estimate suppresses noisy single-node derivatives.
    mask=(g.xs>=3.)&(g.xs<=15.);q=g.xs[mask];h=np.polyfit(q,g.yc[mask],2)
    curvature=float(np.clip(2*h[0],-1/200.,1/200.))
    heading=float(np.clip(h[1],-.12,.12))
    yaw=curvature*travel/(1+heading*heading)
    dy=heading*travel+.5*curvature*travel*travel
    dz=float(g.rail_z(travel)-g.rail_z(0.))
    return float(travel),dy,float(np.clip(dz,-.1,.1)),yaw


def transform(points,motion):
    dx,dy,dz,yaw=motion
    if not any(motion):return points.copy()
    c,s=np.cos(yaw),np.sin(yaw);v=points-np.array([dx,dy,dz])
    return np.column_stack((c*v[:,0]+s*v[:,1],-s*v[:,0]+c*v[:,1],v[:,2])).astype(np.float32)


def transport_geometry(g,motion):
    if not any(motion):return g
    p=transform(np.column_stack((g.xs,g.yc,g.zr)),motion);x=p[:,0]
    if np.any(np.diff(x)<=0):return g
    lerp=lambda a:None if a is None else np.interp(g.xs,x,a)
    return TrackGeometry(g.xs,np.interp(g.xs,x,p[:,1]),np.interp(g.xs,x,p[:,2]),roll=g.roll,
        rail_range=max(0.,g.rail_range-motion[0]),valid_range=max(0.,g.valid_range-motion[0]),ok=g.ok,
        meas=g.meas,sigma_l=lerp(g.sigma_l),sigma_v=lerp(g.sigma_v),rand_l=lerp(g.rand_l),rand_v=lerp(g.rand_v),
        trusted_range=max(0.,g.trusted_range-motion[0]))


class MotionGeometry(GeometryEstimator):
    def __init__(self,cfg=None,hypotheses=True):
        super().__init__(cfg);self.hypotheses=hypotheses;self.motion=(0.,0.,0.,0.);self.alternate_selected=0;self.alternate_tested=0

    def estimate(self,pts,travel=0.):
        if not np.isfinite(travel) or travel<0. or travel>3.:
            self.prev=None;self.bed_hist=[];self.geo_hist=[];self.motion=(0.,0.,0.,0.);self.discontinuity=True
            return super().estimate(pts,travel=0.)
        self.discontinuity=False
        self.motion=motion_from_track(self.prev,travel)
        if any(self.motion):
            self.prev=transport_geometry(self.prev,self.motion)
            # Parent uses a longitudinal shift. Pre-transform compensates the
            # missing yaw/grade without applying longitudinal translation twice.
            shift=np.array([travel,0.,0.],np.float32)
            self.bed_hist=[transform(h,self.motion)+shift for h in self.bed_hist]
            self.geo_hist=[transform(h,self.motion)+shift for h in self.geo_hist]
        return super().estimate(pts,travel)

    def _coarse_dp(self,far,y_ref,z_ref,x_start,search=None):
        xc,off,info=super()._coarse_dp(far,y_ref,z_ref,x_start,search)
        if not self.hypotheses or not any(self.motion) or len(far)<100:return xc,off,info
        q=(self.xs>=5)&(self.xs<=25)
        coeff=np.polyfit(self.xs[q],y_ref[q],2)
        if abs(2*coeff[0])>1/200. or abs(2*coeff[0])<1/2500.:return xc,off,info
        alt=np.polyval(coeff,self.xs)
        # Long range is explicitly limited. The alternative is seeded from the
        # near pair of rails, not from points inside the obstacle corridor.
        ax,ao,ai=super()._coarse_dp(far,alt,z_ref,x_start,None)
        profile0=y_ref+np.interp(self.xs,xc,off)
        profile1=alt+np.interp(self.xs,ax,ao)
        def score(y):
            g=TrackGeometry(self.xs,y,z_ref,roll=self.roll);_,l,h=g.to_track(far)
            select=(np.abs(l)>self.cfg.core_half_width+.15)&(h>.3)&(h<4.5)&(far[:,0]<180)
            if select.sum()<100:return np.inf
            val,_,_=self.template.sample(self.cfg.dp_trunc,l[select],h[select]);bins=(far[select,0]/5).astype(int)
            count=np.bincount(bins);mean=np.bincount(bins,weights=val)/np.maximum(count,1)
            return float(np.mean(mean[count>=5]))
        a,b=score(profile0),score(profile1);self.alternate_tested+=1
        if np.isfinite(b) and b+.02<a and ai.sum()>=3:
            self.alternate_selected+=1
            return ax,np.interp(ax,self.xs,alt-y_ref)+ao,ai
        return xc,off,info


class ReprojectedFarField(FarField):
    def __init__(self,cfg=None,id_base=100000,geometry=None):
        super().__init__(cfg,id_base);self.geometry=geometry;self.history={};self._has_motion=False
    def reset(self):
        geo=self.geometry;self.__init__(self.cfg,self.id_base,geo)
    def update(self,p,g,s_from,s_to,travel=0.,exclude=None,trusted=True):
        if not self.cfg.enabled or g is None or not g.ok:self.reset();return []
        if getattr(self.geometry,'discontinuity',False):self.history.clear();self.ids[:]=0
        cfg=self.cfg;motion=self.geometry.motion if self.geometry is not None else (travel,0.,0.,0.)
        self.history={k:(transform(q,motion),sc) for k,(q,sc) in self.history.items() if self.k-k<cfg.window}
        self._has_motion=self._has_motion or any(abs(v)>1e-5 for v in motion)
        if not self._has_motion:
            self._capture(p,g,s_from,s_to,exclude,trusted)
            return super().update(p,g,s_from,s_to,travel=0.,exclude=exclude,trusted=trusted)
        # Rebuild each old scan into the *current* track coordinates. One row per
        # original stamp is essential: accumulating points must not fake hits.
        for a,fill in ((self.cnt,0),(self.scnt,0),(self.hmin,9.),(self.hmax,-9.),(self.lsum,0.),(self.ssum,0.),(self.smin,np.inf),(self.smax,-np.inf)):a[:]=fill
        for k,(q,scatter) in self.history.items():
            s,l,h=g.to_track(q);good=(s>=max(s_from,cfg.s_min))&(s<=min(s_to,cfg.s_max-2*cfg.cell))&(abs(l)<cfg.half_width)&(h>cfg.h_lo)&(h<cfg.h_hi)
            b=np.floor(s[good]/cfg.cell).astype(int);row=k%cfg.window;sc=scatter[good]
            np.add.at(self.scnt[row],b[sc],1)
            s,l,h,b=s[good][~sc],l[good][~sc],h[good][~sc],b[~sc]
            np.add.at(self.cnt[row],b,1);np.add.at(self.lsum[row],b,l);np.add.at(self.ssum[row],b,s)
            np.minimum.at(self.hmin[row],b,h);np.maximum.at(self.hmax[row],b,h)
            if cfg.span_max>0:np.minimum.at(self.smin[row],b,s);np.maximum.at(self.smax[row],b,s)
        self._capture(p,g,s_from,s_to,exclude,trusted)
        return super().update(p,g,s_from,s_to,travel=0.,exclude=exclude,trusted=trusted)

    def _capture(self,p,g,s_from,s_to,exclude,trusted):
        cfg=self.cfg
        eligible=np.flatnonzero((p[:,0]>max(s_from,cfg.s_min)-2)&(p[:,0]<min(s_to,cfg.s_max)+2))
        if not trusted:eligible=eligible[:0]
        s,l,h=g.to_track(p[eligible])
        keep=(abs(l)<cfg.half_width+.4)&(h>cfg.h_lo-.15)&(h<cfg.h_hi+.15)
        idx=eligible[keep]
        self.history[self.k]=(p[idx].copy(),np.zeros(len(idx),bool) if exclude is None else exclude[idx].copy())



def install(detector,hypotheses=True,reproject=True):
    detector.geo=MotionGeometry(detector.cfg.geometry,hypotheses=hypotheses)
    if reproject:
        detector.far=ReprojectedFarField(detector.cfg.farfield,geometry=detector.geo)
        detector.low=ReprojectedFarField(detector.cfg.lowfield,id_base=200000,geometry=detector.geo)
    return detector
