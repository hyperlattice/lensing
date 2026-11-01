"""Bounded-memory A roots: analytical candidates plus exact-profile multistarts.

The analytical-only prototype missed one faint inner root in qualification.
Sparse inner/outer starts therefore always supplement it. No dense search
image is allocated, and B's independently solved roots remain unchanged.
"""
import numpy as np
from scipy.optimize import root

from src.simulation.precise_point_renderer_v1 import PreciseSceneEngine,fourth_order_amplitudes


class LowMemorySceneEngine(PreciseSceneEngine):
    def points(self):
        if self.renderer!="lenstronomy":
            return super().points()
        if self._points is None:
            p=self.p
            source=np.array([p["source_x_arcsec"],p["source_y_arcsec"]])
            center=np.array([p["lens_center_x_arcsec"],p["lens_center_y_arcsec"]])
            with np.errstate(divide="ignore",invalid="ignore"):
                x,y=self.solver.image_position_from_source(*source,self.mass_kwargs,solver="analytical",arrival_time_sort=False)
            starts=list(np.column_stack((x,y)))
            angles=np.linspace(0,2*np.pi,48,endpoint=False)
            for radius in (.005,.02,.1,.35,.65,.9,1.1,1.4,1.8):
                starts.extend(center+radius*p["theta_E_arcsec"]*np.column_stack((np.cos(angles),np.sin(angles))))
            positions=[]
            def residual(x):
                return x-self.alpha(x[None])[0]-source
            for candidate in starts:
                solved=root(residual,candidate,options={"xtol":1e-11,"maxfev":100})
                position=solved.x
                radius=np.linalg.norm(position-center)
                if (not np.isfinite(position).all() or radius<1e-7 or radius>4*p["theta_E_arcsec"] or
                    np.linalg.norm(residual(position))>1e-8):
                    continue
                if not any(np.linalg.norm(position-other)<1e-6 for other in positions):
                    positions.append(position)
            if len(positions) not in (2,4):
                raise ValueError(f"Low-memory SIE root multiplicity {len(positions)}")
            positions=np.asarray(positions)
            self._points=positions,fourth_order_amplitudes(self,positions)
        return self._points
