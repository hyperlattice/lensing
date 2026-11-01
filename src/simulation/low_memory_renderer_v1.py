"""Low-memory A root initialization with exact-profile refinement.

Use Lenstronomy's specialized one-dimensional analytical solver for starting
solutions, then refine against its actual SIE deflections. B remains the
independent external root search. Pixel quadrature and point Jacobians are
unchanged. Qualification compares against all previous A reference solutions.
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
            with np.errstate(divide="ignore",invalid="ignore"):
                x,y=self.solver.image_position_from_source(*source,self.mass_kwargs,
                    solver="analytical",arrival_time_sort=False)
            initial=np.column_stack((x,y))
            positions=[]
            def residual(x):
                return x-self.alpha(x[None])[0]-source
            for candidate in initial:
                solved=root(residual,candidate,options={"xtol":1e-11})
                position=solved.x
                if not np.isfinite(position).all() or np.linalg.norm(residual(position))>1e-8:
                    raise ValueError("Analytical-start root failed exact SIE refinement")
                if not any(np.linalg.norm(position-other)<1e-6 for other in positions):
                    positions.append(position)
            if len(positions) not in (2,4):
                raise ValueError(f"Low-memory SIE root multiplicity {len(positions)}")
            positions=np.asarray(positions)
            self._points=positions,fourth_order_amplitudes(self,positions)
        return self._points
