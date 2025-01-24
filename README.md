# ScreenPoissonPy

This is a tiny python package for screened poisson in 
gradient-domain rendering. 

## Usage 

```python
import pyexr
import screenpoissonpy as sp

primal = pyexr.read(primal_image_path)
dx = pyexr.read(dx_image_path)
dy = pyexr.read(dy_image_path)

params = sp.PoissonParams()
params.setConfigPreset(args.recon_type)
solver = sp.PoissonSolver(params)
solver.load(primal, dx, dy)
solver.setupBackend()
solver.solve()
recon = solver.getFinalImage()

pyexr.write(recon_path, recon)
```