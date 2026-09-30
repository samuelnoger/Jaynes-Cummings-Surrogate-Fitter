# Jaynes-Cummings Parameter-Fitting Pipeline with Neural Surrogates

A modular, lightweight Python pipeline designed to perform parameter fitting for a Jaynes-Cummings system using a pre-trained neural surrogate model. This project supports both terminal execution and interactive Jupyter Notebook experimentation.

---

## Key Features
* **Neural Surrogate Modeling:** Replaces slow numerical solvers with a fast, pre-trained neural network to predict quantum trajectories.
* **Physical Parameter Optimization:** Fits physical parameters ($g$, $\kappa$, $\gamma$) efficiently using PyTorch with positivity constraints enforced via `softplus`.
* **Flexible Execution:** Works seamlessly in both script-based terminal workflows and interactive Jupyter notebooks via robust argument parsing (`parse_known_args`).
* **Modular Structure:** Clean separation of simulation engines (`sim.py`), model architectures (`model.py`), and fitting pipelines (`fit_param/`).

---

## Tech Stack & References
This project relies on the following core open-source libraries:
* **PyTorch** – Deep learning framework for the surrogate model and optimization loops.
* **QuTiP (Quantum Toolbox in Python)** – Used for simulating the open quantum system dynamics via master equation solvers. 
  > *If you use or build upon the QuTiP components of this project in academic work, please cite:*
  > J. R. Johansson, P. D. Nation, and F. Nori, *"QuTiP 2: A Python framework for the dynamics of open quantum systems,"* Comput. Phys. Commun. **184**, 1234 (2013).
* **NumPy & Matplotlib** – Data manipulation and trajectory visualization.

---

## Acknowledgments & AI Usage
* Developed as part of quantum optics and machine learning explorations. 
* **AI Assistance:** Generative AI tools were utilized during the development of this project to assist with code structuring, debugging argument parsing quirks, and optimizing pipeline modularity.
