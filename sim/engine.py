# /sim/engine.py
import numpy as np
import qutip as qt

# /sim/engine.py
import numpy as np
import qutip as qt

def simulate_jaynes_cummings(wc, wa, g, kappa, gamma, t_max=40.0, t_steps=400):
    """Solves the Lindblad master equation for the Jaynes-Cummings model."""
    N = 10 
    
    a = qt.tensor(qt.destroy(N), qt.qeye(2))
    sm = qt.tensor(qt.qeye(N), qt.sigmam())
    sz = qt.tensor(qt.qeye(N), qt.sigmaz())
    sp = sm.dag()
    
    H = wc * a.dag() * a + 0.5 * wa * sz + g * (a.dag() * sm + a * sp)
    c_ops = [np.sqrt(kappa) * a, np.sqrt(gamma) * sm]
    
    # Initial state: Atom excited, cavity empty
    psi0 = qt.tensor(qt.basis(N, 0), qt.basis(2, 0))
    tlist = np.linspace(0, t_max, t_steps)
    
    # Pass an empty list for e_ops so QuTiP saves the full density matrix at every time step
    result = qt.mesolve(H, psi0, tlist, c_ops=c_ops, e_ops=[])
    
    # Calculate expectation values manually from the saved states
    p_excited = qt.expect(sp * sm, result.states)
    n_photons = qt.expect(a.dag() * a, result.states)
    
    return tlist, p_excited, n_photons, result.states