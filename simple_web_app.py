#!/usr/bin/env python3
"""Simple web interface for KdV soliton propagation visualization."""

import math

import numpy as np
from flask import Flask, jsonify, render_template, request

from kdv_solver.solver import (
    Grid,
    KdVProblem,
    PseudoSpectralSolver,
    cosine_wave,
    gaussian_packet,
    multi_soliton_field,
)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/simulate', methods=['POST'])
def simulate():
    """Run the KdV simulation and return time-evolved snapshots."""
    try:
        params = request.get_json()

        # Validate parameters
        required_params = ['N', 'L', 'dt', 't_final']
        for param in required_params:
            if param not in params:
                return jsonify({'error': f'Missing parameter: {param}'}), 400

        domain_length = float(params['L'])

        # Collect the initial-condition components.
        soliton_specs = [
            (float(s['kappa']), float(s['x0']))
            for s in (params.get('solitons') or [])
        ]
        gaussians = params.get('gaussians') or []
        cosines = params.get('cosines') or []
        if not soliton_specs and not gaussians and not cosines:
            soliton_specs = [(float(params.get('kappa', 0.5)), domain_length / 2)]

        # Auto-resolution: pick a grid and time step fine enough to resolve and
        # stably evolve the tallest feature, so tall/narrow solitons "just work"
        # instead of diverging. A soliton of height A=2*kappa^2 has width ~1/kappa
        # (needs enough points across it) and advects at speed ~6A (sets the
        # explicit-step CFL limit). Gaussians/cosines steepen into solitons up to
        # ~2x their height, so budget for that.
        heights = [2.0 * k * k for k, _ in soliton_specs]
        heights += [2.0 * float(g['amplitude']) for g in gaussians]
        heights += [2.0 * float(c['amplitude']) for c in cosines]
        u_eff = max(heights) if heights else 1.0
        kappa_eff = math.sqrt(u_eff / 2.0)

        # Spatial: ~6 grid points across the narrowest width, rounded up to a
        # power of two (fast FFT); never coarser than requested, capped for sanity.
        n_req = max(int(params['N']), math.ceil(6.0 * kappa_eff * domain_length))
        n_used = min(8192, 1 << (n_req - 1).bit_length())

        grid = Grid(n_used, domain_length)
        problem = KdVProblem(grid, 1.0)  # kappa only used as solver scaffolding

        u0 = multi_soliton_field(grid, soliton_specs)
        for g in gaussians:
            u0 = u0 + gaussian_packet(
                grid, float(g['amplitude']), float(g['x0']), float(g['width'])
            )
        for c in cosines:
            u0 = u0 + cosine_wave(grid, float(c['amplitude']), int(c['mode']))

        # Temporal: nonlinear (advective) CFL, dt <~ dx / (6 * u_peak). Use the
        # actual peak of the field; never coarser than requested.
        u_peak = float(np.max(np.abs(u0))) or 1.0
        dt_stable = grid.dx / (6.0 * u_peak)
        dt = min(float(params['dt']), dt_stable)

        # Bound total work; the finiteness guard below catches the rare case
        # where even this many steps is not enough.
        t_final = float(params['t_final'])
        max_steps = 120000
        if t_final / dt > max_steps:
            dt = t_final / max_steps
        solver = PseudoSpectralSolver(problem, dt)

        # Time-evolve with ETDRK4, recording snapshots for animation
        times, solutions = solver.solve_with_history(
            t_final, n_snapshots=50, u0=u0
        )

        # Safety net: if it still diverged, report cleanly (never emit NaN JSON).
        if not all(np.all(np.isfinite(u)) for u in solutions):
            return jsonify({
                'error': 'Simulation diverged even at maximum resolution. Try a '
                         'smaller domain (L) or a shorter final time.'
            }), 400

        # Report the resolution actually used so the client plots correctly.
        params = dict(params)
        params['N'] = n_used
        params['dt'] = dt
        return jsonify({
            'success': True,
            'times': times.tolist(),
            'solutions': [u.tolist() for u in solutions],
            'params': params,
            'auto_resolution': {'N': n_used, 'dt': dt},
        })

    except Exception as e:
        print(f"Error in simulation: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/status')
def status():
    """Get current simulation status."""
    return jsonify({'running': False, 'params': None})

@app.route('/api/data')
def get_data():
    """Get simulation data."""
    return jsonify({'error': 'No simulation running'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)