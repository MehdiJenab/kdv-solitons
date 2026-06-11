#!/usr/bin/env python3
"""Simple web interface for KdV soliton propagation visualization."""

from flask import Flask, jsonify, render_template, request

from kdv_solver.solver import (
    Grid,
    KdVProblem,
    PseudoSpectralSolver,
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

        # Create the simulation grid
        grid = Grid(int(params['N']), float(params['L']))
        problem = KdVProblem(grid, 1.0)  # kappa only used as solver scaffolding

        # Build the initial condition as a superposition of any number of
        # solitons (each {kappa, x0}) and Gaussian packets (each
        # {amplitude, x0, width}). Fall back to a single centered soliton.
        soliton_specs = [
            (float(s['kappa']), float(s['x0']))
            for s in (params.get('solitons') or [])
        ]
        gaussians = params.get('gaussians') or []
        if not soliton_specs and not gaussians:
            soliton_specs = [(float(params.get('kappa', 0.5)), grid.L / 2)]

        u0 = multi_soliton_field(grid, soliton_specs)
        for g in gaussians:
            u0 = u0 + gaussian_packet(
                grid, float(g['amplitude']), float(g['x0']), float(g['width'])
            )

        # Cap the number of time steps so a long lap (small kappa -> large
        # t_final) stays responsive. ETDRK4 is stable at large dt, so a coarser
        # step only mildly affects accuracy.
        dt = float(params['dt'])
        t_final = float(params['t_final'])
        max_steps = 20000
        if t_final / dt > max_steps:
            dt = t_final / max_steps
        solver = PseudoSpectralSolver(problem, dt)

        # Time-evolve with ETDRK4, recording snapshots for animation
        times, solutions = solver.solve_with_history(
            t_final, n_snapshots=50, u0=u0
        )

        return jsonify({
            'success': True,
            'times': times.tolist(),
            'solutions': [u.tolist() for u in solutions],
            'params': params
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