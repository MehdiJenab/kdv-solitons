#!/usr/bin/env python3
"""Web interface for KdV soliton propagation visualization."""

from flask import Flask, render_template, jsonify, request
import numpy as np
from kdv_solver.solver import Grid, KdVProblem, PseudoSpectralSolver
import threading
import time

app = Flask(__name__)

# Global variables for current simulation
current_simulation = None
simulation_lock = threading.Lock()

class SimulationState:
    def __init__(self):
        self.running = False
        self.params = {
            'kappa': 0.5,
            'N': 256,
            'L': 100.0,
            'dt': 0.001,
            't_final': 1.0
        }
        self.solution_history = []
        self.current_time = 0.0
        self.last_update = time.time()

def run_simulation(params):
    """Run the KdV simulation with given parameters."""
    try:
        # Create the simulation components
        grid = Grid(params['N'], params['L'])
        problem = KdVProblem(grid, params['kappa'])

        # Create solver with parameters
        solver = PseudoSpectralSolver(problem, params['dt'])

        # Time-evolve with ETDRK4, recording snapshots for animation
        times, solutions = solver.solve_with_history(
            float(params['t_final']), n_snapshots=50
        )

        return {
            'success': True,
            'times': times,
            'solutions': solutions,
            'params': params
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'params': params
        }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/simulate', methods=['POST'])
def simulate():
    """Start a simulation with given parameters."""
    global current_simulation

    params = request.get_json()

    # Validate parameters
    required_params = ['kappa', 'N', 'L', 'dt', 't_final']
    for param in required_params:
        if param not in params:
            return jsonify({'error': f'Missing parameter: {param}'}), 400

    # Update simulation parameters
    with simulation_lock:
        current_simulation = SimulationState()
        current_simulation.params.update(params)

    # Run simulation in background thread
    def background_run():
        result = run_simulation(params)
        with simulation_lock:
            if current_simulation:
                current_simulation.solution_history = result.get('solutions', [])
                current_simulation.times = result.get('times', [])
                current_simulation.running = False

    thread = threading.Thread(target=background_run)
    thread.daemon = True
    thread.start()

    with simulation_lock:
        if current_simulation:
            current_simulation.running = True

    return jsonify({'status': 'started'})

@app.route('/api/status')
def status():
    """Get current simulation status."""
    with simulation_lock:
        if current_simulation is None:
            return jsonify({'running': False, 'params': None})

        return jsonify({
            'running': current_simulation.running,
            'params': current_simulation.params,
            'current_time': current_simulation.current_time
        })

@app.route('/api/data')
def get_data():
    """Get simulation data."""
    with simulation_lock:
        if current_simulation is None:
            return jsonify({'error': 'No simulation running'}), 400

        return jsonify({
            'times': [float(t) for t in current_simulation.times],
            'solutions': [u.tolist() for u in current_simulation.solution_history],
            'params': current_simulation.params
        })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)