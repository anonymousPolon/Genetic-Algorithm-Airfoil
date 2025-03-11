import numpy as np
import pygad
import matplotlib.pyplot as plt
import os
import subprocess
import logging
import sys
from shapely.geometry import LineString
from cst import cst, fit
from objective import findMaxCL_CD, findMaxAlpha
from datetime import datetime

# Constants
NUM_COEFFS = 4  # Number of CST coefficients for upper and lower surfaces
POPULATION_SIZE = 16
NUM_GENERATIONS = 50
NUM_PARENTS = 5
MUTATION_RATE = 20
REYNOLDS_NUMBER = 70000
ALPHA_RANGE = (-5, 20)
ALPHA_STEP = 0.25
NUM_ITERATIONS = 200
XFOIL_TIMEOUT = 60  # Timeout in seconds for XFOIL execution
MIN_FITNESS = 1e-6 # Small non-zero fitness value for invalid solutions

# Configure logging
def setup_logging():
    """Set up logging to save logs to a file in the 'logging' folder and capture console output."""
    # Create the 'logging' folder if it doesn't exist
    if not os.path.exists("log"):
        os.makedirs("log")
    
    # Define the log file name with a timestamp
    log_file = f"log/optimization_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,  # Set the log level
        format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
        handlers=[
            logging.FileHandler(log_file),  # Save logs to a file
            logging.StreamHandler()  # Print logs to the console
        ]
    )
    
    # Redirect stdout and stderr to the logging system
    class StreamToLogger:
        def __init__(self, logger, log_level=logging.INFO):
            self.logger = logger
            self.log_level = log_level
            self.linebuf = ''
        
        def write(self, buf):
            for line in buf.rstrip().splitlines():
                self.logger.log(self.log_level, line.rstrip())
        
        def flush(self):
            pass
    
    # Redirect stdout and stderr
    stdout_logger = logging.getLogger('STDOUT')
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl
    
    stderr_logger = logging.getLogger('STDERR')
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

def check_negative_camber(airfoil_coords):
    """
    Checks if an airfoil has negative camber.
    
    Parameters:
        airfoil_coords (list of tuples): List of (x, y) coordinates of the airfoil.
    
    Returns:
        bool: True if the airfoil has negative camber, False otherwise.
    """
    # Sort coordinates by x-value
    airfoil_coords = sorted(airfoil_coords, key=lambda point: point[0])

    # Separate upper and lower surfaces
    x_vals = [p[0] for p in airfoil_coords]
    y_vals = [p[1] for p in airfoil_coords]

    # Assume upper and lower surfaces are split at the leading edge
    midpoint = len(x_vals) // 2
    upper_surface = airfoil_coords[:midpoint+1]
    lower_surface = airfoil_coords[midpoint:]

    # Compute camber line (midpoint between upper and lower surfaces)
    camber_line = []
    for (xu, yu), (xl, yl) in zip(upper_surface, lower_surface):
        xc = (xu + xl) / 2
        yc = (yu + yl) / 2
        camber_line.append((xc, yc))
    
    # Compute chord line (straight line from leading to trailing edge)
    x_le, y_le = airfoil_coords[0]  # Leading edge
    x_te, y_te = airfoil_coords[-1]  # Trailing edge
    chord_slope = (y_te - y_le) / (x_te - x_le)
    
    # Check if any part of the camber line is below the chord line
    for xc, yc in camber_line:
        y_chord = y_le + chord_slope * (xc - x_le)  # Chord line equation
        if yc < y_chord:
            return True  # Negative camber detected
    
    return False  # No negative camber

# Check if an airfoil has self intersection
def has_self_intersection(coordinates):
    # Exclude the first and last points
    points = coordinates[1:-1]
    
    # Create a closed polygon (connect last point to first)
    polygon = points + [points[0]]
    
    # Create LineString objects for each edge
    edges = [LineString([polygon[i], polygon[i+1]]) for i in range(len(polygon)-1)]
    
    # Check for intersections between non-adjacent edges
    for i in range(len(edges)):
        for j in range(i+2, len(edges)-1):  # Skip adjacent edges
            if edges[i].crosses(edges[j]):
                return True
    return False

# Chebyshev grid step
def chebyshev_grid(n):
    """Generate Chebyshev-distributed points in [0, 1], clustering near endpoints."""
    return 0.5 * (1 - np.cos(np.linspace(0, np.pi, n)))

# Find CST coefficients for initial airfoils
def find_CST_coeff(airfoil_path, n):
    """
    Find CST coefficients for a given airfoil in Selig format.
    
    Parameters:
        airfoil_path (str): Path to the airfoil data file.
        n (int): Number of CST coefficients for each surface.
    
    Returns:
        np.ndarray: Array of CST coefficients (upper surface first, then lower surface).
    """
    # Load airfoil data
    airfoil = np.loadtxt(airfoil_path, skiprows=1)
    _x = airfoil[:, 0]
    _y = airfoil[:, 1]
    
    # Find the index of the leading edge (minimum x-value)
    leading_edge_idx = np.argmin(_x)
    
    # Separate upper and lower surfaces
    upper_surface = []
    lower_surface = []
    
    # Traverse from the trailing edge to the leading edge (upper surface)
    for i in range(leading_edge_idx):
        upper_surface.append((_x[i], _y[i]))
    
    # Traverse from the leading edge to the trailing edge (lower surface)
    for i in range(leading_edge_idx, len(_x)):
        lower_surface.append((_x[i], _y[i]))
    
    # Convert lists to numpy arrays
    upper_surface = np.array(upper_surface)
    lower_surface = np.array(lower_surface)
    
    # Reverse the upper surface to match the order from trailing edge to leading edge
    upper_surface = upper_surface[::-1]
    
    # Check if the upper_surface is actually the lower surface
    # Compare the y-coordinates at the same x-coordinates
    if np.mean(upper_surface[:, 1]) < np.mean(lower_surface[:, 1]):
        upper_surface, lower_surface = lower_surface, upper_surface
    
    # Fit CST coefficients for upper and lower surfaces
    up_coeff, _ = fit(upper_surface[:, 0], upper_surface[:, 1], n)
    lo_coeff, _ = fit(lower_surface[:, 0], lower_surface[:, 1], n)
    
    # Return concatenated coefficients (upper surface first)
    return np.concatenate((up_coeff, lo_coeff))

# Run XFOIL with timeout and error handling
def run_xfoil_with_timeout(airfoil_path, polar_file_path, timeout=60):
    """Run XFOIL with a timeout using subprocess.run."""
    try:
        # Create the XFOIL input file
        with open("input_file.in", "w") as input_file:
            input_file.write(f"LOAD {airfoil_path}\n")
            input_file.write("PANE\n")
            input_file.write("OPER\n")
            input_file.write(f"Visc {REYNOLDS_NUMBER}\n")
            input_file.write("PACC\n")
            input_file.write(f"{polar_file_path}\n\n")
            input_file.write(f"ITER {NUM_ITERATIONS}\n")
            input_file.write(f"ASeq {ALPHA_RANGE[0]} {ALPHA_RANGE[1]} {ALPHA_STEP}\n")
            input_file.write("PACC\n")
            input_file.write("QUIT\n")
        
        # Run XFOIL with a timeout
        result = subprocess.run(
            ["xfoil.exe"],
            stdin=open("input_file.in", "r"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout
        )
        
        # Print XFOIL output
        print("XFOIL stderr:\n", result.stderr)
        
        # Check for errors in the output
        # if "VISCAL non-convergence" in result.stdout or "ERROR" in result.stdout or result.stderr:
        #     print("XFOIL encountered an error during execution.")
        #     return False
        
        # Check if the polar file was created and is not empty
        if not os.path.exists(polar_file_path) or os.path.getsize(polar_file_path) == 0:
            print(f"Polar file {polar_file_path} is empty or does not exist.")
            return False
        
        return True  # XFOIL executed successfully
    
    except subprocess.TimeoutExpired:
        print(f"XFOIL execution timed out after {timeout} seconds.")
        try:
            # Use numpy to read the polar file
            polar_data = np.loadtxt(polar_file_path, skiprows=12)
            if polar_data.size == 0:
                print(f"Polar file {polar_file_path} has no data after skipping 12 rows.")
                return False
            else:
                return True
        except Exception as e:
            print(f"Error reading polar file {polar_file_path}: {e}")
            return False
        
    except FileNotFoundError:
        print("Error: xfoil.exe not found. Ensure it is in the correct directory.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred while running XFOIL: {e}")
        return False


# Define fitness function
def fitness_function(ga_instance, Au_Al, solution_idx):
    """Fitness function for the genetic algorithm."""
    generation = ga_instance.generations_completed
    airfoil_name = f"generated_airfoil_gen{generation}_sol{solution_idx}"
    airfoil_path = f"airfoil/{airfoil_name}.dat"

    num_coeffs = len(Au_Al) // 2
    Au = np.array(Au_Al[:num_coeffs])
    Al = np.array(Au_Al[num_coeffs:])
    
    num_points = 100
    _x = chebyshev_grid(num_points)
    
    x_upper, y_upper = _x[::-1], cst(_x[::-1], Au)
    x_lower, y_lower = _x, cst(_x, Al)
    
    x = np.concatenate((x_upper, x_lower))
    y = np.concatenate((y_upper, y_lower))
    
    # Combine x and y into airfoil coordinates
    airfoil_coordinates = list(zip(x, y))
    
    # Plot the airfoil
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o', linestyle='-', markersize=3)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f"Generated Airfoil - Generation {generation} Solution {solution_idx}")
    plt.grid(True)
    plt.axis('equal')
    plt.savefig(f"figure/Generated Airfoil - Generation {generation} Solution {solution_idx}")
    plt.close()
    
    # Check for self-intersections
    if has_self_intersection(airfoil_coordinates):
        print(f"Airfoil in generation {generation}, solution {solution_idx} has self-intersections. Penalizing fitness.")
        return MIN_FITNESS  # Return a low fitness value for invalid airfoils
    
    # if check_negative_camber(airfoil_coordinates):
    #     print(f"Airfoil in generation {generation}, solution {solution_idx} has negative camber. Penalizing fitness.")
    #     return MIN_FITNESS 
    
    # Save the airfoil coordinates with the airfoil name as the header
    header = airfoil_name
    np.savetxt(airfoil_path, airfoil_coordinates, fmt="%.6f", delimiter=" ", header=header, comments="")
    
    # Run XFOIL analysis
    polar_file_name = f"polar_file_generated_airfoil_gen{generation}_sol{solution_idx}_re_{REYNOLDS_NUMBER}.txt"
    polar_file_path = os.path.join("polar", polar_file_name)
    
    if not os.path.exists(polar_file_path):
        if not run_xfoil_with_timeout(airfoil_path, polar_file_path):
            return MIN_FITNESS  # Return a small non-zero fitness value if XFOIL fails
    
    # Calculate fitness
    maxCL_CD = findMaxCL_CD(polar_file_name) or 0
    maxAlpha = findMaxAlpha(polar_file_name) or 0
    fitness = 0.7 * maxCL_CD + 0.3 * maxAlpha  # Weighted fitness
    return max(fitness, MIN_FITNESS)  # Ensure fitness is never zero or negative

# Configure Genetic Algorithm
def configure_ga(parameter_bounds, initial_population):
    """Configure the genetic algorithm."""
    ga_instance = pygad.GA(
        num_generations=NUM_GENERATIONS,
        num_parents_mating=NUM_PARENTS,
        fitness_func=fitness_function,
        sol_per_pop=POPULATION_SIZE,
        num_genes=len(parameter_bounds),
        gene_space=parameter_bounds,
        crossover_type="two_points",
        mutation_percent_genes=MUTATION_RATE,
        parent_selection_type="rws",
        initial_population=initial_population,
        save_best_solutions=True  # Enable tracking of best solutions
    )
    return ga_instance

# Main function
def main():
    """Main function to run the genetic algorithm."""
    # Set up logging
    setup_logging()
    
    # Log the start of the optimization process
    logging.info("Starting the genetic algorithm optimization.")
    
    # # Define bounds for Au0 and Al0
    # au0_bounds = {'low': 0.0, 'high': 0.2}
    # al0_bounds = {'low': -0.2, 'high': 0.0}

    # # Define bounds for the remaining coefficients
    # default_bounds = {'low': -0.2, 'high': 0.2}

    # # Generate parameter bounds
    # parameter_bounds = [au0_bounds] + [default_bounds] * (NUM_COEFFS - 1)  # Upper surface
    # parameter_bounds += [al0_bounds] + [default_bounds] * (NUM_COEFFS - 1)  # Lower surface
    
    parameter_bounds = [
        {'low': 0.0, 'high': 0.2},    # Au0
        {'low': 0.0, 'high': 0.35},   # Au1
        {'low': 0.0, 'high': 0.3},    # Au2
        {'low': -0.1, 'high': 0.3},   # Au3
        {'low': -0.2, 'high': 0.0},   # Al0
        {'low': -0.1, 'high': 0.3},   # Al1
        {'low': -0.2, 'high': 0.1},   # Al2
        {'low': -0.1, 'high': 0.2}    # Al3
        ]   
    
    
    parent_files = [
        "airfoil/parent/sd7032.dat",
        "airfoil/parent/A18 (smoothed).dat",
        "airfoil/parent/arad6.dat",
        "airfoil/parent/e387.dat",
        "airfoil/parent/e471.dat",
        "airfoil/parent/eh2070.dat",
        "airfoil/parent/goe342.dat",
        "airfoil/parent/mh62.dat"
    ]
    
    initial_population = [find_CST_coeff(f, NUM_COEFFS) for f in parent_files]
    ga_instance = configure_ga(parameter_bounds, initial_population)
    
    ga_instance.run()
    ga_instance.plot_fitness()

    # Get the best solution across all generations
    best_solution, best_fitness, best_solution_idx = ga_instance.best_solution()

    # Find the generation in which the best solution was found
    best_solutions = ga_instance.best_solutions  # List of best solutions per generation
    best_fitnesses = ga_instance.best_solutions_fitness  # List of best fitness values per generation

    # Find the generation where the best fitness was achieved
    best_generation = np.argmax(best_fitnesses)  # Index of the generation with the best fitness

    print("Best Airfoil Parameters:", best_solution)
    print("Best Fitness Value:", best_fitness)
    print(f"Best Airfoil is: generated_airfoil_gen{best_generation+1}_sol{best_solution_idx}.dat")

if __name__ == "__main__":
    main() 
#    coordinates = np.loadtxt('airfoil/generated_airfoil_gen1_sol6.dat')
#    test = has_self_intersection(coordinates)
#    print(test)