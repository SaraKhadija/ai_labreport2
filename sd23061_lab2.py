import streamlit as st
import random
import pandas as pd
import time

# --- 1. GA Parameters (Based on User Requirements) ---
POPULATION_SIZE = 300
INDIVIDUAL_LENGTH = 80
GENERATIONS = 50
TARGET_ONES = 50 
MAX_FITNESS = 80

# --- 2. Hyperparameters ---
MUTATION_RATE = 0.05
ELITISM_COUNT = 1 # Keep the single best individual in the next generation

# --- 3. GA Core Functions ---

def create_individual(length):
    """Generates a random bit pattern (list of 0s and 1s)."""
    return [random.randint(0, 1) for _ in range(length)]

def initialize_population(pop_size, length):
    """Creates the initial population."""
    return [create_individual(length) for _ in range(pop_size)]

def calculate_fitness(individual):
    """
    Calculates fitness. Max fitness (80) is achieved when '1's count is 50.
    """
    ones_count = sum(individual)
    
    if ones_count == TARGET_ONES:
        return MAX_FITNESS
    else:
        # Fitness is based on the absolute distance from the target of 50
        distance = abs(ones_count - TARGET_ONES)
        # Linear fitness function: closer to 50 ones = higher fitness
        return MAX_FITNESS - distance

def select_parent(fitnesses):
    """Simple selection method (Tournament selection style for demonstration)."""
    # Randomly pick a few individuals and return the one with the best fitness among them
    sample_size = 5
    sample = random.sample(fitnesses, sample_size)
    best_parent, _ = max(sample, key=lambda x: x[1])
    return best_parent

def crossover(parent1, parent2):
    """Single-point crossover."""
    crossover_point = random.randint(1, INDIVIDUAL_LENGTH - 1)
    child = parent1[:crossover_point] + parent2[crossover_point:]
    return child

def mutate(individual):
    """Bit-flip mutation based on MUTATION_RATE."""
    for i in range(INDIVIDUAL_LENGTH):
        if random.random() < MUTATION_RATE:
            individual[i] = 1 - individual[i] # Flips 0 to 1, or 1 to 0
    return individual

# --- 4. Streamlit UI and Main Loop ---

st.set_page_config(layout="wide")
st.title("🧬 Genetic Algorithm for Bit Pattern Optimization")
st.markdown("---")

# Display required parameters
st.sidebar.header("GA Configuration")
st.sidebar.metric("Population Size", POPULATION_SIZE)
st.sidebar.metric("Individual Length (Bits)", INDIVIDUAL_LENGTH)
st.sidebar.metric("Generations to Run", GENERATIONS)
st.sidebar.metric("Target '1's Count", TARGET_ONES)
st.sidebar.metric("Max Fitness Value", MAX_FITNESS)
st.sidebar.metric("Mutation Rate", MUTATION_RATE)

# Placeholder elements for live updates
status_text = st.empty()
progress_bar = st.progress(0)
best_results_container = st.container()
chart_container = st.empty()
final_result_container = st.empty()

if st.button("Start Genetic Algorithm Simulation"):
    
    # Initialize Population
    population = initialize_population(POPULATION_SIZE, INDIVIDUAL_LENGTH)
    
    # DataFrame to track results for charting
    history_data = []
    
    best_individual_overall = None
    best_fitness_overall = -1

    # Main GA Loop
    for generation in range(1, GENERATIONS + 1):
        
        # 1. Evaluation
        fitnesses = [(individual, calculate_fitness(individual)) for individual in population]
        fitnesses.sort(key=lambda x: x[1], reverse=True)
        
        current_best_individual, current_best_fitness = fitnesses[0]

        # Update overall best
        if current_best_fitness > best_fitness_overall:
            best_fitness_overall = current_best_fitness
            best_individual_overall = current_best_individual

        # Record and display current generation results
        history_data.append({
            'Generation': generation,
            'Best Fitness': current_best_fitness,
            'Ones Count': sum(current_best_individual)
        })
        
        df_history = pd.DataFrame(history_data)
        
        # Update UI elements
        progress_bar.progress(generation / GENERATIONS)
        status_text.info(f"Running Generation **{generation}** of {GENERATIONS}...")
        
        with best_results_container:
            st.subheader(f"Generation {generation} Results")
            col1, col2 = st.columns(2)
            col1.metric("Best Fitness", f"{current_best_fitness} / {MAX_FITNESS}", 
                        f"{current_best_fitness - (history_data[-2]['Best Fitness'] if len(history_data) > 1 else 0):+.2f}")
            col2.metric("Optimal '1's Count", f"{sum(current_best_individual)} / {TARGET_ONES}")
            
        chart_container.line_chart(df_history.set_index('Generation'), y=['Best Fitness', 'Ones Count'])
        
        # 2. Reproduction
        new_population = [ind for ind, fit in fitnesses[:ELITISM_COUNT]] # Elitism
        
        while len(new_population) < POPULATION_SIZE:
            
            # Selection
            parent1 = select_parent(fitnesses)
            parent2 = select_parent(fitnesses)
            
            # Crossover
            child = crossover(parent1, parent2)
            
            # Mutation
            child = mutate(child)
            
            new_population.append(child)
            
        population = new_population
        
        # Add a small delay to make the visualization visible
        time.sleep(0.05) 

    # --- Final Results Display ---
    status_text.success("✅ Simulation Complete!")
    progress_bar.progress(1.0)
    
    with final_result_container:
        st.subheader("Final Best Solution Found")
        st.markdown(f"**Overall Best Fitness:** `{best_fitness_overall}` (Target: `{MAX_FITNESS}`)")
        st.markdown(f"**Overall '1's Count:** `{sum(best_individual_overall)}` (Target: `{TARGET_ONES}`)")
        st.code("".join(map(str, best_individual_overall)), language='text')