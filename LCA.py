import os
import json
import sys
import numpy as np
from tqdm import tqdm
from Evolution import es
import Visuals
from LSystemsNEW import LS
from tqdm import tqdm
import Util
np.set_printoptions(precision=2, suppress=True)



def _reward_function_individual(individual:np.array, target:np.array): 
    #print(target)
    #print(individual)

    # Create LSystem with individual production rules
    X,Y = target.shape
    rules = np.copy(individual)
    rules = rules.reshape(-1, 2, 3, 3) # [N_PRODUCTION_RULES, reactant and products, 3, 3]


    b = LS(X, Y, production_rules=rules)
    for i in range(X):
            b.update()
    
    result = b.data
    loss = np.sum(np.square(target - result[-1]))
    #errors_frames = []
    #for frame in result:
    #    loss = np.sum(np.square(target - frame))
    #    errors_frames.append(loss)
    #loss = np.mean(errors_frames)
    reward = 1 / (1 + loss)
    return reward

def evolve(target:np.array, num_params:int, n_generations=100, popsize=20):
    
    solver = es.CMAES(num_params=num_params, popsize=popsize, weight_decay=0.01, sigma_init=0.5)
    
    results = {'BEST': [],'REWARDS': []}
    
    for g in range(n_generations):
        solutions = solver.ask()
        fitness_list = np.zeros(solver.popsize)

        for i in range(solver.popsize):
            fitness_list[i] = _reward_function_individual(np.array(solutions[i]), target)
            
        solver.tell(fitness_list)
        result = solver.result()
        
        best_params, best_reward, curr_reward, sigma = result[0], result[1], result[2], result[3]
        print(f'G:{g}, BEST PARAMS, BEST REWARD: {best_reward}, CURRENT REWARD: {curr_reward}')
        
        results['BEST'].append(best_params.tolist())
        results['REWARDS'].append(fitness_list.tolist())

        this_dir = os.path.dirname(os.path.abspath(__file__))        
        file_path = os.path.join(this_dir, 'Evolution/results.json')
    
    with open(file_path, 'w') as f:
        json.dump(results, f)

    return best_params

if __name__ == '__main__':
    seed = np.random.randint(0, 100000000)
    np.random.seed(seed)
    
    target = Util.load_image_as_numpy_array('Mario.png', black_and_white=True, binary=True, sensibility=0.1)
    #resize image to 8x8
    #target = np.zeros((5,5))
    #target[4,:] = 1
    #target[:,4] = 1
    #target = target[:int(target.shape[0]/2), :int(target.shape[1]/2)]
    X,Y = target.shape
    N_PRODUCTION_RULES = 15

    print(f'target.size {target.size}')
    
    N_PARAMETERS = N_PRODUCTION_RULES * 2 * 3 * 3 # reactants and products
    
    best_individual = evolve(target=target, 
                             num_params=N_PARAMETERS,
                             popsize=5,
                             n_generations=5)
    
    rules = np.copy(best_individual)
    rules = rules.reshape(-1, 2, 3, 3) # [N_PRODUCTION_RULES, reactant and products, 3, 3]


    b = LS(X, Y, production_rules=rules)
    print(f'P: {b.P}')
    
    for i in range(X):
        b.update()


    data = b.data
    print(data[-1])
    Visuals.create_visualization_grid(data, filename=f'_MEDIA_LCA/Test', duration=100, gif=True, video=False)
    Visuals.visualize_target_result(target, data, filename='_MEDIA_LCA/Result.png')
    Visuals.visualize_evolution_results(result_path='Evolution/results.json', filename='_MEDIA_LCA/Best_rewards.png')

