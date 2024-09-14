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
    #apply sigmoid to rules values
    rules = 1 / (1 + np.exp(-rules))
    rules = np.where(rules > 0.5, 1, 0)

    b = LS(X, Y, production_rules=rules)
    for i in range(X):
            b.update()
    
    result = b.data

    errors_frames = []
    for frame in result:
        loss = np.sum(np.square(target - frame))
        errors_frames.append(loss)
    loss = np.mean(errors_frames)
    reward = 1 / (1 + loss)
    return reward

    sys.exit()
    loss = np.sum(np.square(targets - result))
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
    #target = np.zeros((5,5))
    #target[4,:] = 1
    #target[:,4] = 1
    #target = target[:int(target.shape[0]/2), :int(target.shape[1]/2)]
    X,Y = target.shape
    N_PRODUCTION_RULES = 5

    print(f'target.size {target.size}')
    
    N_PARAMETERS = N_PRODUCTION_RULES * 2 * 3 * 3 # reactants and products
    
    best_individual = evolve(target=target, 
                             num_params=N_PARAMETERS,
                             popsize=100,
                             n_generations=50)
    
    rules = np.copy(best_individual)
    rules = rules.reshape(-1, 2, 3, 3) # [N_PRODUCTION_RULES, reactant and products, 3, 3]
    rules = 1 / (1 + np.exp(-rules))
    rules = np.where(rules > 0.5, 1, 0)

    b = LS(X, Y, production_rules=rules)
    print(f'P: {b.P}')
    
    for i in range(X):
        b.update()


    data = b.data
    print(data[-1])
    Visuals.create_visualization_grid(data, filename=f'Test', duration=100, gif=True, video=False)

    import matplotlib.pyplot as plt
    # side by side comparison
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(target, cmap='gray')
    ax[0].set_title('Target')
    ax[1].imshow(data[-1], cmap='gray')
    ax[1].set_title('Result')
    #save image
    plt.savefig('Result.png', dpi=300, bbox_inches='tight')
    plt.close()

    result_path = 'Evolution/results.json'
    with open(result_path, 'r') as f:
        results = json.load(f)

    rewards = results['REWARDS']
    mean_rewards = np.mean(rewards, axis=1)
    std_rewards = np.std(rewards, axis=1)
    plt.figure()
    plt.plot(mean_rewards)
    #fill between
    plt.fill_between(range(len(mean_rewards)), mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.3)
    #plot max rewards
    best_rewards = np.max(rewards, axis=1)
    plt.plot(best_rewards, 'r')
    plt.title('Best rewards')
    plt.xlabel('Generation')
    plt.ylabel('Reward')
    plt.savefig('Best_rewards.png', dpi=300, bbox_inches='tight')
    plt.close()