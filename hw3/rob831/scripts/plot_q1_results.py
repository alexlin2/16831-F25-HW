import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def get_section_results(logdir):
    ea = event_accumulator.EventAccumulator(logdir)
    ea.Reload()

    steps = []
    returns = []

    if 'Train_AverageReturn' in ea.Tags()['scalars']:
        for event in ea.Scalars('Train_AverageReturn'):
            returns.append(event.value)
        for event in ea.Scalars('Train_EnvstepsSoFar'):
            steps.append(event.value)

    min_len = min(len(steps), len(returns))
    return np.array(steps[:min_len]), np.array(returns[:min_len])

def get_results_for_exp(logdir_pattern):
    logdirs = glob.glob(logdir_pattern)
    results = {}
    for logdir in sorted(logdirs):
        X, Y = get_section_results(logdir)
        if len(X) > 0 and len(Y) > 0:
            dirname = os.path.basename(logdir)
            results[dirname] = (X, Y)
    return results

def interpolate_to_common_steps(results_dict, num_points):
    all_min_steps = []
    all_max_steps = []
    for dirname, (steps, returns) in results_dict.items():
        if len(steps) > 0:
            all_min_steps.append(steps[0])
            all_max_steps.append(steps[-1])
    if len(all_min_steps) == 0:
        return None, None, None
    common_min = max(all_min_steps)
    common_max = min(all_max_steps)
    common_steps = np.linspace(common_min, common_max, num_points)
    interpolated_returns = []
    for dirname, (steps, returns) in results_dict.items():
        interp_returns = np.interp(common_steps, steps, returns)
        interpolated_returns.append(interp_returns)
    returns_array = np.array(interpolated_returns)
    mean_returns = np.mean(returns_array, axis=0)
    std_returns = np.std(returns_array, axis=0)
    return common_steps, mean_returns, std_returns

if __name__ == '__main__':
    data_dir = '../../data'
    save_path = '../../q1_results.png'

    dqn_pattern = os.path.join(data_dir, 'q1_dqn_*')
    dqn_results = get_results_for_exp(dqn_pattern)

    ddqn_pattern = os.path.join(data_dir, 'q1_doubledqn_*')
    ddqn_results = get_results_for_exp(ddqn_pattern)

    dqn_steps, dqn_mean, dqn_std = interpolate_to_common_steps(dqn_results, 300)
    ddqn_steps, ddqn_mean, ddqn_std = interpolate_to_common_steps(ddqn_results, 300)

    plt.figure(figsize=(10, 6))

    if dqn_steps is not None:
        plt.plot(dqn_steps, dqn_mean, label='DQN', linewidth=2)
        plt.fill_between(dqn_steps, dqn_mean - dqn_std, dqn_mean + dqn_std, alpha=0.2)

    if ddqn_steps is not None:
        plt.plot(ddqn_steps, ddqn_mean, label='Double DQN', linewidth=2)
        plt.fill_between(ddqn_steps, ddqn_mean - ddqn_std, ddqn_mean + ddqn_std, alpha=0.2)

    plt.xlabel('Environment Steps')
    plt.ylabel('Average Return')
    plt.title('DQN vs Double DQN on LunarLander-v3')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    ax = plt.gca()
    ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
