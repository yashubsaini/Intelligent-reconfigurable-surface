import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from irs_env import IRSEnv
from td3_agent import TD3Agent
import os
import torch
import sys

def main():
    torch.set_num_threads(1)
    print("--- IRS-Sim DRL Training (TD3) ---")
    
    # 1. Initialize Environment and Agent (8x8 IRS)
    env = IRSEnv(room_size=30.0, num_elements_x=8, num_elements_y=8)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = TD3Agent(state_dim=state_dim, action_dim=action_dim)
    
    num_episodes = 250
    batch_size = 128
    learning_starts = 1000
    
    rewards_history = []
    eval_rewards_history = []
    eval_episodes = []
    
    noise_start = 0.5
    noise_end = 0.05
    noise_decay_episodes = 150
    
    print(f"State Dim: {state_dim}, Action Dim: {action_dim}")
    print("Starting Training Loop...")
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_snr = 0.0
        done = False
        
        total_policy_loss = 0.0
        total_qf_loss = 0.0
        train_steps = 0
        
        noise = max(noise_end, noise_start - (noise_start - noise_end) * episode / noise_decay_episodes)
        
        while not done:
            # TD3 uses additive Gaussian exploration noise
            if len(agent.memory) < learning_starts:
                # Pure exploration initially
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, exploration_noise=noise)
            
            next_state, reward, done, _, info = env.step(action)
            episode_snr += info.get("snr_db", 0.0)
            
            agent.memory.push(state, action, reward, next_state, done)
            
            # Train the network
            if len(agent.memory) > learning_starts:
                for i in range(2):
                    p_loss, q_loss = agent.train(batch_size)
                    total_policy_loss += p_loss
                    total_qf_loss += q_loss
                    train_steps += 1
                
            state = next_state
            episode_reward += reward
            
        rewards_history.append(episode_reward)
        
        avg_policy_loss = total_policy_loss / max(1, train_steps)
        avg_qf_loss = total_qf_loss / max(1, train_steps)
        
        avg_snr = episode_snr / max(1, env.max_steps)
        if (episode + 1) % 10 == 0:
            print(
                f"Episode {episode+1:4d}/{num_episodes} | "
                f"Reward: {episode_reward:7.3f} | "
                f"Avg SNR: {avg_snr:7.3f} dB | "
                f"Noise: {noise:.2f} | "
                f"Actor Loss: {avg_policy_loss:8.4f} | "
                f"Critic Loss: {avg_qf_loss:8.4f}"
            )
            sys.stdout.flush()
            
        # Greedy Evaluation Loop
        if (episode + 1) % 20 == 0:
            eval_state, _ = env.reset()
            eval_reward = 0
            eval_done = False
            while not eval_done:
                eval_action = agent.select_action(eval_state, exploration_noise=0.0)
                eval_state, r, eval_done, _, _ = env.step(eval_action)
                eval_reward += r
            eval_rewards_history.append(eval_reward)
            eval_episodes.append(episode)
        
    print("Training Complete!")
    
    # Save the trained Actor network weights
    save_path = os.path.join(os.path.dirname(__file__), 'td3_actor.pth')
    torch.save(agent.actor.state_dict(), save_path)
    print(f"Saved TD3 actor weights to: {save_path}")
    
    # Plot Learning Curve
    window = 20
    plt.figure(figsize=(10, 5))

    # Raw rewards
    plt.plot(
        rewards_history,
        alpha=0.35,
        color="skyblue",
        label="Episode Reward"
    )

    # Moving average
    if len(rewards_history) >= window:
        moving_avg = np.convolve(
            rewards_history,
            np.ones(window) / window,
            mode="valid"
        )
        plt.plot(
            moving_avg,
            color="blue",
            linewidth=2,
            label=f"{window}-Episode Moving Average"
        )
        
    # Greedy Eval Curve
    if len(eval_episodes) > 0:
        plt.plot(
            eval_episodes,
            eval_rewards_history,
            color="red",
            linewidth=2,
            marker='o',
            label="Greedy Eval (Noise=0)"
        )

    plt.title("TD3 Agent Learning Curve (IRS Beamforming)")
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plot_path = os.path.join(os.path.dirname(__file__), 'learning_curve.png')
    plt.savefig(plot_path)
    print(f"Saved learning curve plot to: {plot_path}")

if __name__ == "__main__":
    main()
