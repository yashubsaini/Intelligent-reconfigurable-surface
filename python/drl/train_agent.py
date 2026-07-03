import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from irs_env import IRSEnv
from ddpg_agent import DDPGAgent
import os

def main():
    print("--- IRS-Sim DRL Training (DDPG) ---")
    
    # 1. Initialize Environment
    # We will use an 8x8 panel for faster training proof-of-concept
    env = IRSEnv(room_size=100.0, num_elements_x=8, num_elements_y=8)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"State Dimension: {state_dim} (UE X, Y)")
    print(f"Action Dimension: {action_dim} (IRS Elements)")
    
    # 2. Initialize Agent
    agent = DDPGAgent(state_dim, action_dim)
    
    num_episodes = 50
    rewards_history = []
    
    # 3. Training Loop
    print(f"Starting training for {num_episodes} episodes...")
    for ep in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        
        # Decay noise over time for exploitation
        noise_scale = max(0.01, 0.5 * (1.0 - ep/num_episodes))
        
        while not (done or truncated):
            action = agent.get_action(state, noise_scale=noise_scale)
            next_state, reward, done, truncated, _ = env.step(action)
            
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            # Train the network
            a_loss, c_loss = agent.train()
            
            state = next_state
            episode_reward += reward
            
        avg_reward = episode_reward / env.max_steps
        rewards_history.append(avg_reward)
        print(f"Episode: {ep+1}/{num_episodes} | Avg SNR: {avg_reward:.2f} dB | Noise: {noise_scale:.2f}")

    print("Training complete!")
    
    # 4. Save Learning Curve
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_history)
    plt.title('DDPG Agent Learning Curve (Avg SNR per Episode)')
    plt.xlabel('Episode')
    plt.ylabel('Average Received Power (dB)')
    plt.grid(True)
    plt.savefig('learning_curve.png', dpi=300, bbox_inches='tight')
    print("Saved learning curve to 'learning_curve.png'")
    
    # Save model weights
    import torch
    torch.save(agent.actor.state_dict(), 'ddpg_actor.pth')
    print("Saved model weights to 'ddpg_actor.pth'")

if __name__ == '__main__':
    main()
