import datetime
import os
import matplotlib.pyplot as plt



def create_video_directory():
    path = os.path.dirname(os.path.abspath(__file__))
    print(path)
    path = os.path.join(path, "Outputs")
    if not os.path.exists(path):
        os.makedirs(path)


def create_timestamped():
    datestamp = datetime.datetime.now().strftime("%d-%m-%y")
    timestamp = datetime.datetime.now().strftime("%H-%M-%S")
    filestamp = f"{datestamp}_{timestamp}"

    return filestamp

def create_logdir():

    create_video_directory()
    base_path = os.path.dirname(os.path.abspath(__file__))
    logdir = os.path.join(base_path, "Outputs")
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    timestamped_dir = os.path.join(logdir, create_timestamped())
    return  timestamped_dir
    



def video(self, filename="lunar_lander_video"):
    # We need to recreate the environment with the Video Wrapper
    from gymnasium.wrappers import RecordVideo
    
    # 1. Create a specific environment for recording
    # render_mode="rgb_array" is required for video
    video_env = gym.make("LunarLander-v3", render_mode="rgb_array")
    
    # 2. Wrap it to save to folder './video'
    video_env = RecordVideo(video_env, video_folder="./video", name_prefix=filename, episode_trigger=lambda x: True)
    
    print("Recording video...")
    state, _ = video_env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
    done = False
    
    # 3. Run the Loop (No Epsilon - Pure Exploitation)
    while not done:
        with torch.no_grad():
            # Always pick the best action (max)
            action = self.policy_net(state).max(1)[1].view(1, 1)
        
        observation, reward, terminated, truncated, _ = video_env.step(action.item())
        done = terminated or truncated
        
        state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

    video_env.close()
    print(f"Video saved to ./video/{filename}.mp4")



def plot_rewards(self):
    #This will have 2 grpahs so 1 and 2
    plt.figure(1)



    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)

    if show_result:
        plt.title('Final Result')
    else:
        plt.clf()
        plt.title('Training...')

    plt.xlabel('Episode')
    plt.ylabel('Reward')

    # Raw reward line
    plt.plot(rewards_t.numpy())

    # 50-episode moving average (same idea as tutorial)
    if len(rewards_t) >= 50:
        means = rewards_t.unfold(0, 50, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(49), means))  # pad so lengths match
        plt.plot(means.numpy(), linewidth=2)

    plt.pause(0.001)

    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


        
