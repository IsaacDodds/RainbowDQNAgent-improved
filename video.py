import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch
import numpy as np
import ale_py
from network import C51NoisyDuelingCNN
from datetime import datetime
import os
import glob

# =====================
# Device
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# Play + record video
# =====================
def play_video():
    game_name = "Frostbite"

    # Timestamp (safe for filenames)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    temp_prefix = f"{game_name}_TEMP_{timestamp}"

    # --- Environment ---
    env = gym.make(
        "ALE/Frostbite-v5",
        render_mode="rgb_array",
        frameskip=1
    )

    env = gym.wrappers.AtariPreprocessing(
        env,
        grayscale_obs=True,
        frame_skip=4,
        scale_obs=True
    )

    env = gym.wrappers.FrameStackObservation(env, 4)

    env = RecordVideo(
        env,
        video_folder="./video",
        name_prefix=temp_prefix,
        episode_trigger=lambda ep: True
    )

    # --- MODEL CONFIG (MATCH TRAINING) ---
    num_actions = 18
    atom_size = 51
    v_min, v_max = 0, 3000
    support = torch.linspace(v_min, v_max, atom_size).to(device)

    model = C51NoisyDuelingCNN(
        num_actions=num_actions,
        atom_size=atom_size,
        support=support
    ).to(device)

    # --- LOAD CHECKPOINT ---
    checkpoint_path = r"runs\run-ALE\Frostbite-v5-20251222-121235\model_1766445160.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["policy_net"])
    model.eval()

    print("🎥 Recording video with trained Rainbow model...")

    obs, _ = env.reset()
    state = torch.from_numpy(np.asarray(obs)).float().unsqueeze(0).to(device)

    done = False
    total_reward = 0.0

    while not done:
        with torch.no_grad():
            dist = model(state)
            q = (dist * support).sum(dim=2)
            action = q.argmax(1).item()

        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

        state = torch.from_numpy(np.asarray(obs)).float().unsqueeze(0).to(device)

    env.close()

    # =====================
    # Rename video file
    # =====================
    video_files = glob.glob(f"./video/{temp_prefix}*.mp4")
    assert len(video_files) == 1, "Expected exactly one video file"

    final_name = (
        f"{game_name}_reward-{int(total_reward)}_{timestamp}.mp4"
    )
    final_path = os.path.join("./video", final_name)

    os.rename(video_files[0], final_path)

    # =====================
    # Print summary
    # =====================
    print(f"✅ Video saved to {final_path}")
    print(f"🏆 Total reward: {total_reward}")
    print(f"🕒 Created at: {timestamp}")

# =====================
# Run
# =====================
if __name__ == "__main__":
    play_video()
