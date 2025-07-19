import ale_py
import shimmy

from typing import Literal
import datetime

import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
import argparse
import cv2
import numpy as np
import datetime
import os


class EpisodeRenderCallback(BaseCallback):
    """
    Renders one episode every N episodes using a dedicated rendering environment.
    The rendering can be accelerated by 'render_speed_factor'.
    The displayed image can be scaled by 'display_scale_factor'.
    Set render_every_n_episodes=0 to disable rendering.
    """
    def __init__(self, env_id: str, verbose: int = 1,
                 render_mode: Literal["rgb_array", "human"]="rgb_array",
                 fps: int = 60,
                 frame_skip = 4):
        super().__init__(verbose)
        self.env_id = env_id # Store env_id
        self.render_mode = render_mode
        self.fps = fps
        self.frame_skip = 4

    def _on_training_start(self) -> None:
        """
        Called once at the beginning of training.
        Initializes the dedicated rendering environment.
        This environment is also frame-stacked to match the model's input expectations.
        """
        print("Creating dedicated rendering environment...")
        self.render_env = make_vec_env(
            env_id=self.env_id,
            n_envs=1,
            seed=0,
            vec_env_cls=DummyVecEnv,
            wrapper_class=AtariWrapper,
            env_kwargs={"render_mode": self.render_mode,
                        "frameskip": self.frame_skip} # , "frameskip": 1},
        )
        print(f"Dedicated rendering environment created for {self.env_id} and frame stacked (using rgb_array).")

    def _on_training_end(self) -> None:
        """
        Called once at the end of training.
        Closes the dedicated rendering environment and any OpenCV windows.
        """
        if self.render_env is not None:
            print("Closing dedicated rendering environment...")
            self.render_env.close()
            print("Dedicated rendering environment closed.")
        cv2.destroyAllWindows()

    def _on_step(self) -> bool:
       return True

    def _render_episode(self):
        """
        Renders a single full episode using the dedicated rendering environment.
        The render_env is a VecFrameStack, so its outputs are batched.
        Rendering is accelerated by skipping visual frames and using OpenCV.
        """
        if self.render_env is None:
            print("Warning: render_env not initialized. Cannot render episode.")
            return

        obs = self.render_env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _states = self.model.predict(obs, deterministic=True) # type: ignore
            obs, reward, done, _info = self.render_env.step(action)
            total_reward += reward[0]

            # frame is none when render_method is rgb_array
            if (frame := self.render_env.render()) is not None:
                cv2.imshow("Game", cv2.resize(frame, (512, 512)))

                # 4 frame skip
                sleep_time = self.frame_skip / self.fps
                time.sleep(sleep_time)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Rendering stopped by user (pressed 'q').")
                    done = True


        print(f"Episode Rendered. Total Reward: {total_reward}")

    def _on_rollout_end(self) -> None:
        """
        Called after each collection of experience (rollout).
        This is where the model has fresh episode statistics.
        We'll print the mean reward from recently completed episodes.
        """
        self._render_episode()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a PPO agent on an Atari environment.")
    parser.add_argument("--env-id", type=str, default="ALE/Pong-v5",
                        help="The ID of the Gymnasium environment to train on (e.g., 'ALE/Pong-v5', 'ALE/Breakout-v5').")
    parser.add_argument("--n-steps", type=int, default=2048,
                        help="PPO number of steps for each environment per update")
    parser.add_argument("--timesteps", type=int, default=100000,
                        help="Total number of timesteps to train the model.")
    parser.add_argument("--num-envs", type=int, default=os.cpu_count() or 4,
                        help="Number of parallel environments for vectorized training (defaults to number of CPU cores).")
    parser.add_argument("--learning-rate", type=float, default=0.00025,
                        help="Learning rate for the optimizer (default: 0.00025).")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Learning rate for the optimizer (default: 0.00025).")
    parser.add_argument("--model-path", type=str, default="weights/",
                        help="Path to the directory for saving/loading model weights. Default is 'weights/'.")
    parser.add_argument("--tb-log-dir", type=str, default="tensorboard_logs",
                        help="Path to the directory for TensorBoard logs. Default is 'tensorboard_logs/'.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="The device to use for training the model")
    parser.add_argument("--render-mode", type=str, choices="human rgb_array".split(), default="rgb_array",
                        help="Render mode")
    parser.add_argument("--fps", type=int, default=60,
                        help="Render mode")
    parser.add_argument("--frame-skip", type=int, default=4,
                        help="Setup the frameskip for the ALE envrionment")
    args = parser.parse_args()

    current_time = datetime.datetime.now().strftime("%A, %B %d, %Y at %I:%M:%S %p %Z")
    print(f"Current time: {current_time}")

    # --- Setup Model Saving/Loading Paths ---
    model_dir = args.model_path
    os.makedirs(model_dir, exist_ok=True)
    
    # Sanitize environment ID for use in filename: replace all slashes with underscores
    env_name_for_filename = args.env_id.replace("/", "_") # Remove -v5 for cleaner name
    model_file_name = f"ppo_{env_name_for_filename}_model.zip"
    model_file_path = os.path.join(model_dir, model_file_name)

    # --- Create Vectorized Training Environment (Using make_vec_env for parallelization) ---
    print(f"Creating {args.num_envs} vectorized environments for training using make_vec_env...")
    train_env = make_vec_env(
        env_id=args.env_id,
        n_envs=args.num_envs,
        seed=0,
        vec_env_cls=SubprocVecEnv,
        wrapper_class=AtariWrapper,
    )

    print("Vectorized training environment created and frame stacked.")
    callback = EpisodeRenderCallback(
        env_id=args.env_id,
        verbose=int(args.verbose),
        render_mode=args.render_mode,
        fps=args.fps,
        frame_skip=args.frame_skip,
    )

    # --- Load or Create Model ---
    model = None
    if os.path.exists(model_file_path):
        print(f"Loading existing model from {model_file_path}...")
        model = PPO.load(model_file_path, env=train_env, device=args.device,
                         n_steps=args.n_steps, tensorboard_log=args.tb_log_dir, learning_rate=args.learning_rate)
        print("Model loaded successfully. Resuming training.")
    else:
        print(f"No existing model found at {model_file_path}. Creating a new model...")
        model = PPO("CnnPolicy", train_env, verbose=1, device=args.device,
                    n_steps=args.n_steps, tensorboard_log=args.tb_log_dir, learning_rate=args.learning_rate)
        print("New model created.")

    # --- Start Training ---
    print(f"Starting PPO training for {args.timesteps} timesteps...")
    try:
        model.learn(total_timesteps=args.timesteps, callback=callback)
        print("Training finished successfully.")
        model.save(model_file_path)
        print(f"Final model saved to: {model_file_path}")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user (Ctrl-C). Saving model state...")
        model.save(model_file_path)
        print(f"Model saved to: {model_file_path}")
    finally:
        if train_env is not None:
            try:
                train_env.close()
                print("Training environment closed successfully.")
            except EOFError:
                print("EOFError encountered while closing SubprocVecEnv. This can happen during an abrupt KeyboardInterrupt.")
                print("The model was saved, and the script will now exit.")
            except Exception as e:
                print(f"An unexpected error occurred while closing the environment: {e}")
        print("Cleanup complete.")

    print("Script execution finished.")
