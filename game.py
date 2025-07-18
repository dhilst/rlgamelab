import ale_py
import shimmy

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, SubprocVecEnv
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.callbacks import BaseCallback
import argparse
import cv2
import numpy as np
import datetime
import os # Import the os module for path operations


def make_env(render_mode=None):
    """
    Creates a single Atari Pong environment.
    Args:
        render_mode (str, optional): The render mode for the environment (e.g., "human", "rgb_array").
                                     Defaults to None, meaning no rendering unless explicitly specified.
    Returns:
        gym.Env: The wrapped Atari Pong environment.
    """
    env = gym.make("ALE/Pong-v5", render_mode=render_mode)
    env = AtariWrapper(env) # This wrapper affects observations, not necessarily render() output directly
    return env


class EpisodeRenderCallback(BaseCallback):
    """
    Renders one episode every N episodes using a dedicated rendering environment.
    The rendering can be accelerated by 'render_speed_factor'.
    The displayed image can be scaled by 'display_scale_factor'.
    Set render_every_n_episodes=0 to disable rendering.
    """
    def __init__(self, render_every_n_episodes: int = 100, render_speed_factor: int = 10,
                 display_scale_factor: int = 4, verbose: int = 1):
        super().__init__(verbose)
        self.render_every_n_episodes = render_every_n_episodes
        self.render_speed_factor = max(1, render_speed_factor) # Ensure it's at least 1
        self.display_scale_factor = max(1, display_scale_factor) # Ensure scale factor is at least 1
        self.episode_counter = 0
        self.render_env = None  # Dedicated environment for rendering
        self.window_name = "Pong Accelerated Render" # Name for the OpenCV window
        self.agent_view_shape = (84, 84, 3) # The shape the agent sees (after AtariWrapper)


    def _on_training_start(self) -> None:
        """
        Called once at the beginning of training.
        Initializes the dedicated rendering environment.
        This environment is also frame-stacked to match the model's input expectations.
        """
        if self.render_every_n_episodes > 0:
            print("Creating dedicated rendering environment...")
            single_base_env = make_env(render_mode="rgb_array")
            # For the rendering environment, DummyVecEnv is fine as it's a single instance
            self.render_env = VecFrameStack(DummyVecEnv([lambda: single_base_env]), n_stack=4)
            print(f"Dedicated rendering environment created and frame stacked (using rgb_array).")
            print(f"Render speed factor: {self.render_speed_factor}x, Display scale factor: {self.display_scale_factor}x.")
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)

    def _on_training_end(self) -> None:
        """
        Called once at the end of training.
        Closes the dedicated rendering environment and any OpenCV windows.
        """
        if self.render_env is not None:
            print("Closing dedicated rendering environment...")
            self.render_env.close()
            print("Dedicated rendering environment closed.")
        cv2.destroyAllWindows() # Close all OpenCV windows

    def _on_step(self) -> bool:
        """
        Called at each training step. Checks if an episode has ended
        and triggers rendering if it's the right interval.
        """
        if self.render_every_n_episodes <= 0:
            return True

        if self.locals.get("dones") is None:
            return True

        for done in self.locals["dones"]:
            if done:
                self.episode_counter += 1
                if self.episode_counter % self.render_every_n_episodes == 0:
                    print(f"\n🎮 Rendering episode {self.episode_counter} using OpenCV...")
                    self._render_episode()
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
        frame_counter = 0

        while not done:
            action, _states = self.model.predict(obs, deterministic=True)
            obs, reward, done_array, info = self.render_env.step(action)
            total_reward += reward[0]

            frame_counter += 1
            if frame_counter % self.render_speed_factor == 0:
                # Get the RAW RGB array from the environment (will typically be 210, 160, 3)
                frame = self.render_env.render()

                assert isinstance(frame, np.ndarray), \
                    f"Expected frame to be a numpy array, but got {type(frame)}"

                raw_game_shape = (210, 160, 3)
                assert frame.shape == raw_game_shape, \
                    f"Expected RAW frame shape {raw_game_shape} from render_env.render(), but got {frame.shape}. " \
                    "This is the native resolution of the Atari game."
                assert frame.dtype == np.uint8, \
                    f"Expected RAW frame dtype to be np.uint8, but got {frame.dtype}"

                # First, resize the raw frame to the agent's observation size (84x84)
                # Using INTER_NEAREST to preserve pixelated features even on downscale
                agent_view_frame = cv2.resize(frame,
                                           (self.agent_view_shape[1], self.agent_view_shape[0]), # (width, height)
                                           interpolation=cv2.INTER_NEAREST)

                # Now, scale up this 84x84 frame for better human visibility
                display_width = int(self.agent_view_shape[1] * self.display_scale_factor)
                display_height = int(self.agent_view_shape[0] * self.display_scale_factor)

                # Use INTER_NEAREST for crisp pixel art scaling
                scaled_display_frame = cv2.resize(agent_view_frame,
                                                  (display_width, display_height),
                                                  interpolation=cv2.INTER_NEAREST)

                # Convert RGB (from env) to BGR (for OpenCV display)
                frame_bgr = cv2.cvtColor(scaled_display_frame, cv2.COLOR_RGB2BGR)

                # Assert the final display shape
                expected_final_display_shape = (display_height, display_width, 3)
                assert frame_bgr.shape == expected_final_display_shape, \
                    f"Expected final BGR frame shape {expected_final_display_shape}, but got {frame_bgr.shape}"

                # Display the frame using OpenCV
                cv2.imshow(self.window_name, frame_bgr)

                # Wait for 1 millisecond for key press events and window updates
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Rendering stopped by user (pressed 'q').")
                    done = True

            if done_array[0]:
                done = True

        # Ensure final frame is rendered if not already (and if not quit by user)
        if not ('key' in locals() and key == ord('q')) and (frame_counter % self.render_speed_factor != 0):
             try:
                 final_frame = self.render_env.render()
                 raw_game_shape = (210, 160, 3)
                 if isinstance(final_frame, np.ndarray) and final_frame.shape == raw_game_shape:
                     final_agent_view_frame = cv2.resize(final_frame,
                                                         (self.agent_view_shape[1], self.agent_view_shape[0]),
                                                         interpolation=cv2.INTER_NEAREST)
                     final_display_width = int(self.agent_view_shape[1] * self.display_scale_factor)
                     final_display_height = int(self.agent_view_shape[0] * self.display_scale_factor)
                     final_scaled_display_frame = cv2.resize(final_agent_view_frame,
                                                             (final_display_width, final_display_height),
                                                             interpolation=cv2.INTER_NEAREST)
                     final_frame_bgr = cv2.cvtColor(final_scaled_display_frame, cv2.COLOR_RGB2BGR)
                     cv2.imshow(self.window_name, final_frame_bgr)
                     cv2.waitKey(1)
                 else:
                     print("Warning: Final frame render did not return expected numpy array or raw shape.")
             except Exception as e:
                 print(f"Warning: Could not render final frame: {e}")


        print(f"Episode Rendered. Total Reward: {total_reward}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a PPO agent on Pong-v5.")
    parser.add_argument("--timesteps", type=int, default=100000,
                        help="Total number of timesteps to train the model.")
    # Changed to --num-envs and default to os.cpu_count()
    parser.add_argument("--num-envs", type=int, default=os.cpu_count(),
                        help="Number of parallel environments for vectorized training (defaults to number of CPU cores).")
    parser.add_argument("--render-every", type=int, default=1,
                        help="Render one episode every N episodes (0 to disable).")
    parser.add_argument("--render-speed", type=int, default=10,
                        help="Accelerate rendered episodes by skipping frames (e.g., 10 means 10x faster visuals). Default is 10.")
    parser.add_argument("--display-scale", type=int, default=4,
                        help="Scale up the 84x84 rendered image for better visibility (e.g., 4 means 4x magnification). Default is 4.")
    parser.add_argument("--model-path", type=str, default="weights/",
                        help="Path to the directory for saving/loading model weights. Default is 'weights/'.")
    args = parser.parse_args()

    current_time = datetime.datetime.now().strftime("%A, %B %d, %Y at %I:%M:%S %p %Z")
    print(f"Current time: {current_time}")

    # --- Setup Model Saving/Loading Paths ---
    model_dir = args.model_path
    os.makedirs(model_dir, exist_ok=True)
    model_file_name = "ppo_pong_model.zip"
    model_file_path = os.path.join(model_dir, model_file_name)

    # --- Create Vectorized Training Environment (Using SubprocVecEnv for parallelization) ---
    print(f"Creating {args.num_envs} vectorized environments for training using SubprocVecEnv...")
    train_env = SubprocVecEnv([lambda: make_env() for _ in range(args.num_envs)])
    train_env = VecFrameStack(train_env, n_stack=4)
    print("Vectorized training environment created and frame stacked.")

    # --- Instantiate the EpisodeRenderCallback ---
    callback = EpisodeRenderCallback(
        render_every_n_episodes=args.render_every,
        render_speed_factor=args.render_speed,
        display_scale_factor=args.display_scale
    )

    # --- Load or Create Model ---
    model = None
    if os.path.exists(model_file_path):
        print(f"Loading existing model from {model_file_path}...")
        model = PPO.load(model_file_path, env=train_env)
        print("Model loaded successfully. Resuming training.")
    else:
        print(f"No existing model found at {model_file_path}. Creating a new model...")
        model = PPO("CnnPolicy", train_env, verbose=1)
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
        # Ensure the training environment is closed,
        # and gracefully handle EOFError that can occur during KeyboardInterrupt with SubprocVecEnv
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
