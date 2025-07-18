import ale_py
import shimmy

import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, SubprocVecEnv
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env # Import make_vec_env
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
    def __init__(self, render_every_n_episodes: int = 100, render_speed_factor: int = 10,
                 display_scale_factor: int = 4, verbose: int = 1, n_stack: int = 4):
        super().__init__(verbose)
        self.render_every_n_episodes = render_every_n_episodes
        self.render_speed_factor = max(1, render_speed_factor) # Ensure it's at least 1
        self.display_scale_factor = max(1, display_scale_factor) # Ensure scale factor is at least 1
        self.episode_counter = 0
        self.render_env = None  # Dedicated environment for rendering
        self.window_name = "Pong Accelerated Render" # Name for the OpenCV window
        self.agent_view_shape = (84, 84, 3) # The shape the agent sees (after AtariWrapper)
        self.n_stack = n_stack # Store n_stack to pass to make_vec_env

    def _on_training_start(self) -> None:
        """
        Called once at the beginning of training.
        Initializes the dedicated rendering environment.
        This environment is also frame-stacked to match the model's input expectations.
        """
        if self.render_every_n_episodes > 0:
            print("Creating dedicated rendering environment...")
            # Use make_vec_env for the rendering environment as well
            # n_envs=1 for a single rendering environment
            # vec_env_cls=DummyVecEnv is appropriate for a single env
            self.render_env = make_vec_env(
                env_id="ALE/Pong-v5",
                n_envs=1,
                seed=0, # Use a fixed seed for consistent rendering
                vec_env_cls=DummyVecEnv,
                wrapper_class=AtariWrapper,
                env_kwargs={"render_mode": "rgb_array"}, # Pass render_mode to gym.make
                # frame_stack=self.n_stack # Apply frame stacking
            )
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
        Also prints individual episode rewards.
        """
        if self.render_every_n_episodes <= 0:
            return True

        if self.locals.get("dones") is None or self.locals.get("infos") is None:
            return True

        for i, done in enumerate(self.locals["dones"]):
            if done:
                self.episode_counter += 1
                info = self.locals["infos"][i]
                # Check if the 'episode' key exists in the info dict
                # This key is added by VecEnv wrappers for completed episodes
                if "episode" in info:
                    episode_reward = info["episode"]["r"]
                    episode_length = info["episode"]["l"]
                    print(f"Episode {self.episode_counter} (Env {i}) Finished: Reward={episode_reward:.2f}, Length={episode_length}")

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

        # self.render_env.reset() returns (observation, info) tuple for VecEnvs in Gymnasium style
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
                frame = self.render_env.render() # Returns a batch of images (n_envs, H, W, C)

                # Since render_env is n_envs=1, take the first (and only) frame
                # frame = frame[0]

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

            if done_array[0]: # Check done_array for the single environment
                done = True

        # Ensure final frame is rendered if not already (and if not quit by user)
        if not ('key' in locals() and key == ord('q')) and (frame_counter % self.render_speed_factor != 0):
             try:
                 final_frame = self.render_env.render()
                 raw_game_shape = (210, 160, 3)
                 if isinstance(final_frame, np.ndarray) and final_frame.shape == (1, *raw_game_shape): # Check for batch dimension
                     final_frame = final_frame[0] # Extract the single frame
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

    def _on_rollout_end(self) -> None:
        print(f"DEBUG: _on_rollout_end called at {self.num_timesteps} timesteps, {self.model.ep_info_buffer}.") # Added debug print
        """
        Called after each collection of experience (rollout).
        This is where the model has fresh episode statistics.
        We'll print the mean reward from recently completed episodes.
        """
        # The model's ep_info_buffer holds info for episodes completed during the last rollout
        # It's a deque, so len(self.model.ep_info_buffer) gives the number of recent episodes
        assert(self.model.ep_info_buffer is not None)
        if len(self.model.ep_info_buffer) > 0:
            mean_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
            mean_length = np.mean([ep_info["l"] for ep_info in self.model.ep_info_buffer])
            print(f"\n--- Rollout Summary ({len(self.model.ep_info_buffer)} episodes) --- Mean Reward: {mean_reward:.2f}, Mean Length: {mean_length:.1f}")
            # The SB3 internal logger also records these to TensorBoard, so you don't
            # necessarily need to call self.logger.record here again unless you
            # want to customize the logging frequency or tag names.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a PPO agent on Pong-v5.")
    parser.add_argument("--timesteps", type=int, default=100000,
                        help="Total number of timesteps to train the model.")
    parser.add_argument("--n-stack", type=int, default=4,
                        help="Number of frames to stack for the observation (default: 4).")
    parser.add_argument("--num-envs", type=int, default=os.cpu_count() or 4,
                        help="Number of parallel environments for vectorized training (defaults to number of CPU cores).")
    parser.add_argument("--learning-rate", type=float, default=0.00025, # Changed default to SB3's PPO default
                        help="Learning rate for the optimizer (default: 0.00025).")
    parser.add_argument("--n-steps", type=int, default=4096, # Changed default to a more common Atari value
                        help="Number of steps (timesteps) to run for each environment per update rollout (default: 4096).")
    parser.add_argument("--render-every", type=int, default=1,
                        help="Render one episode every N episodes (0 to disable).")
    parser.add_argument("--render-speed", type=int, default=10,
                        help="Accelerate rendered episodes by skipping frames (e.g., 10 means 10x faster visuals). Default is 10.")
    parser.add_argument("--display-scale", type=int, default=4,
                        help="Scale up the 84x84 rendered image for better visibility (e.g., 4 means 4x magnification). Default is 4.")
    parser.add_argument("--model-path", type=str, default="weights/",
                        help="Path to the directory for saving/loading model weights. Default is 'weights/'.")
    parser.add_argument("--tb-log-dir", type=str, default="tensorboard_logs",
                        help="Path to the directory for TensorBoard logs. Default is 'tensorboard_logs/'.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="The device to use for training the model")
    args = parser.parse_args()

    current_time = datetime.datetime.now().strftime("%A, %B %d, %Y at %I:%M:%S %p %Z")
    print(f"Current time: {current_time}")

    # --- Setup Model Saving/Loading Paths ---
    model_dir = args.model_path
    os.makedirs(model_dir, exist_ok=True)
    model_file_name = "ppo_pong_model.zip"
    model_file_path = os.path.join(model_dir, model_file_name)

    # --- Create Vectorized Training Environment (Using make_vec_env for parallelization) ---
    print(f"Creating {args.num_envs} vectorized environments for training using make_vec_env...")
    # Using make_vec_env simplifies the creation of multiple environments with wrappers and frame stacking
    train_env = make_vec_env(
        env_id="ALE/Pong-v5",
        n_envs=args.num_envs,
        seed=0, # Fixed seed for reproducibility
        vec_env_cls=SubprocVecEnv, # Use SubprocVecEnv for parallelization
        wrapper_class=AtariWrapper, # Apply AtariWrapper to each environment
    )
    print("Vectorized training environment created and frame stacked.")

    # --- Instantiate the EpisodeRenderCallback ---
    callback = EpisodeRenderCallback(
        render_every_n_episodes=args.render_every,
        render_speed_factor=args.render_speed,
        display_scale_factor=args.display_scale,
        n_stack=args.n_stack, # Pass n_stack to callback so it can configure its render_env correctly
    )

    # --- Load or Create Model ---
    model = None
    if os.path.exists(model_file_path):
        print(f"Loading existing model from {model_file_path}...")
        # Make sure n_steps, learning_rate, n_stack match the loaded model's config if not specified in load
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
        if args.render_every > 0:
            model.learn(total_timesteps=args.timesteps, callback=callback)
        else:
            model.learn(total_timesteps=args.timesteps)
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
