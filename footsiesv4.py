import numpy as np
import pyautogui
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
import pydirectinput as input
import pygetwindow as gw
import os
import cv2
import datetime
import win32gui
from collections import deque
import torchvision.transforms as transforms
from PIL import ImageGrab
import matplotlib
import matplotlib.pyplot as plt
import threading
from PIL import Image

# Constants for actions
ACTION_LEFT_PRESS = 0
ACTION_RIGHT_PRESS = 1
ACTION_ATTACK_HOLD = 2
ACTION_ATTACK_FINISHER = 3

# Game state reward features
P1_SHIELDS = 3
P2_SHIELDS = 3
P1_WINS = 0
P2_WINS = 0

# Game window confid
SCREEN_X = 128
SCREEN_Y = 128

FRAME_COUNT = 4
FPS = 60

input_shape = (FRAME_COUNT, SCREEN_X, SCREEN_Y)  # Greyscale/Colored resized images of size 600x338 pixels
num_actions = 4  # Assuming 3 actions: left, right, attack, attack_hold

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to perform action in the game p1
def p1_perform_action(action):
    if gw.getActiveWindow().title == "FOOTSIES":
        if action == ACTION_LEFT_PRESS:
            input.press(['a'])
        if action == ACTION_RIGHT_PRESS:
            input.press(['d'])
        if action == ACTION_ATTACK_FINISHER:
            input.press(['space'])
        if action == ACTION_ATTACK_HOLD:
            input.keyDown('space')

# Function to perform game reset
def reset_game():
    # Simulate pressing escape, up, space, space, space keys using input
    if gw.getActiveWindow().title == "FOOTSIES":
        input.press('esc')
        input.press('up')
        input.press('space')
        input.press('space')
        input.press('space')
        input.press('space')
    time.sleep(1)
    global P1_WINS, P2_WINS, P1_SHIELDS, P2_SHIELDS
    P1_SHIELDS = 3
    P2_SHIELDS = 3
    P1_WINS = 0
    P2_WINS = 0


# Function to capture screen of the game window
def capture_screen():
    # Find the "FOOTSIES" window
    footsies_window = gw.getWindowsWithTitle("FOOTSIES")

    # Check if the window is found
    if footsies_window:
        footsies_window = footsies_window[0]  # Assuming there's only one window with that title
        # Activate the window (bring it to the front)
        footsies_window.activate()
        # Get the position and size of the window
        x, y, width, height = footsies_window.left, footsies_window.top, footsies_window.width, footsies_window.height
        # Capture the screenshot of the window
        screenshot = np.array(ImageGrab.grab(bbox=(x, y, x + width, y + height)))
        # Convert the screenshot to grayscale
        screenshot_gray = cv2.cvtColor(np.array(screenshot), cv2.COLOR_BGR2GRAY)
        # Resize the image to a fixed size (e.g., 128x128)
        resized_image = cv2.resize(screenshot_gray, (SCREEN_X, SCREEN_Y))
        # Normalize the image (optional)
        normalized_image = resized_image.astype(np.float32) / 255.0  # Assuming pixel values are in the range [0, 255]
        # Convert the image into a tensor
        transform = transforms.Compose([transforms.ToTensor()])
        tensor_image = transform(normalized_image).to(device)

        return tensor_image
    else:
        print("Window not found.")
        return None
        

def stack_frames(frames):
    return torch.squeeze(torch.stack(frames), dim=1).to(device)


def frame_stack():
    frames_buffer = []
    for frames in range(FRAME_COUNT):
        frame = capture_screen()
        time.sleep(1 / FPS)  # 1/60 frames per millisecond
        frames_buffer.append(frame)

    stacked_frames = stack_frames(frames_buffer)
    return stacked_frames

# Function to detect round win or loss
def detect_round_outcome(screen):
    # Define the coordinates for detecting round wins for player 1
    p1_wins = [(37, 104), (29, 104), (21, 104)]
    # Define the coordinates for detecting round wins for player 2
    p2_wins = [(90, 104), (98, 104), (106, 104)]

    # Check pixel values at specified coordinates for player 1 round wins
    p1_round_wins = sum(1 if screen[FRAME_COUNT - 1, y, x] == 0 else 0 for x, y in p1_wins)
    # Check pixel values at specified coordinates for player 2 round wins
    p2_round_wins = sum(1 if screen[FRAME_COUNT - 1, y, x] == 0 else 0 for x, y in p2_wins)

    return p1_round_wins, p2_round_wins


# Function to extract player shield counts
def detect_shields(screen):
    # Define the coordinates for detecting player 1 shields
    p1_shields = [(32, 24), (22, 24), (12, 24)]
    # Define the coordinates for detecting player 2 shields
    p2_shields = [(95, 24), (105, 24), (115, 24)]

    # Check pixel values at specified coordinates for player 1 shields
    p1_shield_count = sum(1 if screen[FRAME_COUNT - 1, y, x] == 0 else 0 for x, y in p1_shields)
    # Check pixel values at specified coordinates for player 2 shields
    p2_shield_count = sum(1 if screen[FRAME_COUNT - 1, y, x] == 0 else 0 for x, y in p2_shields)

    return p1_shield_count, p2_shield_count


# Function to extract game state variables
def calc_rewards(screen):
    global P1_WINS, P2_WINS, P1_SHIELDS, P2_SHIELDS
    # Initialize reward
    reward_p1 = 0
    reward_p2 = 0

    # Detect player wins
    p1_win_count, p2_win_count = detect_round_outcome(screen)
    # Reward for shield changes
    reward_p1 += (p1_win_count - P1_WINS) * 1000
    reward_p1 -= (p2_win_count - P2_WINS) * 1000
    reward_p1 -= 1

    reward_p2 += (p2_win_count - P2_WINS) * 1000
    reward_p2 -= (p1_win_count - P1_WINS) * 1000
    reward_p2 -= 1

    # Detect player shields
    p1_shield_count, p2_shield_count = detect_shields(screen)

    # Reward for shield changes
    reward_p1 -= (P1_SHIELDS - p1_shield_count) * 100  # Penalty for losing shields
    reward_p1 += (P2_SHIELDS - p2_shield_count) * 100  # Reward for taking shields from opponent

    reward_p2 -= (P2_SHIELDS - p2_shield_count) * 100  # Penalty for losing shields
    reward_p2 += (P1_SHIELDS - p1_shield_count) * 100  # Reward for taking shields from opponent

    # Update global wins and shield counts

    P1_WINS = p1_win_count
    P2_WINS = p2_win_count
    P1_SHIELDS = p1_shield_count
    P2_SHIELDS = p2_shield_count


    terminated = False
    if p1_win_count >= 1 or p2_win_count >= 1:
        terminated = True

    return reward_p1, reward_p2, terminated


ROUND_DURATION = datetime.timedelta(minutes=1)


# Function to update plots
def update_plots(cumulative_rewards_p1):
    # Turn on interactive mode
    plt.ion()
    # Create a smaller figure
    plt.figure(1, figsize=(3, 2))  # Adjust the size as needed (6x4 inches in this example)
    plt.clf()

    # Plot cumulative rewards over episodes
    plt.plot(cumulative_rewards_p1)
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('Agent Performance Over Episodes')

    # # Display the plot
    plt.draw()
    # plt.show(block=False)
    plt.pause(0.001)


def save_plots(data, x_label, y_label, title, filename):
    # # Disable interactive mode after training is done
    # plt.ioff()

    # Plot rewards over episodes
    plt.figure()
    plt.plot(data)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(filename)  # Save the plot to a file
    plt.close()

    # plt.show()


def get_current_time():
    return datetime.datetime.now()


# Define the neural network architecture (CNN)
class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4).to(device)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2).to(device)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1).to(device)
        conv_out_size = self._get_conv_output(input_shape)
        self.fc1 = nn.Linear(conv_out_size, 512).to(device)
        self.fc2 = nn.Linear(512, num_actions).to(device)
    # Get the shape of the output from the CNN layers
    def _get_conv_output(self, shape):
        o = self.conv1(torch.zeros(1, *shape).to(device))
        o = self.conv2(o)
        o = self.conv3(o)
        return np.prod(o.size())

    def forward(self, x):
        # Apply ReLU activation after each convolutional layer
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        # Flatten the output before passing it to fully connected layers
        x = x.view(x.size(0), -1)
        # Pass through fully connected layers with ReLU activation
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Define the deep Q-learning agent
class DQNAgent:
    def __init__(self, p_input_shape, p_num_actions, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.9995,
                 epsilon_min=0.01, memory_capacity=10000, batch_size=64, episodes=1000,  # batch size larger for GPU
                 target_model_update_frequency=5):
        self.input_shape = p_input_shape
        self.num_actions = p_num_actions
        self.memory_capacity = memory_capacity
        self.memory = deque(maxlen=self.memory_capacity)
        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.episodes = episodes
        self.target_model_update_frequency = target_model_update_frequency
        self.model = DQN(p_input_shape, p_num_actions).to(device)
        self.target_model = DQN(p_input_shape, p_num_actions).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_function = nn.MSELoss()
    
    # Function for agent to select action
    def select_action(self, state):
        # Choose action (epsilon-greedy)
        if np.random.rand() <= self.epsilon:
            action = np.random.choice(self.num_actions)
        else:
            # print("Book move")
            with torch.no_grad():
                state = state.unsqueeze(0).to(device)  # idk
                q_values = self.model(state)
                action = q_values.argmax().item()
        return action
    
    #   Add the s, a, s', r into replay buffer
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        # mini-batch experiences from meory
        states, actions, rewards, next_states, dones = map(torch.stack, zip(*batch))

        q_values = self.model(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_model(next_states).max(1)[0].detach()

        targets_q_values = rewards + (1 - dones.float()) * self.gamma * next_q_values

        # loss = self.loss_function(q_values.squeeze(), targets_q_values.detach())
        loss = self.loss_function(q_values, targets_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        # print("Optimizing...")
        self.optimizer.step()
    
    # Update toarget model with current model
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())


# Main training loop
def train():
    print("Start Training...")
    # Initialize environment
    screen = frame_stack()  # capture_screen()
    agent_1 = DQNAgent(input_shape, num_actions)

    # Define saving frequency
    nn_save_frequency = 500  # Save weights every 500 episodes
    if screen is not None:
        reset_game()
        rewards_over_episodes_p1 = [0]
        cumulative_reward_p1 = 0
        cumulative_rewards_p1 = [cumulative_reward_p1]
        # Start live performance plot
        update_plots(cumulative_rewards_p1)
        for episode in range(agent_1.episodes):
            current_state = screen = frame_stack()  # capture_screen()
            if screen is not None:
                total_reward_p1 = 0
                reward_p1 = None
                start_time = get_current_time()  # Record start time of episode
                round_time = 0
                terminated = False
                action_count = 0
                while not terminated:
                    # Extract game state variables

                    action = agent_1.select_action(current_state)

                    # Perform action
                    p1_perform_action(action)
                   
                    # time.sleep(0.0001)  # Adjust delay between actions as needed
                    action_count += 1

                    # Capture next screen
                    next_state = frame_stack()  # capture_screen()

                    # Detect round outcome
                    reward_p1, reward_p2, terminated = calc_rewards(next_state)

                    # Check if 2 minutes have elapsed
                    if get_current_time() - start_time >= ROUND_DURATION:
                        terminated = True

                    action = torch.tensor(action).to(device)
                    reward_p1 = torch.tensor(reward_p1).to(device)
                    terminated = torch.tensor(terminated).to(device)

                    agent_1.remember(current_state, action, reward_p1, next_state, terminated)

                    # Replay and train the agent
                    agent_1.replay()

                    # Update total reward
                    total_reward_p1 += reward_p1
                    current_state = next_state
                    # Check if episode is done
                    if terminated:
                        round_time = get_current_time() - start_time
                        break
                total_reward_p1 = int(total_reward_p1)

                rewards_over_episodes_p1.append(total_reward_p1)
                cumulative_reward_p1 += total_reward_p1
                cumulative_rewards_p1.append(cumulative_reward_p1)
                if agent_1.epsilon > agent_1.epsilon_min:
                    agent_1.epsilon *= agent_1.epsilon_decay

                # Check if it's time to save the model weights
                if (episode + 1) % nn_save_frequency == 0 or episode + 1 == 100:
                    # Save model weights
                    folder_path = "F_model_weights"
                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)
                    image_path = os.path.join(folder_path, f"model_weights_episode_{episode + 1}.pt")
                    torch.save(agent_1.model.state_dict(), image_path)

                # Update target network periodically
                if episode % agent_1.target_model_update_frequency == 0:
                    agent_1.update_target_model()
                    # agent_2.update_target_model()
                # print("GG")
                print(
                    f"Episode: {episode + 1}/{agent_1.episodes}, Total Reward P1: {total_reward_p1}"
                    f", Round Time: {round_time}")

                # Update plots
                update_plots(cumulative_rewards_p1)

                reset_game()  # Reset the game environment

        # Plot rewards over episodes
        save_plots(rewards_over_episodes_p1, 'Episode', 'Total Reward', 'Agent Performance Over Episodes',
                   'footsies_total_reward_agent_1.png')
        save_plots(cumulative_rewards_p1, 'Episode', 'Cumulative Reward', 'Agent Performance Over Episodes',
                   'footsies_cumulative_rewards_agent_1.png')

    print("I KNOW KUNG FU.")


def demo():
    screen = frame_stack()  # capture_screen()
    agent47 = DQNAgent(input_shape, num_actions)  # Recreate the model

    image_path = os.path.join("F_model_weights", f"model_weights_episode_10 00.pt")
    # Load the saved weights
    agent47.model.load_state_dict(torch.load(image_path))
    agent47.model.eval()  # Set the model to evaluation mode

    if screen is not None:
        reset_game()
        # Main game loop
        while True:
            # Capture the current game screen
            current_state = screen = frame_stack()  # capture_screen()

            if screen is not None:
                # Select action using the trained model
                action = agent47.select_action(current_state)

                # Perform the action in the game environment
                p1_perform_action(action)
                # time.sleep(0.1)  # Adjust delay as needed

                reward_p1, reward_p2, terminated = calc_rewards(frame_stack())  # capture_screen())
                if terminated:
                    reset_game()


if __name__ == "__main__":
    cuda = torch.cuda.is_available()
    print("CUDA Status: ", cuda)
    if cuda:
        print("Current Device#: ", torch.cuda.current_device())
        print("GPU: ", torch.cuda.get_device_name(torch.cuda.current_device()))
    handle = win32gui.FindWindow(0, "FOOTSIES")
    win32gui.SetForegroundWindow(handle)
    time.sleep(0.1)

    # train()
    demo()
