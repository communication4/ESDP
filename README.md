# Deep Dialogue

This is a dialogue system that uses deep learning models. It is a research project.

## How to run

The main script to run the dialogue system is `run1.py`.

### Dependencies

The script requires the following Python libraries:
- argparse
- json
- copy
- os
- _pickle
- numpy
- random
- time
- logging
- torch

You can install them using pip:
```bash
pip install numpy torch
```
(Note: Other libraries like argparse, json, copy, os, _pickle, random, time, logging are part of the standard Python library and do not need separate installation.)

### Arguments

The `run1.py` script accepts various command-line arguments to configure the simulation. Here are some of the key arguments:

- `--dict_path`: Path to the dictionary file (e.g., `./deep_dialog/data_taxi1/slot_dict.v1_.json`).
- `--movie_kb_path`: Path to the movie knowledge base file (e.g., `./deep_dialog/data_taxi1/taxi.kb.2k.v1_.json`).
- `--act_set`: Path to the dialog act set file (e.g., `./deep_dialog/data_taxi1/dia_acts.txt`).
- `--slot_set`: Path to the slot set file (e.g., `./deep_dialog/data_taxi1/taxi_slots.txt`).
- `--goal_file_path`: Path to the user goals file (e.g., `./deep_dialog/data_taxi1/user_goals_first.part.taxi.v4_.json`).
- `--diaact_nl_pairs`: Path to pre-defined dialog act and NL pairs (e.g., `./deep_dialog/data/dia_act_nl_pairs.v6.json`).
- `--max_turn`: Maximum length of each dialog (default: 40).
- `--episodes`: Total number of episodes to run (default: 300).
- `--slot_err_prob`: Slot error probability (default: 0.05).
- `--intent_err_prob`: Intent error probability (default: 0.00).
- `--agt`: Agent ID to use (default: 13).
    - 0: Command line input
    - 1: InformAgent
    - 2: RequestAllAgent
    - 3: RandomAgent
    - 4: EchoAgent
    - 5: RequestBasicsAgent
    - 9: AgentDQN (movie domain)
    - 13: AgentDQN (taxi domain)
    - 14: AgentDQN (restaurant domain)
- `--usr`: User simulator ID to use (default: 2).
    - 0: Real user (not implemented)
    - 1: RuleSimulator (movie domain)
    - 2: RuleTaxiSimulator (taxi domain)
    - 3: RuleRestSimulator (restaurant domain)
- `--epsilon`: Epsilon for epsilon-greedy agent policies (default: 0).
- `--nlg_model_path`: Path to the NLG model file.
- `--nlu_model_path`: Path to the NLU model file.
- `--run_mode`: Run mode for NLU/NLG.
    - 0: Default NL
    - 1: Dialog act
    - 2: Both
- `--auto_suggest`: Enable auto-suggestion (0 or 1, default: 0).
- `--trained_model_path`: Path to a pre-trained model. If None, the agent will be trained.
- `--write_model_dir`: Directory to save trained models (default: `./deep_dialog/checkpoints/taxi_esdp_07_20/`).
- `--learning_phase`: Phase of learning ('train', 'test', 'all', default: 'all').

### Example Usage

To run the simulation with default parameters for the taxi domain:
```bash
python run1.py --agt 13 --usr 2
```

To run with a specific goal file and save the model to a custom directory:
```bash
python run1.py --goal_file_path ./deep_dialog/data_restaurant/user_goals_first.part.restaurant.v4_.json --agt 14 --usr 3 --write_model_dir ./my_checkpoints/restaurant_model/
```

## Dataset

The dialogue system uses various datasets for training and simulation. For detailed information on the datasets used, including download links and the required file structure, please refer to the dataset-specific README file located at:

[deep_dialogue/dataset/Readme.md](./deep_dialogue/dataset/Readme.md)

## Models

The Deep Dialogue system leverages several machine learning models to understand user input, generate responses, and make decisions. The key types of models used are:

### Natural Language Understanding (NLU)
The NLU component is responsible for interpreting the user's text input. It processes the natural language and converts it into a structured representation that the dialogue manager can understand. This typically involves tasks like intent recognition and slot filling.
The script `run1.py` allows specifying an NLU model path via the `--nlu_model_path` argument.

### Natural Language Generation (NLG)
The NLG component takes the structured representation of the agent's response (dialogue acts) and converts it into natural language text to be presented to the user. This aims to make the system's responses fluent and human-like.
The script `run1.py` allows specifying an NLG model path via the `--nlg_model_path` argument.

### Deep Q-Network (DQN)
DQN models are used for the dialogue policy, which is responsible for deciding the agent's next action based on the current dialogue state. DQN is a reinforcement learning algorithm that learns an optimal policy by interacting with the user (or a user simulator) and receiving rewards. This allows the agent to learn complex behaviors and optimize for long-term success in the conversation.
Different DQN agents can be selected in `run1.py` using the `--agt` argument (e.g., 9 for movie domain, 13 for taxi, 14 for restaurant). The script also provides various arguments for configuring the DQN agent, such as `--experience_replay_pool_size`, `--dqn_hidden_size`, `--batch_size`, and `--gamma`. Pre-trained DQN models can be loaded using `--trained_model_path`, and new models can be saved to the directory specified by `--write_model_dir`.
