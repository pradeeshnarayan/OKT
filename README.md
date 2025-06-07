Introduction
Knowledge tracing is the task of modeling a student's current knowledge state based on their previous learning history. This knowledge state is eventually used to predict the students' upcoming performances for a knowledge concept. The OKT model introduces a new perspective in knowledge tracing by leveraging the OBE system and Affinity mappings to quantify the inherent relationships between outcomes.

Features
Leverages the OBE concept of "Affinity mappings" to ensure the interconnectedness of knowledge concepts across the entire curriculum
Uses a Recurrent Neural Network (RNN) to track students' knowledge states based on their interactions with course and program outcomes
Enhanced with Memory Augmented Neural Networks (MANN) to analyze the specific contribution of each outcome to student knowledge progress
Achieves an impressive 89.81% AUC on a live Learning Management System, outperforming common baseline RNN models such as DKT, DKVMN, SimpleKT, and EKT
Requirements
Python 3.8+
TensorFlow 2.4+
NumPy 1.20+
Pandas 1.3+
Usage
Clone the repository: git clone https://github.com/your-repo/OKT-Model.git
Install the required dependencies: pip install -r requirements.txt
Prepare your dataset: preprocess your data according to the format specified in the data directory
Train the model: python train.py --data_path your_data_path --model_path your_model_path
Evaluate the model: python evaluate.py --data_path your_data_path --model_path your_model_path
Dataset
The dataset used for training and evaluation is not included in this repository. You will need to prepare your own dataset in the format specified in the data directory.

License
This repository is licensed under the MIT License. See the LICENSE file for details.

Citation
If you use this repository in your research, please cite the following paper:

[Your Paper Title] [Your Authors] [Your Journal/Conference] [Year]

Acknowledgments
This work was supported by [Your Funding Agency/Institution]. We would like to thank [Your Collaborators/Contributors] for their contributions to this project.
