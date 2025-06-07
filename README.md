<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
</head>
<body>

  <h1>OKT: Outcome-Based Knowledge Tracing</h1>

  <h2>Introduction</h2>
  <p>
    Knowledge tracing is the task of modeling a student's current knowledge state based on their previous learning history.
    This knowledge state is eventually used to predict the student's upcoming performance on a knowledge concept.
    The OKT model introduces a new perspective in knowledge tracing by leveraging the OBE system and Affinity mappings
    to quantify the inherent relationships between outcomes.
  </p>

  <h2>Features</h2>
  <ul>
    <li>Leverages the OBE concept of <strong>Affinity mappings</strong> to ensure the interconnectedness of knowledge concepts across the entire curriculum.</li>
    <li>Uses a <strong>Recurrent Neural Network (RNN)</strong> to track students' knowledge states based on their interactions with course and program outcomes.</li>
    <li>Enhanced with <strong>Memory Augmented Neural Networks (MANN)</strong> to analyze the specific contribution of each outcome to student knowledge progress.</li>
    <li>Achieves an impressive <strong>89.81% AUC</strong> on a live Learning Management System, outperforming baseline RNN models such as DKT, DKVMN, SimpleKT, and EKT.</li>
  </ul>

  <h2>Requirements</h2>
  <ul>
    <li>Python 3.8+</li>
    <li>TensorFlow 2.4+</li>
    <li>NumPy 1.20+</li>
    <li>Pandas 1.3+</li>
  </ul>

  <h2>Usage</h2>
  <ol>
    <li>Clone the repository: <code>git clone https://github.com/your-repo/OKT-Model.git</code></li>
    <li>Install the required dependencies: <code>pip install -r requirements.txt</code></li>
    <li>Prepare your dataset: preprocess your data according to the format specified in the <code>data</code> directory.</li>
    <li>Train the model: <code>python train.py --data_path your_data_path --model_path your_model_path</code></li>
    <li>Evaluate the model: <code>python evaluate.py --data_path your_data_path --model_path your_model_path</code></li>
  </ol>

  <h2>Dataset</h2>
  <p>
    The dataset used for training and evaluation is not included in this repository.
    You will need to prepare your own dataset in the format specified in the <code>data</code> directory.
  </p>
  <!--
  <h2>License</h2>
  <p>
    This repository is licensed under the MIT License.
    See the <code>LICENSE</code> file for details.
  </p>

  <h2>Citation</h2>
  <p>
    If you use this repository in your research, please cite the following paper:<br>
    <strong>[Your Paper Title]</strong><br>
    [Your Authors]<br>
    [Your Journal/Conference], [Year]
  </p>

  <h2>Acknowledgments</h2>
  <p>
    This work was supported by [Your Funding Agency/Institution].<br>
    We would like to thank [Your Collaborators/Contributors] for their contributions to this project.
  </p>
  -->

</body>
</html>
