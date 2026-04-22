<h1>MNIST Trainer</h1>
<p>This is a simple Neural Network trainer that utilize the MNIST Dataset. </p>
<h3>Description</h3>
<p>Loader.py : Responsible for loading in the MNIST dataset, and preparing it for training.<br>
Models.py : Holds Neural Network models to use for training.<br>
trainer.py : The actual trainer used to train the neural network.<br>
requirements.txt : The requirements for running the trainer.<br>
/Exps : The directory where each training result is saved.<br>
/dataset : The directory where the MNIST dataset should be in.
<h3>How to use</h3>
<p>1. Make sure you have Python and the requirements installed.<br>
To install all the requirements, run <code>pip install -r requirements.txt</code><br>
2. (optional) Change configs and hyperparameters in trainer.py for optimal training.<br>
3. Run the trainer <code>python trainer.py</code><br>
4. The trained module will be saved in /Exps.
