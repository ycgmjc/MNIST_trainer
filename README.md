<h1>MNIST Trainer</h1>
<p>This is a simple Neural Network trainer that utilizes the MNIST Dataset.</p>
<p>It includes a simple Gradio based, and FastAPI based web interfaces to check the trained model.</p>

<h3>Description</h3>
<ul>
    <li><strong>Loader.py</strong>: Responsible for loading in the MNIST dataset and preparing it for training.</li>
    <li><strong>Models.py</strong>: Holds Neural Network models to use for training.</li>
    <li><strong>trainer.py</strong>: The actual trainer used to train the neural network.</li>
    <li><strong>requirements.txt</strong>: The requirements for running the trainer.</li>
    <li><strong>/Exps</strong>: The directory where each training result is saved.</li>
    <li><strong>/dataset</strong>: The directory where the MNIST dataset should be in.</li>
    <li><strong>/static</strong>: The directory containing the frontend web file (i.e. <code>index.html</code>) for the drawing interface.</li>
</ul>

<h2>How to use</h2>
<h3>Training</h3>
<ol>
    <li>Make sure you have Python and the requirements installed.<br>
        To install all requirements, run <code>pip install -r requirements.txt</code></li>
    <li>(Optional) Change configs and hyperparameters in <code>trainer.py</code>.</li>
    <li>Run the trainer: <code>python trainer.py</code></li>
    <li>The trained module will be saved in <code>/Exps</code>.</li>
</ol>
<h3>Testing</h3>
<ol>
    <ui><code>app.py</code></ui>
    <ol>
        <li>To test the trained model, run the web UI: <code>python app.py</code></li>
        <li>Open your browser and navigate to <code>http://127.0.0.1:8000</code> to draw digits and see the model's predictions.</li>
    </ol>
    <ui><code>server.py</code></ui>
    <ol>
        <li>To test the trained model, start the web server: <code>python server.py</code></li>
        <li>Open your browser and navigate to <code>http://127.0.0.1:8000</code> to draw digits and see the model's predictions.</li>
    </ol>
</ol>


