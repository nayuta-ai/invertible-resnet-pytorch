# Import comet_ml at the top of your file
from comet_ml import Experiment

import setting

# Create an experiment with your api key
experiment = Experiment(
    api_key=setting.API_KEY,
    project_name=setting.PROJECT_NAME,
    workspace=setting.WORKSPACE,
)

# Report multiple hyperparameters using a dictionary:
hyper_params = {
    "learning_rate": 0.5,
    "steps": 100000,
    "batch_size": 50,
}
experiment.log_parameters(hyper_params)

# Or report single hyperparameters:
hidden_layer_size = 50
experiment.log_parameter("hidden_layer_size", hidden_layer_size)

# Long any time-series metrics:
train_accuracy = 3.14
experiment.log_metric("accuracy", train_accuracy, step=0)

# Run your code and go to /
