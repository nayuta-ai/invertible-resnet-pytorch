from comet_ml import Experiment

import setting

from config.parse_args import TypedArgs

def exp(args: TypedArgs) -> Experiment:
    experiment = Experiment(
        api_key=setting.API_KEY,
        project_name=setting.PROJECT_NAME,
        workspace=setting.WORKSPACE,
    )
    experiment.log_parameters(args)
    return experiment

def line_plot(experiment: Experiment, title: str, iter, val):
    experiment.log_metric(title, val, step=iter)