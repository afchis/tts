import torch.optim.lr_scheduler as lr_scheduler


def get_scheduler(optimizer, params):
    if params.get("policy") == "StepLR":
        return lr_scheduler.StepLR(optimizer, step_size=params.get("step_size"), gamma=params.get("gamma"))
    elif params.get("policy") == "MultiStepLR":
        return lr_scheduler.MultiStepLR(optimizer, milestones=params.get("milestones"), gamma=params.get("gamma"))
    elif params.get("policy") == "ExponentialLR":
        return lr_scheduler.ExponentialLR(optimizer, gamma=params.get("gamma"))
