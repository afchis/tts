import torch.optim as optim


def get_optim(trainer, params):
    if params["network_name"] == "embedder":
        parameters = list(trainer.model.parameters()) + list(trainer.criterions.body.parameters())
    else:
        parameters = trainer.model.parameters()
    optim_list = ["SGD", "Adam"]
    assert params["solver"] in optim_list, ("Wrong optim name in config. Please choice:", optim_list)
    if params["solver"] == "SGD":
        return optim.SGD(parameters, lr=params["learning_rate"])
    elif params["solver"] == "Adam":
        return optim.Adam(parameters,
                          lr=params["learning_rate"],
                          betas=(params["beta1"], params["beta2"]),
                          eps=params["eps"],
                          weight_decay=1e-06)

