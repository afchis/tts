

def get_model(params):
    model_list = ["fast_speech2", "tacotron2", "embedder"]

    assert params["network_name"] in model_list, (
        "Wrong network name in config. Please choice:", model_list
    )

    if params["network_name"] == "fast_speech2":
        from .tts.fast_speech2 import FastSpeech2
        model = FastSpeech2()
    elif params["network_name"] == "tacotron2":
        from .tts.tacotron2 import Tacotron2
        model = Tacotron2()
    elif params["network_name"] == "embedder":
        if params["dataset"]["data_type"] == "mel":
            from .embedders.ebmedder import EmbedderMel
            model = EmbedderMel()
        elif params["dataset"]["data_type"] == "wav":
            from .embedders.ebmedder import EmbedderWave
            model = EmbedderWave()
    return model

