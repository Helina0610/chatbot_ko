import os
from pyannote.audio import Pipeline
from pathlib import Path

token = "토큰"

def load_pipeline_from_pretrained(path_to_config: str | Path) -> Pipeline:
    path_to_config = Path(path_to_config)

    print(f"Loading pyannote pipeline from {path_to_config}...")
    # the paths in the config are relative to the current working directory
    # so we need to change the working directory to the model path
    # and then change it back

    cwd = Path.cwd().resolve()  # store current working directory

    # first .parent is the folder of the config, second .parent is the folder containing the 'models' folder
    cd_to = path_to_config.parent.parent.resolve()

    print(f"Changing working directory to {cd_to}")
    os.chdir(cd_to)
    pipeline = Pipeline.from_pretrained(path_to_config, use_auth_token=token)

    print(f"Changing working directory back to {cwd}")
    os.chdir(cwd)

    return pipeline

PATH_TO_CONFIG = "./models/pyannote_diarization_config.yaml"
pipeline = load_pipeline_from_pretrained(PATH_TO_CONFIG)

# run the pipeline on an audio file
diarization = pipeline("./data/chat.wav", min_speakers=2, max_speakers=5)
print(diarization)

## dump the diarization output to disk using RTTM format
#with open("./data/chat.rttm", "w") as rttm:
#    diarization.write_rttm(rttm)