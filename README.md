# LatentScope
Source Code and Dataset B for KDD 24 Paper "[Microservice Root Cause Analysis With Limited Observability Through Intervention Recognition in the Latent Space](https://netman.aiops.org/wp-content/uploads/2024/07/LatentScope_CameraReady_final.pdf)".

## Installation
- Download the repo (Dataset B included).
- Install the dependencies in `requirements.txt` with `Python >= 3.9, < 3.11`

## Usage

Run `python main.py [--cpus NUM_WORKERS]`.

Results will be saved under the `results/`.

## Use your own datasets
- According to the format in `data/dataset_b/data/1/metrics.json` and `data/dataset_b/data/1/rccs.json`, organize your metric data and RCC list, RCC edges, and connections between RCCs and Metrics and put them under `data/[dataset_name]/data/[case_name]/`. 
- Specify inter-service dependencies in the dataset's `labels/service_deps.json` to establish causal relationships at the metric layer. 
- Refer to the format in `labels/label.json` to set the trigger, root cause, and trigger time for each case. 
- If necessary, you can add metric category determination code for your dataset in `utils.py` (this category is used to partition meta_variable for metrics during graph construction, for more details, refer to CIRCA).
- Run `python main.py -d [dataset_name] [--cpus NUM_WORKERS]`.

## Contributing
Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[Apache License Version 2.0](https://www.apache.org/licenses/LICENSE-2.0)

## Refenrence
- CIRCA: [CIRCA](https://github.com/NetManAIOps/CIRCA.git).
