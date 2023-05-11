"""Script for aggregating results from local trainings."""
import json
import pickle
from pathlib import Path
from typing import Any, List
import numpy as np
import requests
from feltlabs.config import AggregationConfig, parse_aggregation_args


def load_local_models(config: AggregationConfig) -> List[bytes]:
    """Load results from local algorithm (models) for aggregation.
    The model URLs are provided in custom algorithm data file. This will usually be
    URLs to models from local trainings. For testing purposes this can be local paths.

    Args:
        config: config object containing path to custom data and private key

    Returns:
        list of bytes - local results loaded as bytes
    """
    with config.custom_data_path.open("r") as f:
        conf = json.load(f)
    data_array = []
    for url in conf["model_urls"]:
        if config.download_models:
            if isinstance(url, dict):
                res = requests.get(**url)
            elif isinstance(url, str):
                res = requests.get(url)
            else:
                raise Exception(f"Invalid model URL (type {type(url)}): {url}")
            data_array.append(res.content)
        else:
            data_array.append(Path(url).read_bytes())
    return data_array


def main(config: AggregationConfig):
    """Main function executing the local result loading, aggregation and saving outputs.

    Args:
        config: training config object provided by FELT containing all paths
    """
    # Load data as numpy array
    local_models = load_local_models(config)

    # Run the aggregation algorithm
    W1 = np.vstack([pickle.loads(json.loads(data)["W1"].encode('latin-1')) for data in local_models])
    W2 = np.mean([pickle.loads(json.loads(data)["W2"].encode('latin-1')) for data in local_models], axis=0)

    # Get final output values
    model_bytes = json.dumps({"W1": pickle.dumps(W1).decode('latin-1'), "W2": pickle.dumps(W2).decode('latin-1')})

    # Save models into output folder. You have to name output file as "model"
    with open(config.output_folder / "model", "w") as f:
        f.write(model_bytes)

    print("Training finished.")


if __name__ == "__main__":
    # Get config - we recommend using config parser provided by FELT Labs
    # It automatically provides all input and output paths
    config = parse_aggregation_args()
    main(config)
