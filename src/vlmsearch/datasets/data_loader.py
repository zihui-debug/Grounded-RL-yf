import json
import datasets
import os
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class GeneralConfig(datasets.BuilderConfig):
    def __init__(self, image_root=None, **kwargs):
        super().__init__(**kwargs)
        self.image_root = image_root

class GeneralDataset(datasets.GeneratorBasedBuilder):
    """
    A custom Dataset builder that loads a .jsonl file, unifies "id" as string,
    and returns a huggingface Dataset object.
    """

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIG_CLASS = GeneralConfig

    # Optionally define pre-configured BUILDER_CONFIGS
    BUILDER_CONFIGS = [
        GeneralConfig(
            name="default",
            version=VERSION,
            description="dataset default config"
        )
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description="Custom dataset that unifies 'id' type as string from a JSONL file.",
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "image": datasets.Value("string"),
                    "conversations": datasets.Sequence(
                        {
                            "from": datasets.Value("string"),
                            "value": datasets.Value("string"),
                        }
                    ),
                    "model": datasets.Value("string"),
                    "input_query": datasets.Value("string"),
                    "true_answer": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        data_paths = self.config.data_files  # no download_and_extract
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepaths": data_paths["train"]},
            ),
        ]

    def _generate_examples(self, filepaths):
        """Generator function that yields (key, example) pairs."""
        valid_features = {
            "id",
            "image",
            "conversations",
            "model",
            "input_query",
            "true_answer",
        }
        example_idx = 0
        for filepath in filepaths:
            logger.info(f"Processing file: {filepath}")
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line)
                    # Unify "id" as string
                    record["id"] = str(record["id"])
                    if "image" not in record:
                        logger.info(f"Skipping record because 'image' not in record")
                        continue
                    if isinstance(record["image"], str):
                        record["image"] = os.path.join(self.config.image_root, record["image"])
                    if "model" not in record:
                        record["model"] = "None"
                    for key in list(record.keys()):
                        if key not in valid_features:
                            del record[key]
                    # assume first value is query, second is answer
                    record["input_query"] = record["conversations"][0]["value"]
                    record["true_answer"] = record["conversations"][1]["value"]
                    yield example_idx, record
                    example_idx += 1