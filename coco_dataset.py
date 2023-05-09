# coding=utf-8
# Copyright 2022 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""COCO"""
import json
import os
from pathlib import Path

import datasets


_CITATION = """
@article{DBLP:journals/corr/LinMBHPRDZ14,
  author    = {Tsung{-}Yi Lin and
               Michael Maire and
               Serge J. Belongie and
               Lubomir D. Bourdev and
               Ross B. Girshick and
               James Hays and
               Pietro Perona and
               Deva Ramanan and
               Piotr Doll{\'{a}}r and
               C. Lawrence Zitnick},
  title     = {Microsoft {COCO:} Common Objects in Context},
  journal   = {CoRR},
  volume    = {abs/1405.0312},
  year      = {2014},
  url       = {http://arxiv.org/abs/1405.0312},
  eprinttype = {arXiv},
  eprint    = {1405.0312},
  timestamp = {Mon, 13 Aug 2018 16:48:13 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/LinMBHPRDZ14.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""

_DESCRIPTION = """
MS COCO is a large-scale object detection, segmentation, and captioning dataset.
COCO has several features: Object segmentation, Recognition in context, Superpixel stuff segmentation, 330K images (>200K labeled), 1.5 million object instances, 80 object categories, 91 stuff categories, 5 captions per image, 250,000 people with keypoints.
"""

_HOMEPAGE = "https://cocodataset.org/#home"

_LICENSE = "CC BY 4.0"


_IMAGES_URLS = {
    "train": "datasets/coco/train2014.zip",
    "validation": "datasets/coco/val2014.zip",
}

_KARPATHY_FILES_URL = "datasets/coco/caption_datasets.zip"

_SPLIT_MAP = {"train": "train2014", "validation": "val2014"}

_FEATURES = datasets.Features(
    {
        "image": datasets.Image(),
        "filepath": datasets.Value("string"),
        "sentids": [datasets.Value("int32")],
        "filename": datasets.Value("string"),
        "imgid": datasets.Value("int32"),
        "split": datasets.Value("string"),
        "sentences": {
            "tokens": [datasets.Value("string")],
            "raw": datasets.Value("string"),
            "imgid": datasets.Value("int32"),
            "sentid": datasets.Value("int32"),
        },
        "cocoid": datasets.Value("int32"),
    }
)

_FEATURES_CAPTIONS = datasets.Features(
    {
        "image": datasets.Image(),
        "filepath": datasets.Value("string"),
        "sentids": [datasets.Value("int32")],
        "filename": datasets.Value("string"),
        "imgid": datasets.Value("int32"),
        "split": datasets.Value("string"),
        "sentences_tokens": [[datasets.Value("string")]],
        "sentences_raw": [datasets.Value("string")],
        "sentences_sentid": [datasets.Value("int32")],
        "cocoid": datasets.Value("int32"),
    }
)


class COCO(datasets.GeneratorBasedBuilder):
    """COCO"""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="2014", version=VERSION, description="2014 version of COCO with Karpathy annotations and splits"
        ),
        datasets.BuilderConfig(
            name="2014_captions",
            version=VERSION,
            description="Same as 2014 but with all captions of one image gathered in a single example",
        ),
    ]

    DEFAULT_CONFIG_NAME = "2014"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=_FEATURES if self.config.name == "2014" else _FEATURES_CAPTIONS,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        annotation_file = os.path.join(dl_manager.download_and_extract(_KARPATHY_FILES_URL), "dataset_coco.json")
        image_folders = {k: Path(v) for k, v in dl_manager.download_and_extract(_IMAGES_URLS).items()}

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "annotation_file": annotation_file,
                    "image_folders": image_folders,
                    "split_key": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "annotation_file": annotation_file,
                    "image_folders": image_folders,
                    "split_key": "validation",
                },
            ),
            # datasets.SplitGenerator(
            #     name=datasets.Split.TEST,
            #     gen_kwargs={
            #         "annotation_file": annotation_file,
            #         "image_folders": image_folders,
            #         "split_key": "test",
            #     },
            # ),
        ]

    def _generate_examples(self, annotation_file, image_folders, split_key):
        if self.config.name == "2014_captions":
            return self._generate_examples_2014_captions(annotation_file, image_folders, split_key)
        elif self.config.name == "2014":
            return self._generate_examples_2014(annotation_file, image_folders, split_key)

    def _generate_examples_2014_captions(self, annotation_file, image_folders, split_key):
        with open(annotation_file, "r", encoding="utf-8") as fi:
            annotations = json.load(fi)

            for image_metadata in annotations["images"]:
                if split_key == "train":
                    if image_metadata["split"] != "train" and image_metadata["split"] != "restval":
                        continue
                elif split_key == "validation":
                    if image_metadata["split"] != "val":
                        continue
                elif split_key == "test":
                    if image_metadata["split"] != "test":
                        continue

                if "val2014" in image_metadata["filename"]:
                    image_path = image_folders["validation"] / _SPLIT_MAP["validation"]
                else:
                    image_path = image_folders["train"] / _SPLIT_MAP["train"]

                image_path = image_path / image_metadata["filename"]

                record = {
                    "image": str(image_path.absolute()),
                    "filepath": image_metadata["filename"],
                    "sentids": image_metadata["sentids"],
                    "filename": image_metadata["filename"],
                    "imgid": image_metadata["imgid"],
                    "split": image_metadata["split"],
                    "cocoid": image_metadata["cocoid"],
                    "sentences_tokens": [caption["tokens"] for caption in image_metadata["sentences"]],
                    "sentences_raw": [caption["raw"] for caption in image_metadata["sentences"]],
                    "sentences_sentid": [caption["sentid"] for caption in image_metadata["sentences"]],
                }

                yield record["imgid"], record

    def _generate_examples_2014(self, annotation_file, image_folders, split_key):
        counter = 0
        with open(annotation_file, "r", encoding="utf-8") as fi:
            annotations = json.load(fi)

            for image_metadata in annotations["images"]:
                if split_key == "train":
                    if image_metadata["split"] != "train" and image_metadata["split"] != "restval":
                        continue
                elif split_key == "validation":
                    if image_metadata["split"] != "val":
                        continue
                elif split_key == "test":
                    if image_metadata["split"] != "test":
                        continue

                if "val2014" in image_metadata["filename"]:
                    image_path = image_folders["validation"] / _SPLIT_MAP["validation"]
                else:
                    image_path = image_folders["train"] / _SPLIT_MAP["train"]

                image_path = image_path / image_metadata["filename"]

                for caption in image_metadata["sentences"]:
                    yield counter, {
                        "image": str(image_path.absolute()),
                        "filepath": image_metadata["filename"],
                        "sentids": image_metadata["sentids"],
                        "filename": image_metadata["filename"],
                        "imgid": image_metadata["imgid"],
                        "split": image_metadata["split"],
                        "sentences": {
                            "tokens": caption["tokens"],
                            "raw": caption["raw"],
                            "imgid": caption["imgid"],
                            "sentid": caption["sentid"],
                        },
                        "cocoid": image_metadata["cocoid"],
                    }
                    counter += 1
