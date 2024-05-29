import os
import torch
from loguru import logger
from transformers import AutoTokenizer
from nltk.tokenize import TreebankWordTokenizer as twt
from datasets import Dataset, DatasetDict, load_dataset, Value, ClassLabel, Features, Sequence
import pandas as pd

model_checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


ANNOTATIONS = [
    "O",
    "B-IndustryWorkplace",
    "I-IndustryWorkplace",
    "S-IndustryWorkplace",
    "E-IndustryWorkplace",
    # "IndustryWorkplace",
    "B-OccupationJobTitle",
    "I-OccupationJobTitle",
    "S-OccupationJobTitle",
    "E-OccupationJobTitle",
    # "OccupationJobTitle",
    "B-JobTaskactivity",
    "I-JobTaskactivity",
    "S-JobTaskactivity",
    "E-JobTaskactivity",
    # "JobTaskactivity",
    "B-SubstanceOrExposureMeasured",
    "I-SubstanceOrExposureMeasured",
    "S-SubstanceOrExposureMeasured",
    "E-SubstanceOrExposureMeasured",
    # "SubstanceMeasured",
    "B-OHMeasurementDevice",
    "I-OHMeasurementDevice",
    "S-OHMeasurementDevice",
    "E-OHMeasurementDevice",
    # "MeasurementDevice",
    "B-SampleTypePersonal",
    "I-SampleTypePersonal",
    "S-SampleTypePersonal",
    "E-SampleTypePersonal",
    # "SampleTypePersonal",
]


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids


def read_annotations(path):
    """Read annotations from the file path

    Args:
        path: (str) the file path
    """
    logger.info(f"Read annotations from {path}")
    annos = []

    def process_span(span, line, index):
        if ";" not in span[-1][-1]:
            span[-1] = [int(val) for val in span[-1]]
            return index
        last_end, new_start = span[-1][-1].split(";")
        span[-1] = [int(span[-1][0]), int(last_end)]
        span.append([new_start, line[index+1]])
        return process_span(span, line, index+1)


    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            line = line.split()
            label = line[1]
            if label == "AnnotatorNotes":
                continue
            span = [line[2:4]]
            index = 3
            index = process_span(span, line, index)
            tokens = line[index+1:]
            annos.append((label, span, tokens))
    return annos


def read_text(path):
    """Read text from the file path

    Args:
        path: (str) the file path
    """
    logger.info(f"Read text from {path}")
    with open(path) as f:
        return f.read()


RAW_FILE_READER = {
    "ann": read_annotations,
    "txt": read_text,
}


def read_raw_data(dir_name):
    contents = {}
    for path, dir_list, file_list in os.walk(dir_name):
        for filename in file_list:
            logger.info(f"Path: {path}")
            logger.info(f"File list: {file_list}")
            logger.info(f"File name: {filename}")
            suffix = filename.split(".")[-1]
            name = filename.split(".")[0]
            if name not in contents:
                contents[name] = {}
            contents[name][suffix] = RAW_FILE_READER[suffix](f"{path}/{filename}")

    return contents


def build_dataset(raw_contents):
    train_samples = []
    valid_samples = []
    test_samples = []
    label_map = {label: i for i, label in enumerate(ANNOTATIONS)}
    for key, val in raw_contents.items():
        logger.info(f"Start processing {key}")
        split_token = [
            index
            for index, token in enumerate(val["txt"]) if token == '\n'
#            or
#            (token == '.' and val["txt"][index+1] == ' ')
#            or
#            (token == '.' and val["txt"][index+1] == 'T')
#            or 
#            (token == '.' and val["txt"][index-1] == '4' and val["txt"][index+1] == '(')
#            or
#            (token == ';' and val["txt"][index+1] == ' ')
#            or
#            (token == ',' and val["txt"][index-3:index] == 'rs)')
        ]
        sentence_spans = [-1] + split_token + [len(val["txt"])]
        sentence_spans = [sentence_spans[i:i + 2] for i in range(0, len(sentence_spans)-1)]
        ann_index = 0
        span_index = 0
        for sentence_span in sentence_spans:
            start, end = sentence_span
            text = val["txt"][start+1:end]
            print(f"Text: {text}")
            #assert end - start <= 512
            if not text.strip():
                continue
            word_lengths = list(twt().span_tokenize(text))
            words = [text[st:ed] for st, ed in word_lengths]
            word_spans = [(st+start+1, ed+start+1) for st, ed in word_lengths]
            current_index = start + 1
            labels = ["O" for _ in range(len(words))]

            while "ann" in val and ann_index < len(val["ann"]):
            #and val["ann"][ann_index][1][0][0] >= start and val["ann"][ann_index][1][0][0] <= end:
                for span in val["ann"][ann_index][1]:
                    print(f"Span: {span}")
                    for word_index, word_span in enumerate(word_spans):
                        word_start, word_end = word_span
                        if word_start < span[0]:
                            continue
                        if word_end > span[1]:
                            break
                        if span[1] - span[0] == 1:
                            labels[word_index] = "S-"+val["ann"][ann_index][0]
                            break
                        if word_start == span[0]:
                            labels[word_index] = "B-"+val["ann"][ann_index][0]
                        elif word_end == span[1]:
                            labels[word_index] = "E-"+val["ann"][ann_index][0]
                        else:
                            labels[word_index] = "I-"+val["ann"][ann_index][0]
                        # labels[word_index] = val["ann"][ann_index][0]
                ann_index += 1
            labels = [label_map[v] for v in labels]
            if key == 'Train':
                train_samples.append((words, labels))
            elif key == 'Valid':
                valid_samples.append((words, labels))
            elif key == 'Test':
                test_samples.append((words, labels))
    logger.info(f"Train samples: {len(train_samples)}")
    logger.info(f"Valid samples: {len(valid_samples)}")
    logger.info(f"Test samples: {len(test_samples)}")
    from_samples_to_dataset(train_samples, valid_samples, test_samples, "ner")
    train_samples = []
    valid_samples = []
    test_samples = []

def from_samples_to_dataset(train_samples, valid_samples, test_samples, dataset_name):
    logger.info(f"Start split:")
    train_features = train_samples
    valid_features = valid_samples
    test_features = test_samples
    # train_features = samples
    # valid_features = samples
    # test_features = samples

    train_features = pd.DataFrame(train_features, columns=["tokens", "ner_tags"])
    valid_features = pd.DataFrame(valid_features, columns=["tokens", "ner_tags"])
    test_features = pd.DataFrame(test_features, columns=["tokens", "ner_tags"])

    train_dataset = Dataset.from_pandas(train_features, Features({"tokens": Sequence(Value("string")), "ner_tags": Sequence(ClassLabel(num_classes=len(ANNOTATIONS), names=ANNOTATIONS))}))
    valid_dataset = Dataset.from_pandas(valid_features, Features({"tokens": Sequence(Value("string")), "ner_tags": Sequence(ClassLabel(num_classes=len(ANNOTATIONS), names=ANNOTATIONS))}))
    test_dataset = Dataset.from_pandas(test_features, Features({"tokens": Sequence(Value("string")), "ner_tags": Sequence(ClassLabel(num_classes=len(ANNOTATIONS), names=ANNOTATIONS))}))

    dataset = DatasetDict({"train": train_dataset, "valid": valid_dataset, "test": test_dataset})
    dataset.save_to_disk("/Users/paulthompson/ephor-final-corpus-joined-files/splits-brat-ner-datasets/7/dataset")

# torch.save(train_features, "train.feature")
    # torch.save(valid_features, "valid.feature")
    #  torch.save(test_features, "test.feature")


if __name__ == "__main__":
    raw_contents = read_raw_data("/Users/paulthompson/ephor-final-corpus-joined-files/splits-brat-sep-folders/7")
    build_dataset(raw_contents)
