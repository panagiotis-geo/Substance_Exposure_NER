import os
import torch
from loguru import logger
from transformers import AutoTokenizer
from nltk.tokenize import TreebankWordTokenizer as twt

model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


ANNOTATIONS = [
    "O",
    "I-IndustryWorkplace",
    "B-IndustryWorkplace",
    "I-OccupationJobTitle",
    "B-OccupationJobTitle",
    "I-JobTaskactivity",
    "B-JobTaskactivity",
    "I-SubstanceOrExposureMeasured",
    "B-SubstanceOrExposureMeasured",
    "I-OHMeasurementDevice",
    "B-OHMeasurementDevice",
    "I-SampleTypePersonal",
    "B-SampleTypePersonal",
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
            suffix = filename.split(".")[-1]
            name = filename.split(".")[0]
            if name not in contents:
                contents[name] = {}
            contents[name][suffix] = RAW_FILE_READER[suffix](f"{path}/{filename}")

    return contents


def build_dataset(raw_contents):
    samples = []
    file1 = open("sentence_info.txt", "w")
    for key, val in raw_contents.items():
        logger.info(f"Start processing {key}")
        split_token = [
            index
            for index, token in enumerate(val["txt"]) if token == '\n' or
            (token == '.' and val["txt"][index+1] == ' ')
            or
            (token == '.' and val["txt"][index+1] == 'T')
            or
            (token == ')' and val["txt"][index-3:index] == '(19')
            #or
            #(token == '.' and val["txt"][index-1] == '4' and val["txt"][index+1] == '(')
            or
            (token == ';' and val["txt"][index+1] == ' ')
            or
            (token == ',' and val["txt"][index-3:index] == 'rs)')
        ]
        sentence_spans = [-1] + split_token + [len(val["txt"])]
        sentence_spans = [sentence_spans[i:i + 2] for i in range(0, len(sentence_spans)-1)]
        ann_index = 0
        span_index = 0
        for sentence_span in sentence_spans:
            start, end = sentence_span
            text = val["txt"][start+1:end]
            logger.info(f"File: {key}")
            logger.info(f"Text: {text}")
            logger.info(f"Start: {start}")
            logger.info(f"End: {end}")
            assert end - start <= 512
            if not text.strip():
                continue
            word_lengths = list(twt().span_tokenize(text))
            words = [text[st:ed] for st, ed in word_lengths]
            word_spans = [(st+start+1, ed+start+1) for st, ed in word_lengths]
            current_index = start + 1
            labels = ["O" for _ in range(len(words))]

            while ann_index < len(val["ann"]) and val["ann"][ann_index][1][0][0] >= start and val["ann"][ann_index][1][0][0] <= end:
                for span in val["ann"][ann_index][1]:
                    for word_index, word_span in enumerate(word_spans):
                        word_start, word_end = word_span
                        if word_start < span[0]:
                            continue
                        if word_end > span[1]:
                            break
                        if word_start == span[0]:
                            labels[word_index] = "B-"+val["ann"][ann_index][0]
                        else:
                            labels[word_index] = "I-"+val["ann"][ann_index][0]
                ann_index += 1
            logger.info(f"Words: {words}")
            samples.append((words, labels))
            file1.write(key + "\t");
            file1.write(str(start) + "\t");
            file1.write(str(end) + "\t");
            file1.write(text + "\t[\"" + words[0].replace('"', '\\"') + "\"");
            for word in words[1:]:
                file1.write(",\"" + word.replace('"', '\\"') + "\"")
            file1.write("]\n");
    file1.close()
    logger.info(f"Overall samples: {len(samples)}")

    label_map = {label: i for i, label in enumerate(ANNOTATIONS)}
    pad_token_label_id = torch.nn.CrossEntropyLoss().ignore_index
    sep_token = tokenizer.sep_token
    cls_token = tokenizer.cls_token
    pad_token = tokenizer.pad_token_id
    pad_token_segment_id=tokenizer.pad_token_type_id
    cls_token_segment_id = 0
    sequence_a_segment_id = 0
    max_seq_length = 512
    features = []

    for sample_index, sample in enumerate(samples):
        tokens = []
        label_ids = []
        for word, label in zip(*sample):
            word_tokens = tokenizer.tokenize(word)
            if len(word_tokens) > 0:
                tokens.extend(word_tokens)
                label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))
        special_tokens_count = tokenizer.num_special_tokens_to_add()
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]

        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        segment_ids = [sequence_a_segment_id] * len(tokens)
        tokens = [cls_token] + tokens
        label_ids = [pad_token_label_id] + label_ids
        segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding_length = max_seq_length - len(input_ids)
        input_ids += [pad_token] * padding_length
        input_mask += [0] * padding_length
        segment_ids += [pad_token_segment_id] * padding_length
        label_ids += [pad_token_label_id] * padding_length

        features.append(
            InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, label_ids=label_ids)
        )

    print (len(features))

    logger.info(f"Start split: train 0-5940, validation 5940-6587, test 6587-")
    train_features = features[:5940]
    valid_features = features[5940:6987]
    test_features = features[6987:]

    torch.save(train_features, "train.feature")
    torch.save(valid_features, "valid.feature")
    torch.save(test_features, "test.feature")


if __name__ == "__main__":
    #raw_contents = read_raw_data("/scratch/ace14856qn/ephor-ner")
    raw_contents = read_raw_data("/Users/paulthompson/ephor-final-corpus-joined-files/all")
    build_dataset(raw_contents)
