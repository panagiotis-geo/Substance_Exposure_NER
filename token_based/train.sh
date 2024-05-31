poetry run python ner/train_huggingface.py\
  --model_name_or_path bert-base-uncased \
  --dataset_name ../datasets/token-based/train-valid-test \
  --output_dir ./test-ner \
  --do_train \
  --do_eval \
  --do_predict
  #--overwrite_output_dir
