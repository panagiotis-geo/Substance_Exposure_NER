Datasets=("Foreland-et-al-2013-1" "Foreland-et-al-2013-2" "Linch-2002-1" "Linch-2002-2" "Nieuwenhuijsen-et-al-1999-1")
for dataset in ${Datasets[*]}; do
  echo "Predicting $dataset"
  poetry run python ner/train_huggingface.py\
    --model_name_or_path /scratch/ace14856qn/test-ner \
    --dataset_name /scratch/ace14856qn/${dataset}_dataset \
    --output_dir /scratch/ace14856qn/test-$dataset \
    --do_predict
done
  #--overwrite_output_dir
