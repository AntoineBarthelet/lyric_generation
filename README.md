# lyric_generation
Using and Fine Tuning Google's BERT language model to produce lyrics. 


# Fine Tuning Bert

From huggingface's https://github.com/huggingface/transformers, which allows you to tune a variety of models for many different outcomes.

In this case we are trying to orient BERT's language model to resemble that of a particular (or mixture of) songwriter's style.

Variables:
train_data_file: text file with lyrics, each new line is considered a sample
output_dir: the path that will become the model path in the bert_babble.ipynb
model_type: which class of model is being finetuned
line_by_line: how you want to organze your samples (look into this for future use in whole song building)
model_name_or_path: base model to be finetuned
do_train: just fintetune the model, other options allow for evaluation/test
per_gpu_train_batch_size: number of batches to use, keep low if training on a single GPU. My laptop has a nvidia GTX 2070, and 6 was as high as I could go
mlm: #TBD
num_train_epochs: number of times to go through the whole training set. It is slow, and reading up says it tends to overfit quickly, between 1 and 5 should be adequate
overwrite_output_dir: overwrite if the output dir already exists

code to to fine tune:
python ./examples/run_language_modeling.py --train_data_file mflannery --output_dir tmp\mflannery --model_type bert --line_by_line --model_name_or_path bert-base-uncased --do_train --per_gpu_train_batch_size=6 --mlm --num_train_epochs=3 --overwrite_output_dir

# TO DO:
- connect paths between where transformer places fine tuned model and where bert reads it
- add comments for each bert_babble function explaining input, work, and output
- add library versions 
- add links to download data for uncased model
- add data used in lyric fine tuning, make sure its not copywrited
- more to dos