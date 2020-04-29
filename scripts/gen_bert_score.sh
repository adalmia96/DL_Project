export BERT_BASE_DIR=../models/wwm_uncased_L-24_H-1024_A-16
export INPUT_FILE=../output/50.txt
python3 ./bert-as-language-model/run_lm_predict.py \
  --input_file=$INPUT_FILE \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=50 \
  --output_dir=/tmp/lm_output/
