# llama_training
1. Download Pile dataset [here](https://the-eye.eu/public/AI/pile/train/)
2. Decompress the zstandard file
```python3
python3 data_utils/zst_decompression.py --input_file path_to_input
```
 Can specify the output path by passing --output_file command, otherwise will save the decompressed file in the same directory

3. Download sentenpiece tokenizer weight
4. Train the model
 ```python3
python3 train_llama.py --model_dir checkpoints/pile_llama --load_epoch -1 --tokenizer_path checkpoints/pile_llama/tokenizer.model --epochs 1 --dataset_path data/val.jsonl --batch_size 256 --n_layers 12 --hidden_dim 768 --dataset_size 100000
```
Can replace *model_dir*, *tokenizer_path*, and *dataset_path* based on your directory structure. The model will be loaded and stored under *model_dir*. If train from scratch, specify *load_epoch=1*. 

Current only support taking one jsonl file as the data, and has no streaming functionality at the time, therefore the initial loading of the data can take very long. *data_size* specifies how many lines to use from the training file.
