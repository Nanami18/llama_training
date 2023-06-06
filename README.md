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

5. Evaluate the model on the validation set
```python3
python evaluate.py --model_dir checkpoints/pile_llama_gpu --tokenizer_path checkpoints/pile_llama/tokenizer.model --valset_path data/val.jsonl --valset_size 10000
```
If you wish, download the checkpoint we trained on first million lines of the Pile training set from [here](https://drive.google.com/drive/folders/1LaFJ9a4i14VZ0unNlI5jKFSUoT7CsGrd?usp=sharing). We use the tokenizer provided by the llama team. To get the access, fill this [form](https://docs.google.com/forms/d/e/1FAIpQLSfqNECQnMkycAp2jP4Z9TFX0cGR4uf7b_fBxjY_OjhJILlKGA/viewform).

Disclaimer: our model is significantly undertrained due to the computing resources we have.
