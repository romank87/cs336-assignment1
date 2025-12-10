.PHONY: download-data test train-tokenizer tokenize-tinystories tokenize-owt tokenize clean

download-data:
	mkdir -p data
	cd data && \
	wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt && \
	wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt && \
	wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz && \
	gunzip owt_train.txt.gz && \
	wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz && \
	gunzip owt_valid.txt.gz

test:
	uv run pytest

train-tokenizer:
	uv run cs336_basics/tokenizer_training.py

tokenize-tinystories:
	mkdir -p tokenized
	uv run cs336_basics/tokenizer.py \
		--tokenizer-dir tokenizer/tiny \
		--dataset data/TinyStoriesV2-GPT4-train.txt \
		--output-dir tokenized
	uv run cs336_basics/tokenizer.py \
		--tokenizer-dir tokenizer/tiny \
		--dataset data/TinyStoriesV2-GPT4-valid.txt \
		--output-dir tokenized

tokenize-owt:
	mkdir -p tokenized
	uv run cs336_basics/tokenizer.py \
		--tokenizer-dir tokenizer/owt \
		--dataset data/owt_train.txt \
		--output-dir tokenized
	uv run cs336_basics/tokenizer.py \
		--tokenizer-dir tokenizer/owt \
		--dataset data/owt_valid.txt \
		--output-dir tokenized

tokenize: tokenize-tinystories tokenize-owt

clean:
	rm -rf data/*.txt data/*.gz
