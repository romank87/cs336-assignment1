.PHONY: download-data test train-tokenizer clean

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

clean:
	rm -rf data/*.txt data/*.gz
