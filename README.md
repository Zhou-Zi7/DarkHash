# DarkHash

## Setup Environment

Before you begin, make sure you have installed the following dependencies:

- Python 3.8
- Torch 1.12.1

You can install and set up the project by following these steps:

# Download the ImageNet dataset:

We give the list of training, database and query images in data/imagenet/test.txt, data/imagenet/train.txt and data/imagenet/database.txt

# Enter the project directory:

cd DarkHash

# Run distill_dataset.py to generate distill dataset:

python distill_dataset.py

# Run attack.py to train the backdoored model:

python attack.py 