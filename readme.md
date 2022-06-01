# Text Matching Project
CS229 homework in ACM class.

## Environment
I used python=3.7, torch=1.10.0, cudatoolkit=11.1.

## Run
You can modify the train/test data in "train.tsv"/"test.tsv".
Results will be saved under "submission.csv"/"CoSent_submission.csv"

#### BERT/ RoBERTa (Interaction-based)
Run `python main.py` (default model is RoBERTa).

#### CoSENT (Representation-based)
Run `python CoSent_main.py` (default model is RoBERTa + CoSENT).

## Performance

|   Method    |  BERT   | BERT (with data augmentation) | RoBERTa (with more tricks) | RoBERTa + CoSENT |
| :---------: | :-----: | :---------------------------: | :------------------------: | :--------------: |
| Performance | 73.451% |            80.088%            |          86.725%           |     76.991%      |
