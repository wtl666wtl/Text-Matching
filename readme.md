# Text Matching Project
CS229 homework in ACM class.

## Environment
I used python=3.7, torch=1.10.0, cudatoolkit=11.1.

## Hyperparameters
Overall, batch_size=32, initial learening_rate=2e-5.

For CoSENT, I set $\lambda$ = 20.

## Run
You can modify the train/test data in "train.tsv"/"test.tsv".
Results will be saved under "submission.csv"/"CoSent_submission.csv"/"xlnet_submission.csv".

#### Interaction-based
Run `python main.py` (default model is RoBERTa).

Run `python xlnet_main.py` to use XLNet.

#### Representation-based
Run `python CoSent_main.py` (default model is RoBERTa + CoSENT).

## Performance



|   Method    | BERT (w/o data augmentation) |  BERT   | RoBERTa | CoSENT  |
| :---------: | :--------------------------: | :-----: | :-----: | :-----: |
| Performance |           73.451%            | 80.088% | 88.053% | 76.991% |