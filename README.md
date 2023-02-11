## Environment
- Python 3 (used on 3.6.9)
- Pip package manager (Used on pip 21.3.1 )
- OS (used Linux - Ubuntu)

## Get Started 

```pip install -r requirements.txt```

## Usage 
```
usage: cli.py [-h] [-iNodes INPUT_NODES] [-hNodes HIDDEN_NODES]
              [-hLayers HIDDEN_LAYERS] -epochs EPOCHS [-lr LR] [-seed SEED]
              [-gd GRADIENT_DESCENT] [-a ACTIVATION_FUNCTION]
              [-c COST_FUNCTION] [-lrm LEARNING_RATE_MODE] [-min-lr MIN_LR]
              [-max-lr MAX_LR] [-shuffle] [-mini-batch-size MINI_BATCH_SIZE]

This script runs an ANN network given a set of hyperparameters on the breast-
cancer-wisconsin Dataset https://archive.ics.uci.edu/ml/machine-learning-
                        Number of epochs
  -lr LR, --lr LR       Learning rate when using a constant learning rate mode
  -seed SEED, --seed SEED
                        Seed for randomness
  -gd GRADIENT_DESCENT, --gradient-descent GRADIENT_DESCENT
                        0:BATCH 1:MINIBATCH 2:STOCHASTIC
  -a ACTIVATION_FUNCTION, --activation-function ACTIVATION_FUNCTION
                        0:Sigmoid 1:RELU 2:Tanh 3:Linear 4:One
  -c COST_FUNCTION, --cost-function COST_FUNCTION
                        0:LOGISTIC_LOSS 1:CROSS_ENTROPY 2:MEAN_SQUARED_ROOT
  -lrm LEARNING_RATE_MODE, --learning_rate_mode LEARNING_RATE_MODE
                        0:CONSTANT 1:SCHEDULE
  -min-lr MIN_LR, --min-lr MIN_LR
                        Minimum learning rate when using a decay function
  -max-lr MAX_LR, --max-lr MAX_LR
                        Maximum learning rate when using a decay function
  -shuffle, --shuffle   Shuffle dataset
  -mini-batch-size MINI_BATCH_SIZE, --mini-batch-size MINI_BATCH_SIZE

```


### Examples

```python cli.py -hLayers 1 -hNodes 10 -seed 2 -gd 1 -epochs 100 -lr 0.001 -c 2 -lrm 1 ```

```python cli.py -hLayers 1 -hNodes 10 -seed 2 -gd 1 -epochs 100 -lr 0.001 -c 2 -lrm 1 ```

```python cli.py -hLayers 1 -hNodes 10 -seed 2 -gd 1 -epochs 100 -lr 0.001 -c 2 -lrm 1 ```