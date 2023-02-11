import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from enum import Enum
from ann import MLP, CostFunction,GradientDescent,Activation,LearningRate, plot_cost_vs_epochs,plot_training_vs_validation_acc


# Initialize parser
parser = argparse.ArgumentParser(description = 'This script runs an ANN network given a set of hyperparameters on the breast-cancer-wisconsin Dataset\nhttps://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data')


## Frinedly helper function pickers
friendlyActivationFunctionHelper = ''
for func in Activation.functions:
    friendlyActivationFunctionHelper+=str(func.id) + ":" +func.name +"\n"

friendlyCostFunctionHelper = ''
for func in CostFunction:
    friendlyCostFunctionHelper+=str(func.value) + ":" +func.name +"\n"

friendlyGdHelper = ''
for func in GradientDescent:
    friendlyGdHelper+=str(func.value) + ":" +func.name +"\n"

friendlyLearningRateHelper = ''
for func in LearningRate:
    friendlyLearningRateHelper+=str(func.value) + ":" +func.name +"\n"


# Adding  argument
parser.add_argument("-iNodes", "--input-nodes", type=int,default=0,help='Number of input nodes')
parser.add_argument("-hNodes", "--hidden-nodes", type=int,default=10,help='Number of hidden nodes/layer')
parser.add_argument("-hLayers", "--hidden-layers", type=int,default=1,help='Number of hidden layers')
parser.add_argument("-epochs", "--epochs", type=int,required=True,help='Number of epochs')
parser.add_argument("-lr","--lr",type=float,default=0.01,help='Learning rate when using a constant learning rate mode')
parser.add_argument("-seed","--seed",type=int,default=1,help='Seed for randomness')
parser.add_argument("-gd","--gradient-descent",type=int,default=GradientDescent.BATCH,help=friendlyGdHelper )
parser.add_argument("-a","--activation-function",type=int,default=1,help=friendlyActivationFunctionHelper )
parser.add_argument("-c","--cost-function",type=int,default=CostFunction.CROSS_ENTROPY,help=friendlyCostFunctionHelper )
parser.add_argument("-lrm","--learning_rate_mode",type=int,default=LearningRate.CONSTANT,help=friendlyLearningRateHelper)
parser.add_argument("-min-lr","--min-lr",type=float,default=0.001,help='Minimum learning rate when using a decay function')
parser.add_argument("-max-lr","--max-lr",type=float,default=0.01,help='Maximum learning rate when using a decay function')
parser.add_argument("-shuffle","--shuffle",action='store_true',help='Shuffle dataset')
parser.add_argument("-mini-batch-size","--mini-batch-size",type=int,default=100)



# Read arguments from command line
args = parser.parse_args()



"""# Data Preparation"""
print('Preparing data...')
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data',sep=',')
## Discretization
data=data.replace(to_replace="M",value=1)
data=data.replace(to_replace="B",value=0)
## Normalization (scaling) 
data[:] = minmax_scale(data)

# data

"""# Training """
print('Preparing data... Splitting')
target_data = data.iloc[:,1].to_numpy() ## Target class (Select column 2)
features_data = data.iloc[:,[i for  i in range(2,data.shape[1])]].to_numpy() ## Features (Select column 3 - 32)
X_train, X_test, y_train, y_test = train_test_split(features_data,target_data, test_size=0.2,random_state=40) ## Split

print('Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))

NoInputNodes =args.input_nodes
NhiddenLayers = args.hidden_layers
NhiddenNeurons= args.hidden_nodes
epochs= args.epochs
lr=args.lr
minibatchSize=args.mini_batch_size
shuffle=args.shuffle
seed=args.seed
activation=Activation.functions[args.activation_function]
costFunction=CostFunction(args.cost_function)
GD=GradientDescent(args.gradient_descent)
learningRateMode=args.learning_rate_mode
MIN_LR=args.min_lr
MAX_LR=args.max_lr



nn = MLP(
        NhiddenNeurons=NhiddenNeurons,
        NoInputNodes=NoInputNodes,
        NhiddenLayers=NhiddenLayers, 
        epochs=epochs, 
        lr=lr,
        minibatchSize=minibatchSize, 
        shuffle=shuffle,
        seed=seed,
        activation=activation,
        costFunction=costFunction,
        GD=GD,
        learningRateMode=learningRateMode,
        min_lr=MIN_LR,
        max_lr=MAX_LR
    )

nn.fit(X_train=X_train, 
       y_train=y_train,
       X_actual=X_test,
       y_actual=y_test)
y_test_pred = nn.predict(X_test[:,0:NoInputNodes or X_train.shape[1]])
acc = (np.sum(y_test == y_test_pred)
       .astype(np.float) / X_test.shape[0])

print('Test accuracy: %.2f%%' % (acc * 100))
print("Saving plots...")
## Saving plots 
plot_cost_vs_epochs(nn)
plot_training_vs_validation_acc(nn)
