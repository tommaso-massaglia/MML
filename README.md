### Abstract 

The following work will be a comprehensive analysis of the Wine Quality Data-set found in the UCI machine learning repository. The final goal of this experience was to train and compare different classification/regression methods on the data-set following a comprehensive analysis of the latter carried through statistical and mathematical methods with the goal of cutting out the less important attributes, augmenting or balancing the data, and overall making the subsequent prediction tasks more precise. Regarding the prediction, the target attributes chosen were 'type', which consisted in a binary classification between red/white wine, and 'quality', a measure from 1 to 10 which was carried on through regression and support vector machines. Overall the achieved precision on the task was satisfying, and some prepossessing pipelines were shown to perform admirably. 


### Contents

- modules: contains the modules written for the database importing (WineDb.py) and the preprocessing steops (PreprocessingFunctions.py) as well as the dataset itself, as it is accessed by one of the two modules.
- plot: contains the output plots used in the tesina
- data_exploration.ipynb: contains the steps performed when exploring the data
- data_preparation.ipynb: not very meaningful, was used to test the modules written in PreprocessingFunctions.py
- model_training.ipynb: contains all the needed functions and algorithms that were used to train, test and compare the produced models