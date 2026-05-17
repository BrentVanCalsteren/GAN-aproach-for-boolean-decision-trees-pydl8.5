Short summenary of what this projects want to achieve.

This project will try to generate good synthetic samples with the help of pydl8.5. 
Pydl8.5 is a extended implementation (https://github.com/aia-uclouvain/pydl8.5) of the DL8.5 algorithm 
with a python interface. Dl8.5 will learn optimal decision trees very efficiently.
This project will try with the help of these trees to generate good new samples.
Before diving into using pydl8.5 for generating new data I will briefly go over how decision trees work.
After this i will shortly explain how dl8.5 will learn trees and then I will go 
over how we can use pydl8.5 the generate new data in python.

General background

Decision trees in general

First of all, decision trees are classifiers. 
Meaning they will try to group a group of samples together based on a
criteria which minimizes the global error for a given goal.
Like any machine learning model, decision trees models are 
learned on a given training set. when the model is done learning, each leaf of a decision tree will contain a subgroup of the samples which share a common trait/feature. (this is used for splitting the data).
A good model will try to fit a tree over the data in such a way that it has the lowest possible error it can find.
Now there exist infinite trees for each possible training set.
Further there can be multiple trees for each possible error.
Now given 2 trees with the same error, we say tree 1 is more optimal than tree 2 if tree 1 has less amount of splits than tree 2 aka more compact.
The problem of finding the most optimal tree for a random dataset is np-hard.
For this reason most dc models will go for a greedy approach (exemple CART and C4.5) for finding a good fitting decision tree. 
The focus is more on finding low error trees and the structure of the tree. 
Now if we clamp the depth a tree can go to, meaning we set a maximum number of splits from the root. 
The problem will become solvable since we can always calc a finite upperbound of possible trees. 
With depth limit in place we could use a algoritm that is not greedy in nature and find the optimal tree.
This is where dl8.5 comes into the picture. (there are other algos as well like a MIP or SAT based approaches)

DL8.5

Dl8.5 is a efficienter version of Dl8.
Will first go over the core workings of Dl8.
DL8 will look a decision trees in a unique way and combine it with itemset mining.
Each path in a given tree will be seen as an itemset by DL8.
For this to work can a split in a tree only mean 1 thing: item aka feature 'a' is part of the sample or not.
Meaning that it can only create trees based on some sort of boolean array abstraction of each sample.
So each sample will be abstracted as a boolean array of the same lenght, where each
value in that array tell's us that the sample has item 'x' or not.
So in order to work with dl8 or 8.5 you will need to convert all your training samples to a boolean equivalent.
Further does dl8 allow you to define costum error functions for the leafs.
The functions don't have to be based on the binary data, you can use the original values here.
So where are the boolean arrays used for? each bool value is a possible split, where the error function will 
determine which split will be used. (the gains made in finding a tree by 8.5 will for now not be discused)

Datasets used

Most datasets that will be used are simple tabular datasets (each sample is a 1d array of features) from uci machine learning repository (https://archive.ics.uci.edu/datasets)
But also more complex datasets can be used if it can be converted to a 2d matrix equivalent of some sort!
(may lose structural information in the proces though, exemple: mnist dataset) working with real rgb image has not been explored.

Boolean convertion

The convertion ...

Data generation

...

Testing the gen data

...


