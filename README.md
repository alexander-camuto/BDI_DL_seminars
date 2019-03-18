## Welcome

Welcome to the git repository for the Oxford Big Data Institute's seminar series on Deep Learning !

We'll be posting links to required readings for each week and uploading course slides, notes, and supporting code.

## Readings

### Week 1

Three papers, first two quite technical that can be skimmed just to get the headline ideas, and then one more approachable paper that is the foundation for deep neural networks. We really just want you to get the take-aways for the first two, and if the last doesn't make much sense, hopefully it will after the first session.

1.  **Approximation by Superposition of a Sigmoid Function, Cybenko (1989)**
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.441.7873&rep=rep1&type=pdf

    > This shows the remarkable result that a superposition of sigmoidal activation functions can arbitrarily closely match any continuous function, as long as you have infinite units and unbounded weights. This is the result behind people talking about a neural net with one infinitely wide hidden layer being a universal function approximator.

2.  **The Lack of A Priori Distinctions Between Learning Algorithms, Wolpert (1996)**
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.390.9412&rep=rep1&type=pdf

    > Again please only skim this for the headline result, which is probably familiar to those of you with a background in statistics. The result is that, formally speaking, no supervised learning algorithm can be better than any other in the totally general case of all possible supervised learning problems. Of course the world has regularities that we exploit, but nevertheless in general we cannot expect to find one particular machine learning algorithms that will dominate in all tasks.

3.  **Deep Sparse Rectifier Neural Networks, Glorot, Bordes and Bengio (2011)**
    http://proceedings.mlr.press/v15/glorot11a/glorot11a.pdf

    > Much more recent that the other two. This is the paper that gives us the tool to build very large networks, the RELU activation function. Most of the methods for training models referred to in this paper are now hideously out of date - the field has moved on a lot since 2011. This activation function means that gradient signal can go further through the model.
