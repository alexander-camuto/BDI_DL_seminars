## Welcome

Welcome to the git repository for the Oxford Big Data Institute's seminar series on Deep Learning !

We'll be posting links to required readings for each week and uploading course slides, notes, and supporting code.

## Getting Started with Code

- Clone this repo to your machine

- Install conda (https://conda.io/en/latest/), then run the following command from the root directory of the repository: 

`conda create -n bdi_dl python=3.6`

- Now activate your environment:

`conda activate bdi_dl`

- Install the necessary packages: 

`pip install -r requirements.txt`

- Note that the requirements file may expand from week to week so if you find yourself lacking packages to run a particular script, just re-run the prior command. 

For an introduction to Keras, please refer to <https://keras.io/getting-started/sequential-model-guide/>. 

## Readings

### Week 1 - Introduction to Machine Learning

Three papers, first two quite technical that can be skimmed just to get the headline ideas, and then one more approachable paper that is the foundation for deep neural networks. We really just want you to get the take-aways for the first two, and if the last doesn't make much sense, hopefully it will after the first session.

1. **Approximation by Superposition of a Sigmoid Function, Cybenko (1989)**
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.441.7873&rep=rep1&type=pdf

    > This shows the remarkable result that a superposition of sigmoidal activation functions can arbitrarily closely match any continuous function, as long as you have infinite units and unbounded weights. This is the result behind people talking about a neural net with one infinitely wide hidden layer being a universal function approximator.

2. **The Lack of A Priori Distinctions Between Learning Algorithms, Wolpert (1996)**
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.390.9412&rep=rep1&type=pdf

    > Again please only skim this for the headline result, which is probably familiar to those of you with a background in statistics. The result is that, formally speaking, no supervised learning algorithm can be better than any other in the totally general case of all possible supervised learning problems. Of course the world has regularities that we exploit, but nevertheless in general we cannot expect to find one particular machine learning algorithms that will dominate in all tasks.

3. **Deep Sparse Rectifier Neural Networks, Glorot, Bordes and Bengio (2011)**
    http://proceedings.mlr.press/v15/glorot11a/glorot11a.pdf

    > Much more recent that the other two. This is the paper that gives us the tool to build very large networks, the RELU activation function. Most of the methods for training models referred to in this paper are now hideously out of date - the field has moved on a lot since 2011. This activation function means that gradient signal can go further through the model.

    

### Week 2 - Multilayer Perceptrons, Backpropagation and Optimisation

Three papers, more approachable and relevant to current research than those presented last week. We recommend you skim through the entirety of each paper. 

1. **Understanding Deep Learning Requires Re-Thinking Generalization, Zhang et al. (2017)**

   <https://arxiv.org/pdf/1611.03530.pdf>

   > Large enough networks, where the number of parameters exceeds the number of training data-points, can effectively memorize entire "random" datasets. The core concept being described is that of **overfitting**, where a network fits to random noise present in the training data increasing it's "generalization" error when evaluated on a test dataset. 

2. **Visualizing the Loss Landscape of Neural Nets, Li et al. (2017)**

   <http://papers.nips.cc/paper/7875-visualizing-the-loss-landscape-of-neural-nets.pdf>

   > This paper clearly illustrates how different neural network architectures can generate smoother loss 'landscapes' to produce models that generalize better and are less likely to be stuck in local minima. 

3. **Adam: A Method for Stochastic Optimization, Kingma, Lei Ba (2015)**

   https://arxiv.org/pdf/1412.6980.pdf

   > Introduces the Adam Optimizer a good out-of-the-box optimizer which requires very little tuning of  hyper-parameters. It is also computationally efficient, requiring only first-order gradients and little memory (RAM) requirements. Even though the paper was published 4 years ago, it remains the most popular optimizer.

### Week 3 - Convolutional Neural Networks

1. **ImageNet Classification with Deep Convolutional Neural Networks; Krizhevsk, Sutskever and Hinton; 2012**

   https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

   > This is another foundational paper in the recent renaissance of neural networks. The convolutional neural network proposed here, AlexNet, demonstrated a step-change in image classification performance compared to previous approaches.

2. **Dermatologist-level classification of skin cancer with deep neural networks; Esteva et al; 2017**

   https://www.nature.com/articles/nature21056

   > A recent application paper, showing that convolutional neural networks can reach human-level performance at classifying skin cancers

3. **Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps; Simonyan, Vedaldi and Zisserman; 2014**

   https://arxiv.org/abs/1312.6034

   > How can we understand how neural networks come to their decision? This paper propagates gradients into the input in various ways to try to get a handle on what parts of an image are most important to the prediction.

### Week 3 - Recurrent Neural Networks

1. **A Critical Review of Recurrent Neural Networks for Sequence Learning; Lipton et al.; 2015**
    A brief and approachable review of recurrent neural networks. Sections 3 and 4 are probably the most important
    https://arxiv.org/abs/1506.00019

