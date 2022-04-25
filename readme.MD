### Assignment 5: Reinforcement Learning and Deep Learning

#### Due May 19, 11:59pm. (note: I will not be able to give any extensions to this, as I need time to get final grades submitted.)

Please provide a submission.py that demonstrates your code!

<b>(25 points) Task 1: Q-learning for Approach.</b>

In this task, we'll return to Approach and use Q-learning to learn the optimal policy through repeated play.

Recall the dice game "Approach". It works like this: There are two players, and they choose a target number (n).
Player 1 repeatedly rolls a six-sided die and adds up their total. Whenever they want, they can "hold." Then player 2 must roll;' their goal is 
to beat player 1's score without going past n.

The problem we want to solve is this: 

For player 1, for a given n and a particular total so far (called s) should they
'hold' (action 1) or 'roll' (action 2).

(note that player 2's strategy in this game is pretty boring - they just roll until they either beat player 1's score or exceed n.)

Earlier, we used Monte Carlo simulation to determine the best value to hold. This is fine when there's a small number of potential policies to consider, but it's not very scalable.
Let's now add in Q-learning. 

Q-learning generates a table that stores Q(s,a). Write a function that takes in a limit n
and computes the Q-table for [0,n] with the actions (hold, roll). Include it in a file called approach.py.

For example, for n=10, you might get:

    sum:   hold     roll   [action]
    0: 0.000000 0.493921 [roll]
    1: 0.000000 0.477271 [roll]
    2: 0.000000 0.479286 [roll]
    3: 0.000000 0.505797 [roll]
    4: 0.000000 0.580803 [roll]
    5: 0.051270 0.498027 [roll]
    6: 0.170403 0.427388 [roll]
    7: 0.297536 0.365115 [roll]
    8: 0.478196 0.285432 [hold]
    9: 0.711051 0.164235 [hold]
    10: 1.000000 0.000000 [hold]

We will do this using *epsilon-soft on-policy control*, a form of reinforcement learning. We will select the optimal action with probability (1-epsilon) and explore with probability epsilon.
(use epsilon=0.1 for this.)

Start  by initializing the Q-table to small random values. Our agent should start with a random s and a and then either take the best action (prob. 1-epsilon), or the other action (prob. epsilon). 
At the end of the game, it receives reward 1 (if it wins) or 0 (if it loses). It then uses this to update the Q value for each state-action pair chosen. 
Run for 1000000 iterations and print out the Q-table, and optimal action for each state. 


<b>Task 2: Perceptrons. (25 points) </b>

[(from Russell and Norvig, 18.27)](https://aimacode.github.io/aima-exercises/concept-learning-exercises/ex_27/)

Consider the following set of training examples A1-A14, each with six inputs X1-X6 and one
target output:

<table>
<tr><td>Example</td><td>A1</td><td>A2</td> <td>A3</td> <td>A4</td> <td>A5</td> <td>A6</td> <td>A7</td> <td>A8</td> <td>A9</td> <td>A10</td> <td>A11</td> <td>A12</td> <td>A13</td> <td>A14</td>      </tr>
<tr><td> X1  </td><td> 1 </td><td> 1  </td><td> 1  </td><td> 1 </td><td> 1 </td><td> 1 </td><td> 1  </td><td> 0  </td><td> 0 </td><td> 0 </td><td> 0 </td><td> 0  </td><td> 0  </td><td> 0 </td></tr>
<tr><td> X2 </td><td> 0 </td><td> 0  </td><td> 0  </td><td> 1 </td><td> 1 </td><td> 0 </td><td> 0  </td><td> 1  </td><td> 1 </td><td> 0 </td><td> 1 </td><td> 0  </td><td> 1  </td><td> 1 </td></tr>
<tr> <td>	X3  </td><td> 1 </td><td> 1  </td><td> 1  </td><td> 0 </td><td> 1 </td><td> 0 </td><td> 0  </td><td> 1  </td><td> 1 </td><td> 0 </td><td> 0 </td><td> 0  </td><td> 1  </td><td> 1 </td></tr>
<tr><td>	X4  </td><td> 0 </td><td> 1  </td><td> 0  </td><td> 0 </td><td> 1 </td><td> 0 </td><td> 0  </td><td> 1  </td><td> 0 </td><td> 1 </td><td> 1 </td><td> 1  </td><td> 0  </td><td> 1 </td></tr>
<tr><td>	X5  </td><td> 0 </td><td> 0  </td><td> 1  </td><td> 1 </td><td> 0 </td><td> 1 </td><td> 1  </td><td> 0  </td><td> 1 </td><td> 1 </td><td> 0 </td><td> 0  </td><td> 1  </td><td> 0 </td></tr>
<tr><td>	X6  </td><td> 0 </td><td> 0  </td><td> 0  </td><td> 1 </td><td> 0 </td><td> 1 </td><td> 0  </td><td> 1  </td><td> 1 </td><td> 0 </td><td> 1 </td><td> 1  </td><td> 1  </td><td> 0 </td></tr>
<tr><td>	Output   </td><td> 1 </td><td> 1  </td><td> 1  </td><td> 1 </td><td> 1 </td><td> 1 </td><td> 0  </td><td> 1  </td><td> 0 </td><td> 0 </td><td> 0 </td><td> 0  </td><td> 0  </td><td> 0 </td></tr>
</table>

Write a python function to implement the perceptron learning algorithm on this data and show the final weights. Please include this in a file called perceptron.py. Assume that alpha = 0.1.
Since the data is linearly separable, your code should be able to find a solution that correctly classifies all of the data.

<b>Task 3: TensorFlow.</b>

TensorFlow is a well-known and widely used deep learning package. Google has provided a large set of tutorials and resources to help people learn to use it.
Most of them work in CoLab, so you should not need to install anything locally if you don't want to.

I've identified three tutorials that I think are helpful in getting you started with TensorFlow, and also drawing some comparisons to other techniques we've learned about.

These tutorials are great, but it's easy to wind up just clicking on things and not really absorbing the material. In order to help you think about some of the interesting issues being presented, I've presented a set of questions for you to answer. Please add a document with the answers to your repo. NOTE: Please answer these questions in your own words - do not just cut and paste from the tutorial. Cut-and-paste answers will receive zero credit.

First, please work through these tutorials on Convolutional Networks: 

https://www.tensorflow.org/tutorials/images/cnn

https://www.tensorflow.org/tutorials/images/classification

- What is a MaxPooling2D layer? What's it do?

- What's Adam? 

- What's the softmax function do? 

- What does CategoricalCrossEntropy mean?

- In the CNN example, what does the Flatten layer do?

- In the CNN example, what does the Dense layer do? 

- In the CNN example, why does the height and width get smaller for each convolutional layer?

Next, please work through this tutorial on preprocessing:
https://www.tensorflow.org/tutorials/load_data/csv

- What does it mean to normalize the data? Where else have we seen normalization? 

- Why is it a problem that the Titanic data has different types and ranges? Why did we not have to worry about this with the decision tree?

- What is a one-hot vector?

- The example that shows how to manually slice the feature dictionary uses yield instead of return.
Why is this? What's the difference between them, and why would you want to use yield?

Finally, please work through this tutorial on recurrent NNs: https://www.tensorflow.org/text/tutorials/text_classification_rnn

- As we know, encoding is a particularly important part of working with neural networks. Explain how text is encoded for an RNN. 

- What does the Bidirectional layer do? What are the advantages and disadvantages of this approach? How does it compare to the way we processed sequence data with an HMM?

- What is masking? Why do we need to use it in this example?

<b>Task 4: OpenAI (25 points): </b> 

OpenAI has a number of very interesting projects and initiatives going on. For this final question, you  may select one of them and conduct a small investigation. Depending on your time and interests, you might choose to focus more on coding and implementation, or on larger issues of fairness and transparency.

I've provided a few sample ideas here to give you a sense of size and scope. If you would like to do one of these, feel free! If you'd like to tweak one or propose your own, that's fine too. If you're going to do something original, I'd suggest checking in with me first (an emails is fine)
 to make sure the scope is appropriate.

- OpenAI's API (https://openai.com/api/) provides a programmatic interface to GPT-3. There is a per-token charge (the price varies depending on which engine you use), but new users can get $18 worth of credit, which 
should be plenty for this. Select one of the demo tasks, such as text summarization or translating natural language to code, and construct a series of prompts that you think will both demonstrate and challenge OpenAI.
What sorts of queries and inputs does it do well with? Where does it struggle? What does this tell you about how it works? Prepare a 2-3 paragraph explanation of your experiment and conclusions.
- We've heard quite a bit about some of the risks and potential harms associated with lage language models and deep learning.
Read this page: https://openai.com/blog/language-model-safety-and-misuse/ and prepare an answer (approximately one page) that summarizes the challenges presented by large language models and how OpenAI is addressing them. (note: if you do this one, you *must* use your own words to answer the question. Cutting and pasting from the article will result in a zero.)
- DALL-E 2 is OpenAI's newest effort. (https://openai.com/blog/dall-e/) How does it work? What is novel about it? Prepare an answer of approximately one page that describes DALL-E's contributions. Again, this must be written in your own words.
- Gym (https://gym.openai.com/) is a toolkit for testing reinforcement learning algorithms. This tutorial shows how to build a Deep Q network and train it
on a simple pole-balancing problem. Extend this approach to one of the more interesting environments in Gym, such as the Atari or Robotics worlds.
