# Overview

This thesis was written by Asbjørn Steffensen & Daniel Sørensen from the IT University of Copenhagen during spring 2018.
The repo contains the actual thesis, along with the code created to solve the proposed research questions. 
The [CommandCenter bot](https://github.com/davechurchill/commandcenter "CommandCenter bot by David Churchill") by David Churchill is used.

# Contents

 * Thesis 
 * C++ code for extracting data from replays using the [StarCraft II API](https://github.com/Blizzard/s2client-api) by Blizzard
 * Java code for transforming extracted data into match-up files, for training a neural network
 * Python code to train neural networks using Keras with Tensorflow backend
 * Neural networks trained from data
 
Following is the title and abstract formulated for this thesis.

# Replay-Based Prediction of Opponent Game State in StarCraft II

After creating artificial intelligence (AI) capable of playing games like chess and Go, researchers turn their eyes towards real-time strategy (RTS) games, like StarCraft II. As StarCraft II has more than 10<sup>1000</sup> different states, accurately predicting the opponent state plays a significant role.  This paper aims to answer three research questions, specifically how much data is available for training an algorithm, how accurately an algorithm can predict thegame state, and how an AI would be able to use it.  More than half a million replays were found, usable for training algorithms.  Mining 58,807 replays, 5,497,618 states were extracted for training neural networks.  The data from one  match-up,  consisting  of  616,531  points  of  data,  were  used  to  train  k-Nearest Neighbour (k-NN) and random forest.  Comparing the algorithms, the neural networks were found to be vastly superior.  When looking at the accuracy for select attributes, the network was found to be 70% more accurate than k-NN and random forest. k-NN was slightly better at classifying the number of opponent bases, but making a prediction is approximately 5900 times slower.  To test if the network could be used in real-time, a module was implemented in the CommandCenter bot, enabling it to ultimately store predicted states.  We believe the predicted data can be used to determine the optimal time for scouting, attacking, or if and when to change strategy.
