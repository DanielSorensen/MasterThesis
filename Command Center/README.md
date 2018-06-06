# Extended CommandCenter

The CommandCenter Bot extended to contain a prediction module.
The module periodically writes an observed state to a file, which an external process reads. 
This in turns uses a Keras model to make a prediction, which it writes to a new file. 
The bot module then reads the file containing the prediction, storing it, allowing the other modules to use the prediction.
The external process has to run before starting CommandCenter.

The original, unaltered CommandCenter bot can be found [here](https://github.com/davechurchill/commandcenter).
