# neural_networks_from_scatch_python
Feedforward neural networks in python using only numpy!
This was my "Facharbeit" for my cs honours class in grammar school.
It basically is a "small paper".
The code is written in english.
The paper however, is only available in german.
main.py contains the implementation, as well as a small demonstration
of the key functions at the bottom. The functions and methods are
all commented/described. The implementation includes all
necessary components required for training a neural network,
such as f.e. an activation function and an implementation of both
a feedforward method and the backpropagation algorithm.
To run the demonstration example, please download the .csv-file
of the MNIST dataset from this website (https://www.python-course.eu/neural_network_mnist.php)
("Reading the MNIST data set"), place the files into the MNIST_data_csv-folder and run
csvToPklMNIST.py.
You can also draw your own image and let the neural network predict it!

This project received the Dr.Hans-Riegel (see [here](https://www.hans-riegel-fachpreise.com/ausgezeichnete-arbeiten/details/?no_cache=1&tx_alumni_pi1%5Bpaper%5D=72&tx_alumni_pi1%5Baction%5D=show&tx_alumni_pi1%5Bcontroller%5D=Paper&cHash=c551128a37614c3ba603f27dca3c9a23)) award for
outstanding scientific work ("herausragende wissenschaftliche Arbeit") in 2019.
It ranked 2nd place in computer science in the respective contest at the
Johannes Gutenberg University in Mainz, which represented the state of Rhineland-Palatinate,
as it was the only participating university in said state (it accepted entries from the entire state).


# Performance on MNIST
An exemplary neural network achieves a 95.16% test-accuracy after one training epoch.
