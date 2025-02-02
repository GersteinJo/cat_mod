import numpy as np
from minisom import MiniSom
from time import time
import types
import scipy as sp

class SEP_SOM:

    def __exponential_kernel__(self, X, s, *args):
        '''
        X (np.array) : exemplar space
        s (np.array) : a stimulus
        *args : additional arguments for the kernel (NOT IMPLEMENTED YET)
        '''
        return np.exp(-self.delta*np.linalg.norm(X - s, axis = 1))

    def __gaussian_kernel__(self, X, s, *args):
        '''
        X (np.array) : exemplar space
        s (np.array) : a stimulus
        *args : additional arguments for the kernel (NOT IMPLEMENTED YET)
        '''
        return np.exp(-self.delta*np.power(np.linalg.norm(X - s, axis = 1), 2))

    def __init__(self, hidden_space_shape, output_space_shape,
                 delta = 1, lr = 0.9, omega = 1, dr = 0.1,
                 X = None, seed = None, logger = None):
        '''
        Practically, this class does the following:
        instead of training the hidden space transition we use matrix X (matrix
        of representational objects X.shape[0] -- number of representative objects,
        X.shape[1] -- number of features) to transfer to a hidden layer with contains
        info about proximity of observed objects to the representational ones.
        With the learning rule provided in turner's article we train the layer
        that encodes transition from the hidden layer to the output_layer aka predicted categories

        Args:
            hidden_space_shape : dimentionality of the hidden space -- which is the kernel
                equals to the number of exemplars -- it is size of square SOM squared

            output_space_shape : number of classes; ps in the article the classes
                are ohe-ed, so we assume the same thing

            delta (float) : reversed temperature for kernel

            lr (float) : learning rate

            omega (float) : constant for introducing rescola-wagner

            dr (float) : decay rate for trained layer (than will aparently
                be used as a matrix(?) eq on p 33 of turner)

            X (np.array) : representational points that should be provided
                PS: X.shape[0]==hidden_space_shape!
        Returns:
            None
        '''
        self.hidden_space_shape = hidden_space_shape
        self.output_space_shape = output_space_shape
        self.delta = delta
        self.lr = lr
        self.omega = omega
        self.dr = dr

        kernels = {}
        kernels['exponential'] = self.__exponential_kernel__
        kernels['gaussian'] = self.__gaussian_kernel__
        self.kernels = kernels

        self.som = None


        # self.X = X

        self.P = np.zeros((hidden_space_shape, output_space_shape ))
        # self.P = np.ones((hidden_space_shape, output_space_shape ))/output_space_shape
        self.logger = logger




    def fit(self, s, f = None, kernel = None, seed = None, pretrain_som = True, verbose = True, *args):
        '''
        Black magic happens here which I'm hoping I remember tomorrow
        Args:
            s (np.array) : set of stimuli (s.shape[0] -- numbers of datapoints)
            f (np.array) : category of a stimulus
            kernel (string or function in form
                def func(X, s, **args) where X -- exemplars, *args -- additional arguements) : kernel function
                default 'exponential'
            seed (int) : seed for pseudo random generation default int(time.time())
            pretrain_som (boolean) : if checked true som is trained in advance and not while the main layer is trained
            *args : additional arguements passed to the kernel

        Returns:
            self
        '''
        if len(f.shape) == 1:
            original_f = f.copy()
            f = np.zeros((f.shape[0], np.unique(f).shape[0]))
            f[np.arange(f.shape[0]),original_f] = 1


        if seed is None:
            seed = int(time())

        np.random.seed(seed)

        if kernel is None:
            kernel = 'exponential'

        if kernel in self.kernels.keys():
            kernel_func = self.kernels[kernel]
        elif isinstance(kernel, types.FunctionType):
            kernel_func = kernel
        else:
            print("kernel should be in the following list: ", self.kernels.keys(), ", or be a callable function")
            return

        self.kernel_func = kernel_func



        if pretrain_som:
            if self.som is None:
                self.som = MiniSom(x=int(np.sqrt(self.hidden_space_shape)),
                          y=int(np.sqrt(self.hidden_space_shape)),
                          input_len=s.shape[1], sigma=1.0, learning_rate=0.5,
                          neighborhood_function='gaussian')
                self.som.pca_weights_init(s)

            # Train the SOM with the data
            self.som.train(s, 1000, verbose=verbose)

            # Create a 2D grid of neurons' weights
            weights = self.som.get_weights()
            weights = weights.reshape(weights.shape[0]*weights.shape[1], weights.shape[2])
            self.X = weights

            for t in range(s.shape[0]):
                # K = np.exp(-self.delta*np.linalg.norm(self.X - s[t], axis = 1)) #array of shape (X.shape[0],)
                K = kernel_func(self.X, s[t], *args)
                a = np.repeat(K, self.output_space_shape).reshape(K.shape[0], -1)
                b = np.repeat(f[t].reshape(-1, f[t].shape[0]), self.hidden_space_shape, axis = 0).reshape(-1, f[t].shape[0])
                acc = int(np.argmax(np.matmul(K,self.P)) == np.argmax(f[t]))
                # print(t)
                # print(np.matmul(K,self.P), f[t], end = " \t")
                # print(acc)
                self.P = self.dr * self.P + self.lr*a*(b-self.omega * self.P)
                if self.logger:
                    self.logger.log(
                        {"pred_correct": acc},
                        step = t
                    )
        else:
            # this part seems not only rather excessive it might also have conversion issues as well as it seems to be unpredictable
            # I believe, a better alternative would be having a selected set of samples that would be a plain grid in the feature space
            # rather than using the som alltogether since it is primarily an embedding algorithm rather than active learning
            # especially, since it is so-called topological a.k.a saving structure and when there's only a few points there's no
            # definable structure to save
            #
            # what can probably be a better option is to initialize exemplar space as a kind of grid and
            # primarily collect examples and then once some threshold amount is acquired implement som
            # however, to justify this some kind of actual evidence needs to be provided I guess (probably increase in )
            #
            # we can also consider training in batches but then again we can't use the perks if pca initialization to the fullest it is
            # (howbeit, it is questionable whether it is justifiable to use it in the first place but then again using som it self also need justification IMHO)
            #
            if self.som is None:
                self.som = MiniSom(x=int(np.sqrt(self.hidden_space_shape)),
                          y=int(np.sqrt(self.hidden_space_shape)),
                          input_len=s.shape[1], sigma=1.0, learning_rate=0.5,
                          neighborhood_function='gaussian')
            for t in range(s.shape[0]):
                self.som.train(s[t].reshape(1, s[t].shape[0]), 10, verbose=verbose)

                weights = self.som.get_weights()
                weights = weights.reshape(weights.shape[0]*weights.shape[1], weights.shape[2])
                self.X = weights

                # K = np.exp(-self.delta*np.linalg.norm(self.X - s[t], axis = 1)) #array of shape (X.shape[0],)
                K = kernel_func(self.X, s[t], *args)
                a = np.repeat(K, self.output_space_shape).reshape(K.shape[0], -1)
                b = np.repeat(f[t].reshape(-1, f[t].shape[0]), self.hidden_space_shape, axis = 0).reshape(-1, f[t].shape[0])
                acc = int(np.argmax(np.matmul(K,self.P)) == np.argmax(f[t]))
                # print(t)
                # print(np.matmul(K,self.P), f[t], end = " \t")
                # print(acc)
                self.P = self.dr * self.P + self.lr*a*(b-self.omega * self.P)
                if self.logger:
                    self.logger.log(
                        {"pred_correct": acc}, 
                        step = t
                    )


        return self

    def predict_proba(self, s, *args):
        '''
        This one is also messed up af; oh, scandinavian Gods, spare me this doom
        Args:
            s (np.array) : set of stimuli (s.shape[0] -- numbers of datapoints)
        Returns:
            predictions :  predictions.shape[0] = s.shape[0],
                predictions.shape[1] = self.output_space_shape
        '''
        total_K = np.zeros((0, self.X.shape[0]))
        for t in range(s.shape[0]):
            K = self.kernel_func(self.X, s[t], *args)
            total_K = np.concatenate((total_K, K.reshape(1, K.shape[0])))
        # total_K = np.sum(
        #     + np.repeat(self.X.reshape(-1,self.X.shape[0],self.X.shape[1]), s.shape[0], axis = 0)
        #     - np.repeat(s.reshape(s.shape[0], -1, s.shape[1]), self.X.shape[0], axis = 1)
        #     , axis=-1).T # this one is just messed up, but i need to think how to transfer this
        #                  # to fitting method so it does not last forever
        # print(self.P.shape, total_K.shape, s.shape)

        predictions = sp.special.softmax(np.matmul(total_K,self.P), axis = 1)
        return predictions

    def predict(self, s, *args):
        predictions = self.predict_proba(s, *args)
        return(np.argmax(predictions, axis = 1))
