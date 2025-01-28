import numpy as np

class SEP:


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
                 X = None):
        '''
        Practically, this classdoes the following:
        instead of training the hidden space transition we use matrix X (matrix
        of representational objects X.shape[0] -- number of representative objects,
        X.shape[1] -- number of features) to transfer to a hidden layer with contains
        info about proximity of observed objects to the representational ones.
        With the learning rule provided in turner's article we train the layer
        that encodes transition from the hidden layer to the output_layer aka predicted categories

        Args:
            hidden_space_shape : dimentionality of the hidden space -- which is the kernel
                equals to the number of exemplars

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
        self.X = X

        kernels = {}
        kernels['exponential'] = self.__exponential_kernel__
        kernels['gaussian'] = self.__gaussian_kernel__
        self.kernels = kernels

        self.P = np.zeros((hidden_space_shape, output_space_shape ))

    def fit(self, s, f = None, kernel = None, *args):
        '''
        Black magic happens here which I'm hoping I remember tomorrow
        Args:
            s (np.array) : set of stimuli (s.shape[0] -- numbers of datapoints)
        Returns:
            self
        '''

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

        for t in range(s.shape[0]):
            K = kernel_func(self.X, s[t], *args)
            a = np.repeat(K, self.output_space_shape).reshape(K.shape[0], -1)
            b = np.repeat(f[t].reshape(-1, f[t].shape[0]), self.hidden_space_shape, axis = 0).reshape(-1, f[t].shape[0])
            self.P = self.dr * self.P + self.lr*a*(b-self.omega * self.P)


        return self

    def predict(self, s, *args):
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

        predictions = np.matmul(total_K,self.P)
        return predictions
