from keras import Model
from keras.layers import Input, Conv2D, Activation, ReLU, BatchNormalization, Flatten, Dense, Reshape, Conv2DTranspose
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from keras import backend as K
import numpy as np
class Autoencoder:
    """
    Autoencoder represents a Deep Convolutional autoencoder architecture with 
    mirrored encoder and decoder components.
    """

    def __init__(self, input_shape, conv_filters, conv_kernels, conv_strides, latent_space_dim):
        self.input_shape = input_shape # eg [width, height channel]
        self.conv_filters = conv_filters # eg [2, 4, 8] here first layer will have two filters, 2nd 4 and 3rd 8 filters
        self.conv_strides = conv_strides # eg kernel size at each layer [3, 5, 3] here 1st layer will have kernel size of 3x3, 2nd 5x5 and so on. 
        self.conv_kernels = conv_kernels # eg [1, 2, 2]  
        self.latent_space_dim = latent_space_dim
        
        self.encoder = None
        self.decoder = None
        self.model = None

        self._num_conv_layers = len(conv_filters) # no. of conv layer in the arch
        self._model_input = None
        self._shape_before_bottleneck = None
        self._build()

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()


    def _build(self):
        self._build_encoder()
        self._build_dencoder()
        self._build_autoencoder()
        
    def compile(self, learning_rate=0.0001):
        optimizer = Adam(learning_rate=learning_rate)
        mse_loss = MeanSquaredError()
        self.model.compile(optimizer=optimizer, loss=mse_loss)
    
    def train(self, x_train, batch_size, num_epochs):
        self.model.fit(x_train,
                       x_train,
                       batch_size=batch_size,
                       epochs=num_epochs,
                       shuffle=True)
        
    # -------------------------------------autoencoder------------------------
    def _build_autoencoder(self):
        model_input =  self._model_input
        model_output = self.decoder(self.encoder(model_input))
        self.model = Model(model_input, model_output, name="autoencoder")
    # -------------------------------------Decoder------------------------
    def _build_dencoder(self):
        decoder_input = self._add_dencoder_input()
        dense_layer = self._add_dense_layer(decoder_input)
        reshape_layer = self._add_reshape_layer(dense_layer)
        conv_transoise_layers = self._add_conv_transpose_layers(reshape_layer)
        decoder_output = self._add_decoder_output(conv_transoise_layers)
        self.decoder = Model(decoder_input, decoder_output, name="decoder")

    def _add_dencoder_input(self):
        return Input(shape=self.latent_space_dim)
    
    def _add_dense_layer(self, decoder_input):
        num_neuron = np.prod(self._shape_before_bottleneck) #3d array
        dense_layer = Dense(num_neuron, name="decoder_dense")(decoder_input)
        return dense_layer    
    
    def _add_reshape_layer(self, dense_layer):
        return Reshape(self._shape_before_bottleneck)(dense_layer)
    
    def _add_conv_transpose_layers(self, x):
        """Add conv transpose block."""
        # loop through all the conv layers in reverse order and stop at the first layer
        for layer_index in reversed(range(1, self._num_conv_layers)):
            # [0, 1, 2] -> [2, 1] 
            x=self._add_conv_transpose_layer(layer_index, x)
        return x
    
    def _add_conv_transpose_layer(self, layer_index, x):
        layer_num = self._num_conv_layers-layer_index
        conv_transpose_layer = Conv2DTranspose(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            name=f"decoder_conv_transpose_layer_{layer_num}"
        )
        x =  conv_transpose_layer(x)
        x = ReLU(name=f"decoder_relu_{layer_num}")(x)
        x = BatchNormalization(name=f"decoder_bn_{layer_num}")(x)
        return x
    
    def _add_decoder_output(self, x):
        conv_transpose_layer = Conv2DTranspose(
            filters=1,
            kernel_size=self.conv_kernels[0],
            strides=self.conv_strides[0],
            padding="same",
            name=f"decoder_conv_transpose_layer_{self._num_conv_layers}"
        )
        x = conv_transpose_layer(x)
        output_layer =  Activation("sigmoid", name="sigmoid_layer")(x)
        return output_layer


    # -------------------------------------Encoder------------------------

    def _build_encoder(self):
        encoder_input = self._add_encoder_input()
        conv_layers = self._add_conv_layers(encoder_input)
        bottleneck = self._add_bottleneck(conv_layers)
        self._model_input = encoder_input
        self.encoder = Model(encoder_input, bottleneck, name = "encoder")

    def _add_encoder_input(self):
        return Input(shape=self.input_shape, name="encoder_input")
    
    def _add_conv_layers(self, encoder_input):
        """Creates all convolutionals blocks in encoder"""
        x = encoder_input
        for layer_index in range(self._num_conv_layers):
            x = self._add_conv_layer(layer_index, x)
        return x
    
    def _add_conv_layer(self, layer_index, x):
        """Adds a convolutional block to graph of layers, consisting of conv
         2d + ReLu + Batch normalization"""
        layer_number = layer_index+1
        conv_layer =  Conv2D(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            name=f"encoder_conv_layer_{layer_number}"
        )
        x = conv_layer(x)
        x = ReLU(name=f"encoder_relu_{layer_number}")(x)
        x = BatchNormalization(name=f"encoder_bn_{layer_number}")(x)
        return x
    
    def _add_bottleneck(self, x):
        """Flatten data and bottleneck (Dense Layer)."""
        self._shape_before_bottleneck = K.int_shape(x)[1:] 
        x = Flatten()(x)
        x = Dense(self.latent_space_dim, name="encoder_output")(x)
        return x

if __name__ == "__main__":
    autoencoder = Autoencoder(input_shape=(28, 28, 1), 
                              conv_filters=(32, 64, 64, 64), 
                              conv_kernels=(3, 3, 3, 3), 
                              conv_strides=(1, 2, 2, 1), 
                              latent_space_dim=2)
    autoencoder.summary()
    