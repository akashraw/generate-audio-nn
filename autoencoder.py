from keras import Model
from keras.layers import Input, Conv2D, ReLU, BatchNormalization, Flatten, Dense
from keras import backend as K
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

        self._build()

    def _build(self):
        self._build_encoder()
        self._build_dencoder()
        self._build_autoencoder()

    def _build_encoder(self):
        encoder_input = self._add_encoder_input()
        conv_layers = self._add_conv_layers(encoder_input)
        bottleneck = self._add_bottleneck(conv_layers)
        self.encoder = Model(encoder_input, bottleneck, name = "encoder")

    def _add_encoder_input(self):
        return Input(shape=self.input_shape, name="encoder_input")
    
    def _add_conv_layers(self, encoder_input):
        """Creates all convolutionals blocks in encoder"""
        x = encoder_input
        for layer_index in range(self._num_conv_layers):
            x = self._add_conv_layer(layer_index, x)
        return x
    
    def _add_conv_layers(self, layer_index, x):
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
    
    