import keras.backend as K
from keras.models import Model
from keras.layers import Input, GRU, Dense, Conv2D, Conv1D, concatenate, Flatten, Lambda, TimeDistributed


# Defining the prediction model.

class T2D_Prediction(Model):

    def __init__(self,
                 sensors_dimensions = [3, 3],   #Assigning the dimensions of the sensor measurements. Accelertation is 3 and CGM is converted from 1 to 3 in pre-processing.
                 patient_timepoints,    #The number of measured timepoints for each patient.
                 tau,   #The width of each window
                 cnn_filters=64, #Number of filters in each convolutional layer
                 cnn1_kernel_height=2,  #Size of the first convolutional layer filters in the intra-sensor discovery level.
                 cnn2_kernel_size=3,    #Size of the second convolutional layer filters in the intra-sensor discovery level.
                 cnn3_kernel_size=2,    #Size of the third convolutional layer filters in the intra-sensor discovery level.
                 cnn4_kernel_height=2,  #Size of the first convolutional layer filters in the inter-sensor discovery level.
                 cnn5_kernel_size=3,    #Size of the second convolutional layer filters in the inter-sensor discovery level.
                 cnn6_kernel_size=2,    #Size of the first convolutional layer filters in the inter-sensor discovery level.
                 gru_units=32,          #The number of GRU units in the recurrent layer which is equal to the number of frames. It is set to 32 as default.
                 dropout=0.1,
                 output_dim=1):


        self.nb_sensors = len(sensors_dimensions)
        window_size = 2 * tau
        cnn1_stride = 2 


        self.main_input = [Input(shape=(patient_timepoints, sensor_k_dim)) for sensor_k_dim in sensors_dimensions]  #Getting the inputs which are in the frequency domain representation using the DFT conversion.

        split_input = [Lambda(
            lambda x: K.reshape(x, (-1, patient_timepoints//tau, window_size, int(input_k.shape[2]))))(input_k)
                       for input_k in self.main_input]

        extended_dim = [Lambda(lambda x: K.expand_dims(x, axis=-1))(input_k) for input_k in split_input]

        # First convolutional layer for intra-sensor discovery.
        conv1 = [TimeDistributed(Conv2D(filters=cnn_filters,
                                        kernel_size=(cnn1_kernel_height, ext_input_k.get_shape().as_list()[-2]),
                                        activation='relu',
                                        strides=cnn1_stride,
                                        use_bias=True))(ext_input_k) for ext_input_k in extended_dim]

        flat_conv1_out = [Lambda(lambda x: K.squeeze(x, axis=3))(conv1_out_k) for conv1_out_k in conv1]

        # Second convolutional layer for intra-sensor discovery.
        conv2 = [TimeDistributed(Conv1D(filters=cnn_filters,
                                        kernel_size=cnn2_kernel_size,
                                        activation='relu',
                                        use_bias=True))(conv1_out_k) for conv1_out_k in flat_conv1_out]

        # Third convolutional layer for intra-sensor discovery.
        conv3 = [TimeDistributed(Conv1D(filters=cnn_filters,
                                        kernel_size=cnn3_kernel_size,
                                        activation='relu',
                                        use_bias=True))(conv2_out_k) for conv2_out_k in conv2]

        flat_conv3_out = [TimeDistributed(Flatten())(conv3_out_k) for conv3_out_k in conv3]

        # Merged outputs
        extended_dim2 = [TimeDistributed(Lambda(lambda x: K.expand_dims(x, axis=-1)))(conv3_out_k)
                         for conv3_out_k in flat_conv3_out]

        merged = concatenate(extended_dim2) if len(extended_dim2) > 1 else extended_dim2[0]

        # Add the "channels" dimension to meet Conv2D expected input format
        extended_dim3 = Lambda(lambda x: K.expand_dims(x, axis=-1))(merged)

        # First convolutional layer for inter-sensor discovery.
        conv4 = TimeDistributed(Conv2D(filters=cnn_filters,
                                       kernel_size=(cnn4_kernel_height, self.nb_sensors),
                                       activation='relu',
                                       use_bias=True))(extended_dim3)
        flat_conv4_out = TimeDistributed(Lambda(lambda x: K.squeeze(x, axis=2)))(conv4)

        # Second convolutional layer for inter-sensor discovery.
        conv5 = TimeDistributed(Conv1D(filters=cnn_filters,
                                       kernel_size=cnn5_kernel_size,
                                       activation='relu',
                                       use_bias=True))(flat_conv4_out)

        # Third convolutional layer for inter-sensor discovery.
        conv6 = TimeDistributed(Conv1D(filters=cnn_filters,
                                       kernel_size=cnn6_kernel_size,
                                       activation='relu',
                                       use_bias=True))(conv5)

        # Flattened output
        flat_conv6_out = TimeDistributed(Flatten())(conv6)

        # First GRU layer
        rec1_out = GRU(units=gru_units,
                       use_bias=True,
                       activation='relu',
                       dropout=dropout,
                       return_sequences=True)(flat_conv6_out)

        # Second GRU layer
        rec2_out = GRU(units=gru_units,
                       use_bias=True,
                       activation='relu',
                       dropout=dropout)(rec1_out)

        self.main_output = Dense(units=output_dim,
                                 activation="linear",
                                 use_bias=True)(rec2_out)
        
        # Lab test result & physical attributes
        lab_input = Input(shape=(8,), name='Lab_network')
        dense1 = Dense(8)
        lab_output_1 = dense1(lab_input)
        dense2 = Dense(12)
        lab_output_2 = dense2(lab_output_1)

        # Concatenation of two networks
        combined_output = K.layers.concatenate([rec2_out, lab_output_2])

        self.main_output = Dense(units=output_dim,
                         activation="linear",
                         use_bias=True)(combined_output)

        super(T2D_Prediction, self).__init__(inputs=self.main_input, outputs=[self.main_output])