from HelperFunctions import *
import torch.nn as nn

# List dictionaries to specify amount of layers in each depth - programatically build convolution layers
# As described by architechture in article  (Milletari et al, 2016)
VNET_PROPERTIES = [
    {'depth': 0, 'layers': 1},
    {'depth': 1, 'layers': 2},
    {'depth': 2, 'layers': 3},
    {'depth': 3, 'layers': 3},
    {'depth': 4, 'layers': 3},
]

# Custom Vnet built using architechture described in article (Milletari et al, 2016)
# Adjusted to allow for 5 channel input instead where each channel is an windowed transform - allows better recognition of subtle intensity differences and therefore better pancreas segmentation
class Vnet(nn.Module):
    def __init__(self, dropout=False):
        super(Vnet, self).__init__()

        # Convert input into 16 channels
        # self.in_tr = ConvBlock(in_channels=1, out_channels=16)
        self.in_tr = ConvBlock(in_channels=5, out_channels=16)  # 3 channel to account for windowed intensities

        # Down depth layers
        self.down_depth0 = VnetDepthHandler(in_channels=16, layer_count=self.__get_layer_count(0),
                                            sample_type=ResampleTypeEnum.Downsample, dropout=dropout)
        self.down_depth1 = VnetDepthHandler(in_channels=32, layer_count=self.__get_layer_count(1),
                                            sample_type=ResampleTypeEnum.Downsample, dropout=dropout)
        self.down_depth2 = VnetDepthHandler(in_channels=64, layer_count=self.__get_layer_count(2),
                                            sample_type=ResampleTypeEnum.Downsample, dropout=dropout)
        self.down_depth3 = VnetDepthHandler(in_channels=128, layer_count=self.__get_layer_count(3),
                                            sample_type=ResampleTypeEnum.Downsample, dropout=dropout)

        # Up depth layers
        self.up_depth4 = VnetDepthHandler(in_channels=256, layer_count=self.__get_layer_count(4),
                                          sample_type=ResampleTypeEnum.Upsample, out_channels=256,
                                          dropout=dropout)  # Upsample at depth 4 maintains channel size
        self.up_depth3 = VnetDepthHandler(in_channels=256, layer_count=self.__get_layer_count(3),
                                          sample_type=ResampleTypeEnum.Upsample, out_channels=128, initial_conv=True,
                                          dropout=dropout)
        self.up_depth2 = VnetDepthHandler(in_channels=128, layer_count=self.__get_layer_count(2),
                                          sample_type=ResampleTypeEnum.Upsample, out_channels=64, initial_conv=True,
                                          dropout=dropout)
        self.up_depth1 = VnetDepthHandler(in_channels=64, layer_count=self.__get_layer_count(1),
                                          sample_type=ResampleTypeEnum.Upsample, out_channels=32, initial_conv=True,
                                          dropout=dropout)

        # Final depth - set sampling type to None to prevent any resampling
        self.up_depth0 = VnetDepthHandler(in_channels=32, layer_count=self.__get_layer_count(0), sample_type=None,
                                          out_channels=16, initial_conv=True, dropout=dropout)

        # Final convolution to convert to 3 channels (number of classes - backgroun, pancreas, cancer)
        self.final_conv = ConvBlock(in_channels=16, out_channels=3, kernel_size=1, padding=0, stride=1)

    def __get_layer_count(self, depth_value):
        # Returns the number of layers based on depth level
        return get_dict_from_dicts(VNET_PROPERTIES, 'depth', depth_value)['layers']

    def forward(self, x):
        # Process input and convert to 16 channels
        out = self.in_tr(x)

        # Down depth layers, returns residual and processed data
        out, residual_0 = self.down_depth0(out)
        out, residual_1 = self.down_depth1(out)
        out, residual_2 = self.down_depth2(out)
        out, residual_3 = self.down_depth3(out)

        # Pass data through depth 4
        out = self.up_depth4(out)

        # Forward pass data up right side of V net feeding in the residuals
        out = self.up_depth3(out, residual_3, 'depth 3 test')
        out = self.up_depth2(out, residual_2, 'depth 2')
        out = self.up_depth1(out, residual_1, 'depth 1')
        out = self.up_depth0(out, residual_0, 'depth 0')

        # Convert to 3 channels
        out = self.final_conv(out)

        # Softmax final data to get likelihood probabilities of each class
        out = F.softmax(out, dim=1)

        return out


# Same depth convolution blocks as described by vnet article (Milletari et al, 2016)
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, kernel_size=5, padding=2, stride=1,
                 dropout=False):  # drop out param added for uncertainty measurement
        '''
          Convolution layer block consisting of convolution > activation.
          Convolution parameters can be adjusted to change output dimension and channels as wished.
        '''
        super(ConvBlock, self).__init__()

        # Keep in_channels=out_channels unless specified
        out_channels = in_channels if out_channels is None else out_channels

        # Initialise convolution and activation
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.activation = nn.Dropout3d(p=0.2) if dropout else nn.PReLU(out_channels)

        torch.nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x):
        return self.activation(self.conv(x))

# Convolution block at end of each depth to resample up or down according to acritecture  (Milletari et al, 2016)
class ConvResampleBlock(nn.Module):
    def __init__(self, in_channels, sample_type, kernel_size=2, stride=2,
                 dropout=False):  # drop out param added for uncertainty measurement
        '''
          Convolution resample block
        '''
        super(ConvResampleBlock, self).__init__()

        # Sample type to tell class whether to up or downsample
        self.sample_type = sample_type

        # get desired out_channels
        out_channels = self.__get_out_channels(in_channels)

        # Build convolution block and activation
        self.conv = self.__get_conv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                    stride=stride)
        self.activation = nn.Dropout3d(p=0.2) if dropout else nn.PReLU(out_channels)

        torch.nn.init.xavier_uniform_(self.conv.weight)

    def __get_conv(self, in_channels, out_channels, kernel_size, stride):
        # Upsample or down sample based on the sample type provided
        if self.sample_type == ResampleTypeEnum.Upsample:
            result = nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                        stride=stride)
        elif self.sample_type == ResampleTypeEnum.Downsample:
            result = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride)
        else:
            raise ValueError(f'ResampleTypeEnum Value Error\nSample type given: {self.sample_type}')

        return result

    def __get_out_channels(self, in_channels, out_channels=None):
        # Scale out_channels wrt the in_channels based on sampletype provided
        scale = 2 if self.sample_type == ResampleTypeEnum.Downsample else 0.5

        # Apply out_channels calculation
        out_channels = int(in_channels * scale)
        return out_channels

    def forward(self, x):
        return self.activation(self.conv(x))

# Used to build each depth layers together
class VnetDepthHandler(nn.Module):
    def __init__(self, in_channels, layer_count, sample_type=None, out_channels=None, initial_conv=False,
                 dropout=False):
        super(VnetDepthHandler, self).__init__()
        self.layer_count = layer_count
        self.in_channels = in_channels
        self.sample_type = sample_type
        self.initial_conv = initial_conv

        # Special case where initial conv requires concatination
        if initial_conv:
            self.conv_initial = ConvBlock(in_channels=self.in_channels, out_channels=out_channels)
            self.in_channels = out_channels

        # Initialise convolution block
        self.conv = ConvBlock(in_channels=self.in_channels, dropout=dropout)
        # Initialise resample convolution block if upsample or downsample is specified
        self.resample_conv = ConvResampleBlock(in_channels=self.in_channels,
                                               sample_type=self.sample_type) if self.sample_type is not None else None

    def forward(self, x, residual=None, message=''):
        out = x

        # Concatenate if residual is provided
        if residual is not None:
            out = torch.cat((out, residual), 1)

        # Pass data through number of convolutional layers determined by the amount of layers required
        for i in range(self.layer_count):
            if self.initial_conv and i == 0:
                out = self.conv_initial(out)
            else:
                out = self.conv(out)

        out += x
        residual = out

        out = self.resample_conv(out) if self.sample_type is not None else out

        # Only return residual if downsampling to save memory.
        if self.sample_type == ResampleTypeEnum.Downsample:
            return out, residual
        else:
            return out
