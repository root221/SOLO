from torch import nn
class ConvBlock(nn.Module):
    def __init__(self, in_channels):
        super(ConvBlock, self).__init__()
        # Shared hyperparameters
        kernel_size = 3
        padding = 1
        channels = 256
        # Define layers
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size, stride=1, padding=padding, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=32, num_channels=channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size, stride=1, padding=padding, bias=False)
        self.gn2 = nn.GroupNorm(num_groups=32, num_channels=channels)

        self.conv3 = nn.Conv2d(channels, channels, kernel_size, stride=1, padding=padding, bias=False)
        self.gn3 = nn.GroupNorm(num_groups=32, num_channels=channels)

        self.conv4 = nn.Conv2d(channels, channels, kernel_size, stride=1, padding=padding, bias=False)
        self.gn4 = nn.GroupNorm(num_groups=32, num_channels=channels)

        self.conv5 = nn.Conv2d(channels, channels, kernel_size, stride=1, padding=padding, bias=False)
        self.gn5 = nn.GroupNorm(num_groups=32, num_channels=channels)

        self.conv6 = nn.Conv2d(channels, channels, kernel_size, stride=1, padding=padding, bias=False)
        self.gn6 = nn.GroupNorm(num_groups=32, num_channels=channels)

        self.conv7 = nn.Conv2d(channels, channels, kernel_size, stride=1, padding=padding, bias=False)
        self.gn7 = nn.GroupNorm(num_groups=32, num_channels=channels)
    
    def forward(self, x):
        x = nn.ReLU()(self.gn1(self.conv1(x)))
        x = nn.ReLU()(self.gn2(self.conv2(x)))
        x = nn.ReLU()(self.gn3(self.conv3(x)))
        x = nn.ReLU()(self.gn4(self.conv4(x)))
        x = nn.ReLU()(self.gn5(self.conv5(x)))
        x = nn.ReLU()(self.gn6(self.conv6(x)))
        x = nn.ReLU()(self.gn7(self.conv7(x)))
        return x

class CategoryBranch(nn.Module):
    def __init__(self, C=4):
        super(CategoryBranch, self).__init__()
        kernel_size = 3
        padding = 1
        channels = 256
        self.conv_block = ConvBlock(channels)
        self.conv_out = nn.Conv2d(channels, C-1, kernel_size, stride=1, padding=padding, bias=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv_block(x)  
        x = self.sigmoid(self.conv_out(x))
        return x 
       
    
class MaskBranch(nn.Module):
    def __init__(self, num_grid, channels=256):
        super(MaskBranch, self).__init__()

        # Use the defined ConvBlock
        self.conv_block = ConvBlock(channels+2)
        self.conv_out = nn.Conv2d(channels, num_grid*num_grid, kernel_size=1, stride=1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv_block(x)
        x = self.sigmoid(self.conv_out(x))
        return x

