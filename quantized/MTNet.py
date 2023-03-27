# GENETARED BY NNDCT, DO NOT EDIT!

import torch
import pytorch_nndct as py_nndct
class MTNet(torch.nn.Module):
    def __init__(self):
        super(MTNet, self).__init__()
        self.module_0 = py_nndct.nn.Input() #MTNet::input_0
        self.module_1 = py_nndct.nn.Module('const') #MTNet::800
        self.module_2 = py_nndct.nn.Module('const') #MTNet::817
        self.module_3 = py_nndct.nn.Module('const') #MTNet::873
        self.module_4 = py_nndct.nn.Module('const') #MTNet::890
        self.module_5 = py_nndct.nn.Module('const') #MTNet::946
        self.module_6 = py_nndct.nn.Module('const') #MTNet::963
        self.module_7 = py_nndct.nn.Module('const') #MTNet::1141
        self.module_8 = py_nndct.nn.Module('const') #MTNet::1158
        self.module_9 = py_nndct.nn.Module('const') #MTNet::1178
        self.module_10 = py_nndct.nn.Module('const') #MTNet::1195
        self.module_11 = py_nndct.nn.Module('const') #MTNet::1283
        self.module_12 = py_nndct.nn.Module('const') #MTNet::1300
        self.module_13 = py_nndct.nn.Module('const') #MTNet::1330
        self.module_14 = py_nndct.nn.Module('const') #MTNet::1347
        self.module_15 = py_nndct.nn.Module('const') #MTNet::1435
        self.module_16 = py_nndct.nn.Module('const') #MTNet::1452
        self.module_17 = py_nndct.nn.Module('const') #MTNet::1550
        self.module_18 = py_nndct.nn.Module('const') #MTNet::1567
        self.module_19 = py_nndct.nn.Module('const') #MTNet::1620
        self.module_20 = py_nndct.nn.Module('const') #MTNet::1637
        self.module_21 = py_nndct.nn.Module('const') #MTNet::MTNet/1677
        self.module_22 = py_nndct.nn.Module('const') #MTNet::1755
        self.module_23 = py_nndct.nn.Module('const') #MTNet::1772
        self.module_24 = py_nndct.nn.Module('const') #MTNet::1802
        self.module_25 = py_nndct.nn.Module('const') #MTNet::1819
        self.module_26 = py_nndct.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=[7, 7], stride=[2, 2], padding=[3, 3], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Conv2d/input.2
        self.module_28 = py_nndct.nn.ReLU(inplace=True) #MTNet::MTNet/ReLU/433
        self.module_29 = py_nndct.nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=False) #MTNet::MTNet/MaxPool2d/input.4
        self.module_30 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential/BasicBlock[0]/Conv2d[conv1]/input.5
        self.module_32 = py_nndct.nn.ReLU(inplace=True) #MTNet::MTNet/Sequential/BasicBlock[0]/ReLU[relu]/input.7
        self.module_33 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential/BasicBlock[0]/Conv2d[conv2]/input.8
        self.module_35 = py_nndct.nn.Add() #MTNet::MTNet/Sequential/BasicBlock[0]/input.9
        self.module_36 = py_nndct.nn.ReLU(inplace=True) #MTNet::MTNet/Sequential/BasicBlock[0]/ReLU[relu]/input.10
        self.module_37 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential/BasicBlock[1]/Conv2d[conv1]/input.11
        self.module_39 = py_nndct.nn.ReLU(inplace=True) #MTNet::MTNet/Sequential/BasicBlock[1]/ReLU[relu]/input.13
        self.module_40 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential/BasicBlock[1]/Conv2d[conv2]/input.14
        self.module_42 = py_nndct.nn.Add() #MTNet::MTNet/Sequential/BasicBlock[1]/input.15
        self.module_43 = py_nndct.nn.ReLU(inplace=True) #MTNet::MTNet/Sequential/BasicBlock[1]/ReLU[relu]/input.16
        self.module_44 = py_nndct.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential/BasicBlock[0]/Conv2d[conv1]/input.17
        self.module_46 = py_nndct.nn.ReLU(inplace=True) #MTNet::MTNet/Sequential/BasicBlock[0]/ReLU[relu]/input.19
        self.module_47 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential/BasicBlock[0]/Conv2d[conv2]/input.20
        self.module_49 = py_nndct.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=[1, 1], stride=[2, 2], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential/BasicBlock[0]/Sequential[downsample]/Conv2d[0]/input.21
        self.module_51 = py_nndct.nn.Add() #MTNet::MTNet/Sequential/BasicBlock[0]/input.22
        self.module_52 = py_nndct.nn.ReLU(inplace=True) #MTNet::MTNet/Sequential/BasicBlock[0]/ReLU[relu]/input.23
        self.module_53 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential/BasicBlock[1]/Conv2d[conv1]/input.24
        self.module_55 = py_nndct.nn.ReLU(inplace=True) #MTNet::MTNet/Sequential/BasicBlock[1]/ReLU[relu]/input.26
        self.module_56 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential/BasicBlock[1]/Conv2d[conv2]/input.27
        self.module_58 = py_nndct.nn.Add() #MTNet::MTNet/Sequential/BasicBlock[1]/input.28
        self.module_59 = py_nndct.nn.ReLU(inplace=True) #MTNet::MTNet/Sequential/BasicBlock[1]/ReLU[relu]/input.29
        self.module_60 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential/BasicBlock[0]/Conv2d[conv1]/input.30
        self.module_62 = py_nndct.nn.ReLU(inplace=True) #MTNet::MTNet/Sequential/BasicBlock[0]/ReLU[relu]/input.32
        self.module_63 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential/BasicBlock[0]/Conv2d[conv2]/input.33
        self.module_65 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[1, 1], stride=[2, 2], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential/BasicBlock[0]/Sequential[downsample]/Conv2d[0]/input.34
        self.module_67 = py_nndct.nn.Add() #MTNet::MTNet/Sequential/BasicBlock[0]/input.35
        self.module_68 = py_nndct.nn.ReLU(inplace=True) #MTNet::MTNet/Sequential/BasicBlock[0]/ReLU[relu]/input.36
        self.module_69 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential/BasicBlock[1]/Conv2d[conv1]/input.37
        self.module_71 = py_nndct.nn.ReLU(inplace=True) #MTNet::MTNet/Sequential/BasicBlock[1]/ReLU[relu]/input.39
        self.module_72 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential/BasicBlock[1]/Conv2d[conv2]/input.40
        self.module_74 = py_nndct.nn.Add() #MTNet::MTNet/Sequential/BasicBlock[1]/input.41
        self.module_75 = py_nndct.nn.ReLU(inplace=True) #MTNet::MTNet/Sequential/BasicBlock[1]/ReLU[relu]/input.42
        self.module_76 = py_nndct.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential/BasicBlock[0]/Conv2d[conv1]/input.43
        self.module_78 = py_nndct.nn.ReLU(inplace=True) #MTNet::MTNet/Sequential/BasicBlock[0]/ReLU[relu]/input.45
        self.module_79 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential/BasicBlock[0]/Conv2d[conv2]/input.46
        self.module_81 = py_nndct.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=[1, 1], stride=[2, 2], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential/BasicBlock[0]/Sequential[downsample]/Conv2d[0]/input.47
        self.module_83 = py_nndct.nn.Add() #MTNet::MTNet/Sequential/BasicBlock[0]/input.48
        self.module_84 = py_nndct.nn.ReLU(inplace=True) #MTNet::MTNet/Sequential/BasicBlock[0]/ReLU[relu]/input.49
        self.module_85 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential/BasicBlock[1]/Conv2d[conv1]/input.50
        self.module_87 = py_nndct.nn.ReLU(inplace=True) #MTNet::MTNet/Sequential/BasicBlock[1]/ReLU[relu]/input.52
        self.module_88 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential/BasicBlock[1]/Conv2d[conv2]/input.53
        self.module_90 = py_nndct.nn.Add() #MTNet::MTNet/Sequential/BasicBlock[1]/input.54
        self.module_91 = py_nndct.nn.ReLU(inplace=True) #MTNet::MTNet/Sequential/BasicBlock[1]/ReLU[relu]/input.55
        self.module_92 = py_nndct.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential[toplayer3]/Conv2d[toplayer3_conv]/input.56
        self.module_94 = py_nndct.nn.ReLU(inplace=True) #MTNet::MTNet/Sequential[toplayer3]/ReLU[toplayer3_relu]/input.58
        self.module_95 = py_nndct.nn.Module('shape') #MTNet::MTNet/793
        self.module_96 = py_nndct.nn.Module('tensor') #MTNet::MTNet/794
        self.module_97 = py_nndct.nn.Module('cast') #MTNet::MTNet/799
        self.module_98 = py_nndct.nn.Module('mul') #MTNet::MTNet/801
        self.module_99 = py_nndct.nn.Module('cast') #MTNet::MTNet/806
        self.module_100 = py_nndct.nn.Module('floor') #MTNet::MTNet/807
        self.module_101 = py_nndct.nn.Int() #MTNet::MTNet/808
        self.module_102 = py_nndct.nn.Module('shape') #MTNet::MTNet/810
        self.module_103 = py_nndct.nn.Module('tensor') #MTNet::MTNet/811
        self.module_104 = py_nndct.nn.Module('cast') #MTNet::MTNet/816
        self.module_105 = py_nndct.nn.Module('mul') #MTNet::MTNet/818
        self.module_106 = py_nndct.nn.Module('cast') #MTNet::MTNet/823
        self.module_107 = py_nndct.nn.Module('floor') #MTNet::MTNet/824
        self.module_108 = py_nndct.nn.Int() #MTNet::MTNet/825
        self.module_109 = py_nndct.nn.Interpolate() #MTNet::MTNet/828
        self.module_110 = py_nndct.nn.Cat() #MTNet::MTNet/input.59
        self.module_111 = py_nndct.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/conv_bn_relu[conv_cat_3]/Conv2d[conv]/input.60
        self.module_113 = py_nndct.nn.ReLU(inplace=False) #MTNet::MTNet/conv_bn_relu[conv_cat_3]/ReLU[relu]/input.62
        self.module_114 = py_nndct.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential[toplayer2]/Conv2d[toplayer2_conv]/input.63
        self.module_116 = py_nndct.nn.ReLU(inplace=True) #MTNet::MTNet/Sequential[toplayer2]/ReLU[toplayer2_relu]/input.65
        self.module_117 = py_nndct.nn.Module('shape') #MTNet::MTNet/866
        self.module_118 = py_nndct.nn.Module('tensor') #MTNet::MTNet/867
        self.module_119 = py_nndct.nn.Module('cast') #MTNet::MTNet/872
        self.module_120 = py_nndct.nn.Module('mul') #MTNet::MTNet/874
        self.module_121 = py_nndct.nn.Module('cast') #MTNet::MTNet/879
        self.module_122 = py_nndct.nn.Module('floor') #MTNet::MTNet/880
        self.module_123 = py_nndct.nn.Int() #MTNet::MTNet/881
        self.module_124 = py_nndct.nn.Module('shape') #MTNet::MTNet/883
        self.module_125 = py_nndct.nn.Module('tensor') #MTNet::MTNet/884
        self.module_126 = py_nndct.nn.Module('cast') #MTNet::MTNet/889
        self.module_127 = py_nndct.nn.Module('mul') #MTNet::MTNet/891
        self.module_128 = py_nndct.nn.Module('cast') #MTNet::MTNet/896
        self.module_129 = py_nndct.nn.Module('floor') #MTNet::MTNet/897
        self.module_130 = py_nndct.nn.Int() #MTNet::MTNet/898
        self.module_131 = py_nndct.nn.Interpolate() #MTNet::MTNet/901
        self.module_132 = py_nndct.nn.Cat() #MTNet::MTNet/input.66
        self.module_133 = py_nndct.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/conv_bn_relu[conv_cat_2]/Conv2d[conv]/input.67
        self.module_135 = py_nndct.nn.ReLU(inplace=False) #MTNet::MTNet/conv_bn_relu[conv_cat_2]/ReLU[relu]/input.69
        self.module_136 = py_nndct.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential[toplayer1]/Conv2d[toplayer1_conv]/input.70
        self.module_138 = py_nndct.nn.ReLU(inplace=True) #MTNet::MTNet/Sequential[toplayer1]/ReLU[toplayer1_relu]/input.72
        self.module_139 = py_nndct.nn.Module('shape') #MTNet::MTNet/939
        self.module_140 = py_nndct.nn.Module('tensor') #MTNet::MTNet/940
        self.module_141 = py_nndct.nn.Module('cast') #MTNet::MTNet/945
        self.module_142 = py_nndct.nn.Module('mul') #MTNet::MTNet/947
        self.module_143 = py_nndct.nn.Module('cast') #MTNet::MTNet/952
        self.module_144 = py_nndct.nn.Module('floor') #MTNet::MTNet/953
        self.module_145 = py_nndct.nn.Int() #MTNet::MTNet/954
        self.module_146 = py_nndct.nn.Module('shape') #MTNet::MTNet/956
        self.module_147 = py_nndct.nn.Module('tensor') #MTNet::MTNet/957
        self.module_148 = py_nndct.nn.Module('cast') #MTNet::MTNet/962
        self.module_149 = py_nndct.nn.Module('mul') #MTNet::MTNet/964
        self.module_150 = py_nndct.nn.Module('cast') #MTNet::MTNet/969
        self.module_151 = py_nndct.nn.Module('floor') #MTNet::MTNet/970
        self.module_152 = py_nndct.nn.Int() #MTNet::MTNet/971
        self.module_153 = py_nndct.nn.Interpolate() #MTNet::MTNet/input.102
        self.module_154 = py_nndct.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential[conv_block6]/Conv2d[conv_block6_conv1]/input.73
        self.module_156 = py_nndct.nn.ReLU(inplace=True) #MTNet::MTNet/Sequential[conv_block6]/ReLU[conv_block6_relu1]/input.75
        self.module_157 = py_nndct.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential[conv_block6]/Conv2d[conv_block6_conv2]/input.76
        self.module_159 = py_nndct.nn.ReLU(inplace=True) #MTNet::MTNet/Sequential[conv_block6]/ReLU[conv_block6_relu2]/input.78
        self.module_160 = py_nndct.nn.Conv2d(in_channels=512, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential[conv_block7]/Conv2d[conv_block7_conv1]/input.79
        self.module_162 = py_nndct.nn.ReLU(inplace=True) #MTNet::MTNet/Sequential[conv_block7]/ReLU[conv_block7_relu1]/input.81
        self.module_163 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential[conv_block7]/Conv2d[conv_block7_conv2]/input.82
        self.module_165 = py_nndct.nn.ReLU(inplace=True) #MTNet::MTNet/Sequential[conv_block7]/ReLU[conv_block7_relu2]/input.84
        self.module_166 = py_nndct.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential[conv_block8]/Conv2d[conv_block8_conv1]/input.85
        self.module_168 = py_nndct.nn.ReLU(inplace=True) #MTNet::MTNet/Sequential[conv_block8]/ReLU[conv_block8_relu1]/input.87
        self.module_169 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential[conv_block8]/Conv2d[conv_block8_conv2]/input.88
        self.module_171 = py_nndct.nn.ReLU(inplace=True) #MTNet::MTNet/Sequential[conv_block8]/ReLU[conv_block8_relu2]/input
        self.module_172 = py_nndct.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential[encode_fe3]/Conv2d[0]/input.90
        self.module_173 = py_nndct.nn.ReLU(inplace=True) #MTNet::MTNet/Sequential[encode_fe3]/ReLU[1]/input.91
        self.module_174 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential[encode_feature]/conv_bn_relu[0]/Conv2d[conv]/input.92
        self.module_176 = py_nndct.nn.ReLU(inplace=False) #MTNet::MTNet/Sequential[encode_feature]/conv_bn_relu[0]/ReLU[relu]/input.94
        self.module_177 = py_nndct.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential[encode_feature]/conv_bn_relu[1]/Conv2d[conv]/input.95
        self.module_179 = py_nndct.nn.ReLU(inplace=False) #MTNet::MTNet/Sequential[encode_feature]/conv_bn_relu[1]/ReLU[relu]/input.97
        self.module_180 = py_nndct.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential[encode_feature]/conv_bn_relu[2]/Conv2d[conv]/input.98
        self.module_182 = py_nndct.nn.ReLU(inplace=False) #MTNet::MTNet/Sequential[encode_feature]/conv_bn_relu[2]/ReLU[relu]/input.100
        self.module_183 = py_nndct.nn.Module('shape') #MTNet::MTNet/1134
        self.module_184 = py_nndct.nn.Module('tensor') #MTNet::MTNet/1135
        self.module_185 = py_nndct.nn.Module('cast') #MTNet::MTNet/1140
        self.module_186 = py_nndct.nn.Module('mul') #MTNet::MTNet/1142
        self.module_187 = py_nndct.nn.Module('cast') #MTNet::MTNet/1147
        self.module_188 = py_nndct.nn.Module('floor') #MTNet::MTNet/1148
        self.module_189 = py_nndct.nn.Int() #MTNet::MTNet/1149
        self.module_190 = py_nndct.nn.Module('shape') #MTNet::MTNet/1151
        self.module_191 = py_nndct.nn.Module('tensor') #MTNet::MTNet/1152
        self.module_192 = py_nndct.nn.Module('cast') #MTNet::MTNet/1157
        self.module_193 = py_nndct.nn.Module('mul') #MTNet::MTNet/1159
        self.module_194 = py_nndct.nn.Module('cast') #MTNet::MTNet/1164
        self.module_195 = py_nndct.nn.Module('floor') #MTNet::MTNet/1165
        self.module_196 = py_nndct.nn.Int() #MTNet::MTNet/1166
        self.module_197 = py_nndct.nn.Interpolate() #MTNet::MTNet/input.101
        self.module_198 = py_nndct.nn.Module('shape') #MTNet::MTNet/1171
        self.module_199 = py_nndct.nn.Module('tensor') #MTNet::MTNet/1172
        self.module_200 = py_nndct.nn.Module('cast') #MTNet::MTNet/1177
        self.module_201 = py_nndct.nn.Module('mul') #MTNet::MTNet/1179
        self.module_202 = py_nndct.nn.Module('cast') #MTNet::MTNet/1184
        self.module_203 = py_nndct.nn.Module('floor') #MTNet::MTNet/1185
        self.module_204 = py_nndct.nn.Int() #MTNet::MTNet/1186
        self.module_205 = py_nndct.nn.Module('shape') #MTNet::MTNet/1188
        self.module_206 = py_nndct.nn.Module('tensor') #MTNet::MTNet/1189
        self.module_207 = py_nndct.nn.Module('cast') #MTNet::MTNet/1194
        self.module_208 = py_nndct.nn.Module('mul') #MTNet::MTNet/1196
        self.module_209 = py_nndct.nn.Module('cast') #MTNet::MTNet/1201
        self.module_210 = py_nndct.nn.Module('floor') #MTNet::MTNet/1202
        self.module_211 = py_nndct.nn.Int() #MTNet::MTNet/1203
        self.module_212 = py_nndct.nn.Interpolate() #MTNet::MTNet/1206
        self.module_213 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential[encode_layer0_seg]/conv_bn_relu[0]/Conv2d[conv]/input.103
        self.module_215 = py_nndct.nn.ReLU(inplace=False) #MTNet::MTNet/Sequential[encode_layer0_seg]/conv_bn_relu[0]/ReLU[relu]/input.105
        self.module_216 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential[encode_layer0_seg]/conv_bn_relu[1]/Conv2d[conv]/input.106
        self.module_218 = py_nndct.nn.ReLU(inplace=False) #MTNet::MTNet/Sequential[encode_layer0_seg]/conv_bn_relu[1]/ReLU[relu]/1240
        self.module_219 = py_nndct.nn.Add() #MTNet::MTNet/input.108
        self.module_220 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential[conv_cat_seg2]/Conv2d[0]/input.109
        self.module_222 = py_nndct.nn.ReLU(inplace=True) #MTNet::MTNet/Sequential[conv_cat_seg2]/ReLU[2]/input.111
        self.module_223 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential[conv_seg]/Conv2d[0]/input.112
        self.module_225 = py_nndct.nn.ReLU(inplace=True) #MTNet::MTNet/Sequential[conv_seg]/ReLU[2]/input.114
        self.module_226 = py_nndct.nn.Module('shape') #MTNet::MTNet/1276
        self.module_227 = py_nndct.nn.Module('tensor') #MTNet::MTNet/1277
        self.module_228 = py_nndct.nn.Module('cast') #MTNet::MTNet/1282
        self.module_229 = py_nndct.nn.Module('mul') #MTNet::MTNet/1284
        self.module_230 = py_nndct.nn.Module('cast') #MTNet::MTNet/1289
        self.module_231 = py_nndct.nn.Module('floor') #MTNet::MTNet/1290
        self.module_232 = py_nndct.nn.Int() #MTNet::MTNet/1291
        self.module_233 = py_nndct.nn.Module('shape') #MTNet::MTNet/1293
        self.module_234 = py_nndct.nn.Module('tensor') #MTNet::MTNet/1294
        self.module_235 = py_nndct.nn.Module('cast') #MTNet::MTNet/1299
        self.module_236 = py_nndct.nn.Module('mul') #MTNet::MTNet/1301
        self.module_237 = py_nndct.nn.Module('cast') #MTNet::MTNet/1306
        self.module_238 = py_nndct.nn.Module('floor') #MTNet::MTNet/1307
        self.module_239 = py_nndct.nn.Int() #MTNet::MTNet/1308
        self.module_240 = py_nndct.nn.Interpolate() #MTNet::MTNet/input.115
        self.module_241 = py_nndct.nn.Conv2d(in_channels=64, out_channels=16, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential[reg_2s_seg]/Conv2d[reg_2s_seg_conv]/input.116
        self.module_242 = py_nndct.nn.Module('shape') #MTNet::MTNet/1323
        self.module_243 = py_nndct.nn.Module('tensor') #MTNet::MTNet/1324
        self.module_244 = py_nndct.nn.Module('cast') #MTNet::MTNet/1329
        self.module_245 = py_nndct.nn.Module('mul') #MTNet::MTNet/1331
        self.module_246 = py_nndct.nn.Module('cast') #MTNet::MTNet/1336
        self.module_247 = py_nndct.nn.Module('floor') #MTNet::MTNet/1337
        self.module_248 = py_nndct.nn.Int() #MTNet::MTNet/1338
        self.module_249 = py_nndct.nn.Module('shape') #MTNet::MTNet/1340
        self.module_250 = py_nndct.nn.Module('tensor') #MTNet::MTNet/1341
        self.module_251 = py_nndct.nn.Module('cast') #MTNet::MTNet/1346
        self.module_252 = py_nndct.nn.Module('mul') #MTNet::MTNet/1348
        self.module_253 = py_nndct.nn.Module('cast') #MTNet::MTNet/1353
        self.module_254 = py_nndct.nn.Module('floor') #MTNet::MTNet/1354
        self.module_255 = py_nndct.nn.Int() #MTNet::MTNet/1355
        self.module_256 = py_nndct.nn.Interpolate() #MTNet::MTNet/1358
        self.module_257 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential[encode_layer0_drivable]/conv_bn_relu[0]/Conv2d[conv]/input.117
        self.module_259 = py_nndct.nn.ReLU(inplace=False) #MTNet::MTNet/Sequential[encode_layer0_drivable]/conv_bn_relu[0]/ReLU[relu]/input.119
        self.module_260 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential[encode_layer0_drivable]/conv_bn_relu[1]/Conv2d[conv]/input.120
        self.module_262 = py_nndct.nn.ReLU(inplace=False) #MTNet::MTNet/Sequential[encode_layer0_drivable]/conv_bn_relu[1]/ReLU[relu]/1392
        self.module_263 = py_nndct.nn.Add() #MTNet::MTNet/input.122
        self.module_264 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential[conv_cat_drivable2]/Conv2d[0]/input.123
        self.module_266 = py_nndct.nn.ReLU(inplace=True) #MTNet::MTNet/Sequential[conv_cat_drivable2]/ReLU[2]/input.125
        self.module_267 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential[conv_drivable]/Conv2d[0]/input.126
        self.module_269 = py_nndct.nn.ReLU(inplace=True) #MTNet::MTNet/Sequential[conv_drivable]/ReLU[2]/input.128
        self.module_270 = py_nndct.nn.Module('shape') #MTNet::MTNet/1428
        self.module_271 = py_nndct.nn.Module('tensor') #MTNet::MTNet/1429
        self.module_272 = py_nndct.nn.Module('cast') #MTNet::MTNet/1434
        self.module_273 = py_nndct.nn.Module('mul') #MTNet::MTNet/1436
        self.module_274 = py_nndct.nn.Module('cast') #MTNet::MTNet/1441
        self.module_275 = py_nndct.nn.Module('floor') #MTNet::MTNet/1442
        self.module_276 = py_nndct.nn.Int() #MTNet::MTNet/1443
        self.module_277 = py_nndct.nn.Module('shape') #MTNet::MTNet/1445
        self.module_278 = py_nndct.nn.Module('tensor') #MTNet::MTNet/1446
        self.module_279 = py_nndct.nn.Module('cast') #MTNet::MTNet/1451
        self.module_280 = py_nndct.nn.Module('mul') #MTNet::MTNet/1453
        self.module_281 = py_nndct.nn.Module('cast') #MTNet::MTNet/1458
        self.module_282 = py_nndct.nn.Module('floor') #MTNet::MTNet/1459
        self.module_283 = py_nndct.nn.Int() #MTNet::MTNet/1460
        self.module_284 = py_nndct.nn.Interpolate() #MTNet::MTNet/input.129
        self.module_285 = py_nndct.nn.Conv2d(in_channels=64, out_channels=3, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential[score_2drivable]/Conv2d[score_2drivable_conv]/1473
        self.module_286 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential[encode_layer0_depth]/conv_bn_relu[0]/Conv2d[conv]/input.130
        self.module_288 = py_nndct.nn.ReLU(inplace=False) #MTNet::MTNet/Sequential[encode_layer0_depth]/conv_bn_relu[0]/ReLU[relu]/input.132
        self.module_289 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential[encode_layer0_depth]/conv_bn_relu[1]/Conv2d[conv]/input.133
        self.module_291 = py_nndct.nn.ReLU(inplace=False) #MTNet::MTNet/Sequential[encode_layer0_depth]/conv_bn_relu[1]/ReLU[relu]/1507
        self.module_292 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential[conv_cat_depth_encode]/Conv2d[0]/input.135
        self.module_294 = py_nndct.nn.ReLU(inplace=True) #MTNet::MTNet/Sequential[conv_cat_depth_encode]/ReLU[2]/1523
        self.module_295 = py_nndct.nn.Add() #MTNet::MTNet/input.137
        self.module_296 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential[conv_cat_depth2]/Conv2d[0]/input.138
        self.module_298 = py_nndct.nn.ReLU(inplace=True) #MTNet::MTNet/Sequential[conv_cat_depth2]/ReLU[2]/input.140
        self.module_299 = py_nndct.nn.Module('shape') #MTNet::MTNet/1543
        self.module_300 = py_nndct.nn.Module('tensor') #MTNet::MTNet/1544
        self.module_301 = py_nndct.nn.Module('cast') #MTNet::MTNet/1549
        self.module_302 = py_nndct.nn.Module('mul') #MTNet::MTNet/1551
        self.module_303 = py_nndct.nn.Module('cast') #MTNet::MTNet/1556
        self.module_304 = py_nndct.nn.Module('floor') #MTNet::MTNet/1557
        self.module_305 = py_nndct.nn.Int() #MTNet::MTNet/1558
        self.module_306 = py_nndct.nn.Module('shape') #MTNet::MTNet/1560
        self.module_307 = py_nndct.nn.Module('tensor') #MTNet::MTNet/1561
        self.module_308 = py_nndct.nn.Module('cast') #MTNet::MTNet/1566
        self.module_309 = py_nndct.nn.Module('mul') #MTNet::MTNet/1568
        self.module_310 = py_nndct.nn.Module('cast') #MTNet::MTNet/1573
        self.module_311 = py_nndct.nn.Module('floor') #MTNet::MTNet/1574
        self.module_312 = py_nndct.nn.Int() #MTNet::MTNet/1575
        self.module_313 = py_nndct.nn.Interpolate() #MTNet::MTNet/input.141
        self.module_314 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential[conv_depth]/Conv2d[0]/input.142
        self.module_316 = py_nndct.nn.ReLU(inplace=True) #MTNet::MTNet/Sequential[conv_depth]/ReLU[2]/input.144
        self.module_317 = py_nndct.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential[conv_depth]/conv_bn_relu[3]/Conv2d[conv]/input.145
        self.module_319 = py_nndct.nn.ReLU(inplace=False) #MTNet::MTNet/Sequential[conv_depth]/conv_bn_relu[3]/ReLU[relu]/input.147
        self.module_320 = py_nndct.nn.Module('shape') #MTNet::MTNet/1613
        self.module_321 = py_nndct.nn.Module('tensor') #MTNet::MTNet/1614
        self.module_322 = py_nndct.nn.Module('cast') #MTNet::MTNet/1619
        self.module_323 = py_nndct.nn.Module('mul') #MTNet::MTNet/1621
        self.module_324 = py_nndct.nn.Module('cast') #MTNet::MTNet/1626
        self.module_325 = py_nndct.nn.Module('floor') #MTNet::MTNet/1627
        self.module_326 = py_nndct.nn.Int() #MTNet::MTNet/1628
        self.module_327 = py_nndct.nn.Module('shape') #MTNet::MTNet/1630
        self.module_328 = py_nndct.nn.Module('tensor') #MTNet::MTNet/1631
        self.module_329 = py_nndct.nn.Module('cast') #MTNet::MTNet/1636
        self.module_330 = py_nndct.nn.Module('mul') #MTNet::MTNet/1638
        self.module_331 = py_nndct.nn.Module('cast') #MTNet::MTNet/1643
        self.module_332 = py_nndct.nn.Module('floor') #MTNet::MTNet/1644
        self.module_333 = py_nndct.nn.Int() #MTNet::MTNet/1645
        self.module_334 = py_nndct.nn.Interpolate() #MTNet::MTNet/input.148
        self.module_335 = py_nndct.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential[reg_2s_depth]/conv_bn_relu[reg_2s_depth_conv_encode]/Conv2d[conv]/input.149
        self.module_337 = py_nndct.nn.ReLU(inplace=False) #MTNet::MTNet/Sequential[reg_2s_depth]/conv_bn_relu[reg_2s_depth_conv_encode]/ReLU[relu]/input.151
        self.module_338 = py_nndct.nn.Conv2d(in_channels=32, out_channels=1, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential[reg_2s_depth]/Conv2d[reg_2s_depth_conv]/1675
        self.module_339 = py_nndct.nn.Sigmoid() #MTNet::MTNet/Sequential[reg_2s_depth]/Sigmoid[reg_2s_depth_sigmoid]/1676
        self.module_340 = py_nndct.nn.Module('elemwise_mul') #MTNet::MTNet/1678
        self.module_341 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential[encode_layer0_lane]/conv_bn_relu[0]/Conv2d[conv]/input.152
        self.module_343 = py_nndct.nn.ReLU(inplace=False) #MTNet::MTNet/Sequential[encode_layer0_lane]/conv_bn_relu[0]/ReLU[relu]/input.154
        self.module_344 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential[encode_layer0_lane]/conv_bn_relu[1]/Conv2d[conv]/input.155
        self.module_346 = py_nndct.nn.ReLU(inplace=False) #MTNet::MTNet/Sequential[encode_layer0_lane]/conv_bn_relu[1]/ReLU[relu]/1712
        self.module_347 = py_nndct.nn.Add() #MTNet::MTNet/input.157
        self.module_348 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential[conv_cat_lane2]/Conv2d[0]/input.158
        self.module_350 = py_nndct.nn.ReLU(inplace=True) #MTNet::MTNet/Sequential[conv_cat_lane2]/ReLU[2]/input.160
        self.module_351 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential[conv_lane]/Conv2d[0]/input.161
        self.module_353 = py_nndct.nn.ReLU(inplace=True) #MTNet::MTNet/Sequential[conv_lane]/ReLU[2]/input.163
        self.module_354 = py_nndct.nn.Module('shape') #MTNet::MTNet/1748
        self.module_355 = py_nndct.nn.Module('tensor') #MTNet::MTNet/1749
        self.module_356 = py_nndct.nn.Module('cast') #MTNet::MTNet/1754
        self.module_357 = py_nndct.nn.Module('mul') #MTNet::MTNet/1756
        self.module_358 = py_nndct.nn.Module('cast') #MTNet::MTNet/1761
        self.module_359 = py_nndct.nn.Module('floor') #MTNet::MTNet/1762
        self.module_360 = py_nndct.nn.Int() #MTNet::MTNet/1763
        self.module_361 = py_nndct.nn.Module('shape') #MTNet::MTNet/1765
        self.module_362 = py_nndct.nn.Module('tensor') #MTNet::MTNet/1766
        self.module_363 = py_nndct.nn.Module('cast') #MTNet::MTNet/1771
        self.module_364 = py_nndct.nn.Module('mul') #MTNet::MTNet/1773
        self.module_365 = py_nndct.nn.Module('cast') #MTNet::MTNet/1778
        self.module_366 = py_nndct.nn.Module('floor') #MTNet::MTNet/1779
        self.module_367 = py_nndct.nn.Int() #MTNet::MTNet/1780
        self.module_368 = py_nndct.nn.Interpolate() #MTNet::MTNet/input.164
        self.module_369 = py_nndct.nn.Conv2d(in_channels=64, out_channels=2, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential[reg_2s_lane]/Conv2d[reg_2s_lane_conv]/input.165
        self.module_370 = py_nndct.nn.Module('shape') #MTNet::MTNet/1795
        self.module_371 = py_nndct.nn.Module('tensor') #MTNet::MTNet/1796
        self.module_372 = py_nndct.nn.Module('cast') #MTNet::MTNet/1801
        self.module_373 = py_nndct.nn.Module('mul') #MTNet::MTNet/1803
        self.module_374 = py_nndct.nn.Module('cast') #MTNet::MTNet/1808
        self.module_375 = py_nndct.nn.Module('floor') #MTNet::MTNet/1809
        self.module_376 = py_nndct.nn.Int() #MTNet::MTNet/1810
        self.module_377 = py_nndct.nn.Module('shape') #MTNet::MTNet/1812
        self.module_378 = py_nndct.nn.Module('tensor') #MTNet::MTNet/1813
        self.module_379 = py_nndct.nn.Module('cast') #MTNet::MTNet/1818
        self.module_380 = py_nndct.nn.Module('mul') #MTNet::MTNet/1820
        self.module_381 = py_nndct.nn.Module('cast') #MTNet::MTNet/1825
        self.module_382 = py_nndct.nn.Module('floor') #MTNet::MTNet/1826
        self.module_383 = py_nndct.nn.Int() #MTNet::MTNet/1827
        self.module_384 = py_nndct.nn.Interpolate() #MTNet::MTNet/1830
        self.module_385 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential[encode_layer0_det]/conv_bn_relu[0]/Conv2d[conv]/input.166
        self.module_387 = py_nndct.nn.ReLU(inplace=False) #MTNet::MTNet/Sequential[encode_layer0_det]/conv_bn_relu[0]/ReLU[relu]/input.168
        self.module_388 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential[encode_layer0_det]/conv_bn_relu[1]/Conv2d[conv]/input.169
        self.module_390 = py_nndct.nn.ReLU(inplace=False) #MTNet::MTNet/Sequential[encode_layer0_det]/conv_bn_relu[1]/ReLU[relu]/1864
        self.module_391 = py_nndct.nn.Add() #MTNet::MTNet/input.171
        self.module_392 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential[conv_cat_layer0_det2]/Conv2d[0]/input.172
        self.module_394 = py_nndct.nn.ReLU(inplace=True) #MTNet::MTNet/Sequential[conv_cat_layer0_det2]/ReLU[2]/input.174
        self.module_395 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential[conv_det]/Conv2d[0]/input.175
        self.module_397 = py_nndct.nn.ReLU(inplace=True) #MTNet::MTNet/Sequential[conv_det]/ReLU[2]/input.177
        self.module_398 = py_nndct.nn.Conv2d(in_channels=64, out_channels=8, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential[loc_0]/Conv2d[loc_0_conv2]/1908
        self.module_399 = py_nndct.nn.Conv2d(in_channels=128, out_channels=8, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential[loc_1]/Conv2d[loc_1_conv2]/1918
        self.module_400 = py_nndct.nn.Conv2d(in_channels=256, out_channels=8, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential[loc_2]/Conv2d[loc_2_conv2]/1928
        self.module_401 = py_nndct.nn.Conv2d(in_channels=512, out_channels=8, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential[loc_3]/Conv2d[loc_3_conv2]/1938
        self.module_402 = py_nndct.nn.Conv2d(in_channels=512, out_channels=8, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential[loc_4]/Conv2d[loc_4_conv2]/1948
        self.module_403 = py_nndct.nn.Conv2d(in_channels=256, out_channels=8, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential[loc_5]/Conv2d[loc_5_conv2]/1958
        self.module_404 = py_nndct.nn.Conv2d(in_channels=256, out_channels=8, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential[loc_6]/Conv2d[loc_6_conv2]/1968
        self.module_405 = py_nndct.nn.Conv2d(in_channels=64, out_channels=6, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential[conf_0]/Conv2d[conf_0_conv2]/1978
        self.module_406 = py_nndct.nn.Conv2d(in_channels=128, out_channels=6, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential[conf_1]/Conv2d[conf_1_conv2]/1988
        self.module_407 = py_nndct.nn.Conv2d(in_channels=256, out_channels=6, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential[conf_2]/Conv2d[conf_2_conv2]/1998
        self.module_408 = py_nndct.nn.Conv2d(in_channels=512, out_channels=6, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential[conf_3]/Conv2d[conf_3_conv2]/2008
        self.module_409 = py_nndct.nn.Conv2d(in_channels=512, out_channels=6, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential[conf_4]/Conv2d[conf_4_conv2]/2018
        self.module_410 = py_nndct.nn.Conv2d(in_channels=256, out_channels=6, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential[conf_5]/Conv2d[conf_5_conv2]/2028
        self.module_411 = py_nndct.nn.Conv2d(in_channels=256, out_channels=6, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential[conf_6]/Conv2d[conf_6_conv2]/2038
        self.module_412 = py_nndct.nn.Conv2d(in_channels=64, out_channels=2, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential[centerness_0]/Conv2d[centerness_0_conv]/2048
        self.module_413 = py_nndct.nn.Conv2d(in_channels=128, out_channels=2, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential[centerness_1]/Conv2d[centerness_1_conv]/2058
        self.module_414 = py_nndct.nn.Conv2d(in_channels=256, out_channels=2, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential[centerness_2]/Conv2d[centerness_2_conv]/2068
        self.module_415 = py_nndct.nn.Conv2d(in_channels=512, out_channels=2, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential[centerness_3]/Conv2d[centerness_3_conv]/2078
        self.module_416 = py_nndct.nn.Conv2d(in_channels=512, out_channels=2, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential[centerness_4]/Conv2d[centerness_4_conv]/2088
        self.module_417 = py_nndct.nn.Conv2d(in_channels=256, out_channels=2, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential[centerness_5]/Conv2d[centerness_5_conv]/2098
        self.module_418 = py_nndct.nn.Conv2d(in_channels=256, out_channels=2, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #MTNet::MTNet/Sequential[centerness_6]/Conv2d[centerness_6_conv]/2108

    def forward(self, *args):
        self.output_module_0 = self.module_0(input=args[0])
        self.output_module_1 = self.module_1(device='cpu', dtype=torch.float, data=2.0)
        self.output_module_2 = self.module_2(device='cpu', dtype=torch.float, data=2.0)
        self.output_module_3 = self.module_3(device='cpu', dtype=torch.float, data=2.0)
        self.output_module_4 = self.module_4(device='cpu', dtype=torch.float, data=2.0)
        self.output_module_5 = self.module_5(device='cpu', dtype=torch.float, data=2.0)
        self.output_module_6 = self.module_6(device='cpu', dtype=torch.float, data=2.0)
        self.output_module_7 = self.module_7(device='cpu', dtype=torch.float, data=4.0)
        self.output_module_8 = self.module_8(device='cpu', dtype=torch.float, data=4.0)
        self.output_module_9 = self.module_9(device='cpu', dtype=torch.float, data=2.0)
        self.output_module_10 = self.module_10(device='cpu', dtype=torch.float, data=2.0)
        self.output_module_11 = self.module_11(device='cpu', dtype=torch.float, data=2.0)
        self.output_module_12 = self.module_12(device='cpu', dtype=torch.float, data=2.0)
        self.output_module_13 = self.module_13(device='cpu', dtype=torch.float, data=2.0)
        self.output_module_14 = self.module_14(device='cpu', dtype=torch.float, data=2.0)
        self.output_module_15 = self.module_15(device='cpu', dtype=torch.float, data=2.0)
        self.output_module_16 = self.module_16(device='cpu', dtype=torch.float, data=2.0)
        self.output_module_17 = self.module_17(device='cpu', dtype=torch.float, data=2.0)
        self.output_module_18 = self.module_18(device='cpu', dtype=torch.float, data=2.0)
        self.output_module_19 = self.module_19(device='cpu', dtype=torch.float, data=2.0)
        self.output_module_20 = self.module_20(device='cpu', dtype=torch.float, data=2.0)
        self.output_module_21 = self.module_21(device='cpu', dtype=torch.float, data=80.0)
        self.output_module_22 = self.module_22(device='cpu', dtype=torch.float, data=2.0)
        self.output_module_23 = self.module_23(device='cpu', dtype=torch.float, data=2.0)
        self.output_module_24 = self.module_24(device='cpu', dtype=torch.float, data=2.0)
        self.output_module_25 = self.module_25(device='cpu', dtype=torch.float, data=2.0)
        self.output_module_26 = self.module_26(self.output_module_0)
        self.output_module_28 = self.module_28(self.output_module_26)
        self.output_module_29 = self.module_29(self.output_module_28)
        self.output_module_30 = self.module_30(self.output_module_29)
        self.output_module_32 = self.module_32(self.output_module_30)
        self.output_module_33 = self.module_33(self.output_module_32)
        self.output_module_35 = self.module_35(input=self.output_module_33, alpha=1, other=self.output_module_29)
        self.output_module_36 = self.module_36(self.output_module_35)
        self.output_module_37 = self.module_37(self.output_module_36)
        self.output_module_39 = self.module_39(self.output_module_37)
        self.output_module_40 = self.module_40(self.output_module_39)
        self.output_module_42 = self.module_42(input=self.output_module_40, alpha=1, other=self.output_module_36)
        self.output_module_43 = self.module_43(self.output_module_42)
        self.output_module_44 = self.module_44(self.output_module_43)
        self.output_module_46 = self.module_46(self.output_module_44)
        self.output_module_47 = self.module_47(self.output_module_46)
        self.output_module_49 = self.module_49(self.output_module_43)
        self.output_module_51 = self.module_51(input=self.output_module_47, alpha=1, other=self.output_module_49)
        self.output_module_52 = self.module_52(self.output_module_51)
        self.output_module_53 = self.module_53(self.output_module_52)
        self.output_module_55 = self.module_55(self.output_module_53)
        self.output_module_56 = self.module_56(self.output_module_55)
        self.output_module_58 = self.module_58(input=self.output_module_56, alpha=1, other=self.output_module_52)
        self.output_module_59 = self.module_59(self.output_module_58)
        self.output_module_60 = self.module_60(self.output_module_59)
        self.output_module_62 = self.module_62(self.output_module_60)
        self.output_module_63 = self.module_63(self.output_module_62)
        self.output_module_65 = self.module_65(self.output_module_59)
        self.output_module_67 = self.module_67(input=self.output_module_63, alpha=1, other=self.output_module_65)
        self.output_module_68 = self.module_68(self.output_module_67)
        self.output_module_69 = self.module_69(self.output_module_68)
        self.output_module_71 = self.module_71(self.output_module_69)
        self.output_module_72 = self.module_72(self.output_module_71)
        self.output_module_74 = self.module_74(input=self.output_module_72, alpha=1, other=self.output_module_68)
        self.output_module_75 = self.module_75(self.output_module_74)
        self.output_module_76 = self.module_76(self.output_module_75)
        self.output_module_78 = self.module_78(self.output_module_76)
        self.output_module_79 = self.module_79(self.output_module_78)
        self.output_module_81 = self.module_81(self.output_module_75)
        self.output_module_83 = self.module_83(input=self.output_module_79, alpha=1, other=self.output_module_81)
        self.output_module_84 = self.module_84(self.output_module_83)
        self.output_module_85 = self.module_85(self.output_module_84)
        self.output_module_87 = self.module_87(self.output_module_85)
        self.output_module_88 = self.module_88(self.output_module_87)
        self.output_module_90 = self.module_90(input=self.output_module_88, alpha=1, other=self.output_module_84)
        self.output_module_91 = self.module_91(self.output_module_90)
        self.output_module_92 = self.module_92(self.output_module_91)
        self.output_module_94 = self.module_94(self.output_module_92)
        self.output_module_95 = self.module_95(input=self.output_module_94, dim=2)
        self.output_module_96 = self.module_96(device='cpu', dtype=torch.int, data=self.output_module_95)
        self.output_module_97 = self.module_97(input=self.output_module_96, dtype=torch.float)
        self.output_module_98 = self.module_98(input=self.output_module_97, other=self.output_module_1)
        self.output_module_99 = self.module_99(input=self.output_module_98, dtype=torch.float)
        self.output_module_100 = self.module_100(input=self.output_module_99)
        self.output_module_101 = self.module_101(input=self.output_module_100)
        self.output_module_102 = self.module_102(input=self.output_module_94, dim=3)
        self.output_module_103 = self.module_103(device='cpu', dtype=torch.int, data=self.output_module_102)
        self.output_module_104 = self.module_104(input=self.output_module_103, dtype=torch.float)
        self.output_module_105 = self.module_105(input=self.output_module_104, other=self.output_module_2)
        self.output_module_106 = self.module_106(input=self.output_module_105, dtype=torch.float)
        self.output_module_107 = self.module_107(input=self.output_module_106)
        self.output_module_108 = self.module_108(input=self.output_module_107)
        self.output_module_109 = self.module_109(input=self.output_module_94, size=[self.output_module_101,self.output_module_108], scale_factor=None, mode='bilinear', align_corners=False)
        self.output_module_110 = self.module_110(tensors=[self.output_module_109,self.output_module_75], dim=1)
        self.output_module_111 = self.module_111(self.output_module_110)
        self.output_module_113 = self.module_113(self.output_module_111)
        self.output_module_114 = self.module_114(self.output_module_113)
        self.output_module_116 = self.module_116(self.output_module_114)
        self.output_module_117 = self.module_117(input=self.output_module_116, dim=2)
        self.output_module_118 = self.module_118(device='cpu', dtype=torch.int, data=self.output_module_117)
        self.output_module_119 = self.module_119(input=self.output_module_118, dtype=torch.float)
        self.output_module_120 = self.module_120(input=self.output_module_119, other=self.output_module_3)
        self.output_module_121 = self.module_121(input=self.output_module_120, dtype=torch.float)
        self.output_module_122 = self.module_122(input=self.output_module_121)
        self.output_module_123 = self.module_123(input=self.output_module_122)
        self.output_module_124 = self.module_124(input=self.output_module_116, dim=3)
        self.output_module_125 = self.module_125(device='cpu', dtype=torch.int, data=self.output_module_124)
        self.output_module_126 = self.module_126(input=self.output_module_125, dtype=torch.float)
        self.output_module_127 = self.module_127(input=self.output_module_126, other=self.output_module_4)
        self.output_module_128 = self.module_128(input=self.output_module_127, dtype=torch.float)
        self.output_module_129 = self.module_129(input=self.output_module_128)
        self.output_module_130 = self.module_130(input=self.output_module_129)
        self.output_module_131 = self.module_131(input=self.output_module_116, size=[self.output_module_123,self.output_module_130], scale_factor=None, mode='bilinear', align_corners=False)
        self.output_module_132 = self.module_132(tensors=[self.output_module_131,self.output_module_59], dim=1)
        self.output_module_133 = self.module_133(self.output_module_132)
        self.output_module_135 = self.module_135(self.output_module_133)
        self.output_module_136 = self.module_136(self.output_module_135)
        self.output_module_138 = self.module_138(self.output_module_136)
        self.output_module_139 = self.module_139(input=self.output_module_138, dim=2)
        self.output_module_140 = self.module_140(device='cpu', dtype=torch.int, data=self.output_module_139)
        self.output_module_141 = self.module_141(input=self.output_module_140, dtype=torch.float)
        self.output_module_142 = self.module_142(input=self.output_module_141, other=self.output_module_5)
        self.output_module_143 = self.module_143(input=self.output_module_142, dtype=torch.float)
        self.output_module_144 = self.module_144(input=self.output_module_143)
        self.output_module_145 = self.module_145(input=self.output_module_144)
        self.output_module_146 = self.module_146(input=self.output_module_138, dim=3)
        self.output_module_147 = self.module_147(device='cpu', dtype=torch.int, data=self.output_module_146)
        self.output_module_148 = self.module_148(input=self.output_module_147, dtype=torch.float)
        self.output_module_149 = self.module_149(input=self.output_module_148, other=self.output_module_6)
        self.output_module_150 = self.module_150(input=self.output_module_149, dtype=torch.float)
        self.output_module_151 = self.module_151(input=self.output_module_150)
        self.output_module_152 = self.module_152(input=self.output_module_151)
        self.output_module_153 = self.module_153(input=self.output_module_138, size=[self.output_module_145,self.output_module_152], scale_factor=None, mode='bilinear', align_corners=False)
        self.output_module_154 = self.module_154(self.output_module_91)
        self.output_module_156 = self.module_156(self.output_module_154)
        self.output_module_157 = self.module_157(self.output_module_156)
        self.output_module_159 = self.module_159(self.output_module_157)
        self.output_module_160 = self.module_160(self.output_module_159)
        self.output_module_162 = self.module_162(self.output_module_160)
        self.output_module_163 = self.module_163(self.output_module_162)
        self.output_module_165 = self.module_165(self.output_module_163)
        self.output_module_166 = self.module_166(self.output_module_165)
        self.output_module_168 = self.module_168(self.output_module_166)
        self.output_module_169 = self.module_169(self.output_module_168)
        self.output_module_171 = self.module_171(self.output_module_169)
        self.output_module_172 = self.module_172(self.output_module_91)
        self.output_module_173 = self.module_173(self.output_module_172)
        self.output_module_174 = self.module_174(self.output_module_173)
        self.output_module_176 = self.module_176(self.output_module_174)
        self.output_module_177 = self.module_177(self.output_module_176)
        self.output_module_179 = self.module_179(self.output_module_177)
        self.output_module_180 = self.module_180(self.output_module_179)
        self.output_module_182 = self.module_182(self.output_module_180)
        self.output_module_183 = self.module_183(input=self.output_module_182, dim=2)
        self.output_module_184 = self.module_184(device='cpu', dtype=torch.int, data=self.output_module_183)
        self.output_module_185 = self.module_185(input=self.output_module_184, dtype=torch.float)
        self.output_module_186 = self.module_186(input=self.output_module_185, other=self.output_module_7)
        self.output_module_187 = self.module_187(input=self.output_module_186, dtype=torch.float)
        self.output_module_188 = self.module_188(input=self.output_module_187)
        self.output_module_189 = self.module_189(input=self.output_module_188)
        self.output_module_190 = self.module_190(input=self.output_module_182, dim=3)
        self.output_module_191 = self.module_191(device='cpu', dtype=torch.int, data=self.output_module_190)
        self.output_module_192 = self.module_192(input=self.output_module_191, dtype=torch.float)
        self.output_module_193 = self.module_193(input=self.output_module_192, other=self.output_module_8)
        self.output_module_194 = self.module_194(input=self.output_module_193, dtype=torch.float)
        self.output_module_195 = self.module_195(input=self.output_module_194)
        self.output_module_196 = self.module_196(input=self.output_module_195)
        self.output_module_197 = self.module_197(input=self.output_module_182, size=[self.output_module_189,self.output_module_196], scale_factor=None, mode='bilinear', align_corners=False)
        self.output_module_198 = self.module_198(input=self.output_module_197, dim=2)
        self.output_module_199 = self.module_199(device='cpu', dtype=torch.int, data=self.output_module_198)
        self.output_module_200 = self.module_200(input=self.output_module_199, dtype=torch.float)
        self.output_module_201 = self.module_201(input=self.output_module_200, other=self.output_module_9)
        self.output_module_202 = self.module_202(input=self.output_module_201, dtype=torch.float)
        self.output_module_203 = self.module_203(input=self.output_module_202)
        self.output_module_204 = self.module_204(input=self.output_module_203)
        self.output_module_205 = self.module_205(input=self.output_module_197, dim=3)
        self.output_module_206 = self.module_206(device='cpu', dtype=torch.int, data=self.output_module_205)
        self.output_module_207 = self.module_207(input=self.output_module_206, dtype=torch.float)
        self.output_module_208 = self.module_208(input=self.output_module_207, other=self.output_module_10)
        self.output_module_209 = self.module_209(input=self.output_module_208, dtype=torch.float)
        self.output_module_210 = self.module_210(input=self.output_module_209)
        self.output_module_211 = self.module_211(input=self.output_module_210)
        self.output_module_212 = self.module_212(input=self.output_module_197, size=[self.output_module_204,self.output_module_211], scale_factor=None, mode='bilinear', align_corners=False)
        self.output_module_213 = self.module_213(self.output_module_153)
        self.output_module_215 = self.module_215(self.output_module_213)
        self.output_module_216 = self.module_216(self.output_module_215)
        self.output_module_218 = self.module_218(self.output_module_216)
        self.output_module_219 = self.module_219(input=self.output_module_218, alpha=1, other=self.output_module_212)
        self.output_module_220 = self.module_220(self.output_module_219)
        self.output_module_222 = self.module_222(self.output_module_220)
        self.output_module_223 = self.module_223(self.output_module_222)
        self.output_module_225 = self.module_225(self.output_module_223)
        self.output_module_226 = self.module_226(input=self.output_module_225, dim=2)
        self.output_module_227 = self.module_227(device='cpu', dtype=torch.int, data=self.output_module_226)
        self.output_module_228 = self.module_228(input=self.output_module_227, dtype=torch.float)
        self.output_module_229 = self.module_229(input=self.output_module_228, other=self.output_module_11)
        self.output_module_230 = self.module_230(input=self.output_module_229, dtype=torch.float)
        self.output_module_231 = self.module_231(input=self.output_module_230)
        self.output_module_232 = self.module_232(input=self.output_module_231)
        self.output_module_233 = self.module_233(input=self.output_module_225, dim=3)
        self.output_module_234 = self.module_234(device='cpu', dtype=torch.int, data=self.output_module_233)
        self.output_module_235 = self.module_235(input=self.output_module_234, dtype=torch.float)
        self.output_module_236 = self.module_236(input=self.output_module_235, other=self.output_module_12)
        self.output_module_237 = self.module_237(input=self.output_module_236, dtype=torch.float)
        self.output_module_238 = self.module_238(input=self.output_module_237)
        self.output_module_239 = self.module_239(input=self.output_module_238)
        self.output_module_240 = self.module_240(input=self.output_module_225, size=[self.output_module_232,self.output_module_239], scale_factor=None, mode='bilinear', align_corners=False)
        self.output_module_241 = self.module_241(self.output_module_240)
        self.output_module_242 = self.module_242(input=self.output_module_241, dim=2)
        self.output_module_243 = self.module_243(device='cpu', dtype=torch.int, data=self.output_module_242)
        self.output_module_244 = self.module_244(input=self.output_module_243, dtype=torch.float)
        self.output_module_245 = self.module_245(input=self.output_module_244, other=self.output_module_13)
        self.output_module_246 = self.module_246(input=self.output_module_245, dtype=torch.float)
        self.output_module_247 = self.module_247(input=self.output_module_246)
        self.output_module_248 = self.module_248(input=self.output_module_247)
        self.output_module_249 = self.module_249(input=self.output_module_241, dim=3)
        self.output_module_250 = self.module_250(device='cpu', dtype=torch.int, data=self.output_module_249)
        self.output_module_251 = self.module_251(input=self.output_module_250, dtype=torch.float)
        self.output_module_252 = self.module_252(input=self.output_module_251, other=self.output_module_14)
        self.output_module_253 = self.module_253(input=self.output_module_252, dtype=torch.float)
        self.output_module_254 = self.module_254(input=self.output_module_253)
        self.output_module_255 = self.module_255(input=self.output_module_254)
        self.output_module_256 = self.module_256(input=self.output_module_241, size=[self.output_module_248,self.output_module_255], scale_factor=None, mode='bilinear', align_corners=False)
        self.output_module_257 = self.module_257(self.output_module_153)
        self.output_module_259 = self.module_259(self.output_module_257)
        self.output_module_260 = self.module_260(self.output_module_259)
        self.output_module_262 = self.module_262(self.output_module_260)
        self.output_module_263 = self.module_263(input=self.output_module_262, alpha=1, other=self.output_module_212)
        self.output_module_264 = self.module_264(self.output_module_263)
        self.output_module_266 = self.module_266(self.output_module_264)
        self.output_module_267 = self.module_267(self.output_module_266)
        self.output_module_269 = self.module_269(self.output_module_267)
        self.output_module_270 = self.module_270(input=self.output_module_269, dim=2)
        self.output_module_271 = self.module_271(device='cpu', dtype=torch.int, data=self.output_module_270)
        self.output_module_272 = self.module_272(input=self.output_module_271, dtype=torch.float)
        self.output_module_273 = self.module_273(input=self.output_module_272, other=self.output_module_15)
        self.output_module_274 = self.module_274(input=self.output_module_273, dtype=torch.float)
        self.output_module_275 = self.module_275(input=self.output_module_274)
        self.output_module_276 = self.module_276(input=self.output_module_275)
        self.output_module_277 = self.module_277(input=self.output_module_269, dim=3)
        self.output_module_278 = self.module_278(device='cpu', dtype=torch.int, data=self.output_module_277)
        self.output_module_279 = self.module_279(input=self.output_module_278, dtype=torch.float)
        self.output_module_280 = self.module_280(input=self.output_module_279, other=self.output_module_16)
        self.output_module_281 = self.module_281(input=self.output_module_280, dtype=torch.float)
        self.output_module_282 = self.module_282(input=self.output_module_281)
        self.output_module_283 = self.module_283(input=self.output_module_282)
        self.output_module_284 = self.module_284(input=self.output_module_269, size=[self.output_module_276,self.output_module_283], scale_factor=None, mode='bilinear', align_corners=False)
        self.output_module_285 = self.module_285(self.output_module_284)
        self.output_module_286 = self.module_286(self.output_module_138)
        self.output_module_288 = self.module_288(self.output_module_286)
        self.output_module_289 = self.module_289(self.output_module_288)
        self.output_module_291 = self.module_291(self.output_module_289)
        self.output_module_292 = self.module_292(self.output_module_197)
        self.output_module_294 = self.module_294(self.output_module_292)
        self.output_module_295 = self.module_295(input=self.output_module_291, alpha=1, other=self.output_module_294)
        self.output_module_296 = self.module_296(self.output_module_295)
        self.output_module_298 = self.module_298(self.output_module_296)
        self.output_module_299 = self.module_299(input=self.output_module_298, dim=2)
        self.output_module_300 = self.module_300(device='cpu', dtype=torch.int, data=self.output_module_299)
        self.output_module_301 = self.module_301(input=self.output_module_300, dtype=torch.float)
        self.output_module_302 = self.module_302(input=self.output_module_301, other=self.output_module_17)
        self.output_module_303 = self.module_303(input=self.output_module_302, dtype=torch.float)
        self.output_module_304 = self.module_304(input=self.output_module_303)
        self.output_module_305 = self.module_305(input=self.output_module_304)
        self.output_module_306 = self.module_306(input=self.output_module_298, dim=3)
        self.output_module_307 = self.module_307(device='cpu', dtype=torch.int, data=self.output_module_306)
        self.output_module_308 = self.module_308(input=self.output_module_307, dtype=torch.float)
        self.output_module_309 = self.module_309(input=self.output_module_308, other=self.output_module_18)
        self.output_module_310 = self.module_310(input=self.output_module_309, dtype=torch.float)
        self.output_module_311 = self.module_311(input=self.output_module_310)
        self.output_module_312 = self.module_312(input=self.output_module_311)
        self.output_module_313 = self.module_313(input=self.output_module_298, size=[self.output_module_305,self.output_module_312], scale_factor=None, mode='bilinear', align_corners=False)
        self.output_module_314 = self.module_314(self.output_module_313)
        self.output_module_316 = self.module_316(self.output_module_314)
        self.output_module_317 = self.module_317(self.output_module_316)
        self.output_module_319 = self.module_319(self.output_module_317)
        self.output_module_320 = self.module_320(input=self.output_module_319, dim=2)
        self.output_module_321 = self.module_321(device='cpu', dtype=torch.int, data=self.output_module_320)
        self.output_module_322 = self.module_322(input=self.output_module_321, dtype=torch.float)
        self.output_module_323 = self.module_323(input=self.output_module_322, other=self.output_module_19)
        self.output_module_324 = self.module_324(input=self.output_module_323, dtype=torch.float)
        self.output_module_325 = self.module_325(input=self.output_module_324)
        self.output_module_326 = self.module_326(input=self.output_module_325)
        self.output_module_327 = self.module_327(input=self.output_module_319, dim=3)
        self.output_module_328 = self.module_328(device='cpu', dtype=torch.int, data=self.output_module_327)
        self.output_module_329 = self.module_329(input=self.output_module_328, dtype=torch.float)
        self.output_module_330 = self.module_330(input=self.output_module_329, other=self.output_module_20)
        self.output_module_331 = self.module_331(input=self.output_module_330, dtype=torch.float)
        self.output_module_332 = self.module_332(input=self.output_module_331)
        self.output_module_333 = self.module_333(input=self.output_module_332)
        self.output_module_334 = self.module_334(input=self.output_module_319, size=[self.output_module_326,self.output_module_333], scale_factor=None, mode='bilinear', align_corners=False)
        self.output_module_335 = self.module_335(self.output_module_334)
        self.output_module_337 = self.module_337(self.output_module_335)
        self.output_module_338 = self.module_338(self.output_module_337)
        self.output_module_339 = self.module_339(self.output_module_338)
        self.output_module_340 = self.module_340(input=self.output_module_339, other=self.output_module_21)
        self.output_module_341 = self.module_341(self.output_module_153)
        self.output_module_343 = self.module_343(self.output_module_341)
        self.output_module_344 = self.module_344(self.output_module_343)
        self.output_module_346 = self.module_346(self.output_module_344)
        self.output_module_347 = self.module_347(input=self.output_module_346, alpha=1, other=self.output_module_212)
        self.output_module_348 = self.module_348(self.output_module_347)
        self.output_module_350 = self.module_350(self.output_module_348)
        self.output_module_351 = self.module_351(self.output_module_350)
        self.output_module_353 = self.module_353(self.output_module_351)
        self.output_module_354 = self.module_354(input=self.output_module_353, dim=2)
        self.output_module_355 = self.module_355(device='cpu', dtype=torch.int, data=self.output_module_354)
        self.output_module_356 = self.module_356(input=self.output_module_355, dtype=torch.float)
        self.output_module_357 = self.module_357(input=self.output_module_356, other=self.output_module_22)
        self.output_module_358 = self.module_358(input=self.output_module_357, dtype=torch.float)
        self.output_module_359 = self.module_359(input=self.output_module_358)
        self.output_module_360 = self.module_360(input=self.output_module_359)
        self.output_module_361 = self.module_361(input=self.output_module_353, dim=3)
        self.output_module_362 = self.module_362(device='cpu', dtype=torch.int, data=self.output_module_361)
        self.output_module_363 = self.module_363(input=self.output_module_362, dtype=torch.float)
        self.output_module_364 = self.module_364(input=self.output_module_363, other=self.output_module_23)
        self.output_module_365 = self.module_365(input=self.output_module_364, dtype=torch.float)
        self.output_module_366 = self.module_366(input=self.output_module_365)
        self.output_module_367 = self.module_367(input=self.output_module_366)
        self.output_module_368 = self.module_368(input=self.output_module_353, size=[self.output_module_360,self.output_module_367], scale_factor=None, mode='bilinear', align_corners=False)
        self.output_module_369 = self.module_369(self.output_module_368)
        self.output_module_370 = self.module_370(input=self.output_module_369, dim=2)
        self.output_module_371 = self.module_371(device='cpu', dtype=torch.int, data=self.output_module_370)
        self.output_module_372 = self.module_372(input=self.output_module_371, dtype=torch.float)
        self.output_module_373 = self.module_373(input=self.output_module_372, other=self.output_module_24)
        self.output_module_374 = self.module_374(input=self.output_module_373, dtype=torch.float)
        self.output_module_375 = self.module_375(input=self.output_module_374)
        self.output_module_376 = self.module_376(input=self.output_module_375)
        self.output_module_377 = self.module_377(input=self.output_module_369, dim=3)
        self.output_module_378 = self.module_378(device='cpu', dtype=torch.int, data=self.output_module_377)
        self.output_module_379 = self.module_379(input=self.output_module_378, dtype=torch.float)
        self.output_module_380 = self.module_380(input=self.output_module_379, other=self.output_module_25)
        self.output_module_381 = self.module_381(input=self.output_module_380, dtype=torch.float)
        self.output_module_382 = self.module_382(input=self.output_module_381)
        self.output_module_383 = self.module_383(input=self.output_module_382)
        self.output_module_384 = self.module_384(input=self.output_module_369, size=[self.output_module_376,self.output_module_383], scale_factor=None, mode='bilinear', align_corners=False)
        self.output_module_385 = self.module_385(self.output_module_153)
        self.output_module_387 = self.module_387(self.output_module_385)
        self.output_module_388 = self.module_388(self.output_module_387)
        self.output_module_390 = self.module_390(self.output_module_388)
        self.output_module_391 = self.module_391(input=self.output_module_390, alpha=1, other=self.output_module_212)
        self.output_module_392 = self.module_392(self.output_module_391)
        self.output_module_394 = self.module_394(self.output_module_392)
        self.output_module_395 = self.module_395(self.output_module_394)
        self.output_module_397 = self.module_397(self.output_module_395)
        self.output_module_398 = self.module_398(self.output_module_397)
        self.output_module_399 = self.module_399(self.output_module_135)
        self.output_module_400 = self.module_400(self.output_module_113)
        self.output_module_401 = self.module_401(self.output_module_91)
        self.output_module_402 = self.module_402(self.output_module_159)
        self.output_module_403 = self.module_403(self.output_module_165)
        self.output_module_404 = self.module_404(self.output_module_171)
        self.output_module_405 = self.module_405(self.output_module_397)
        self.output_module_406 = self.module_406(self.output_module_135)
        self.output_module_407 = self.module_407(self.output_module_113)
        self.output_module_408 = self.module_408(self.output_module_91)
        self.output_module_409 = self.module_409(self.output_module_159)
        self.output_module_410 = self.module_410(self.output_module_165)
        self.output_module_411 = self.module_411(self.output_module_171)
        self.output_module_412 = self.module_412(self.output_module_397)
        self.output_module_413 = self.module_413(self.output_module_135)
        self.output_module_414 = self.module_414(self.output_module_113)
        self.output_module_415 = self.module_415(self.output_module_91)
        self.output_module_416 = self.module_416(self.output_module_159)
        self.output_module_417 = self.module_417(self.output_module_165)
        self.output_module_418 = self.module_418(self.output_module_171)
        return self.output_module_398,self.output_module_399,self.output_module_400,self.output_module_401,self.output_module_402,self.output_module_403,self.output_module_404,self.output_module_405,self.output_module_406,self.output_module_407,self.output_module_408,self.output_module_409,self.output_module_410,self.output_module_411,self.output_module_412,self.output_module_413,self.output_module_414,self.output_module_415,self.output_module_416,self.output_module_417,self.output_module_418,self.output_module_256,self.output_module_285,self.output_module_340,self.output_module_384
