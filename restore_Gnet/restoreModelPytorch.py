#coding:utf-8
"""
using interpreter /home/h/anaconda2/bin/python2.7
save the parameters of 3d-gan in hdf5 file
rebuilt Gnet of 3d-gan by pytorch, using parameters saved genetate 64*64*64 matrix
"""
import torch
import torchvision.models as models
from torch.utils.serialization import load_lua
import h5py
import numpy as np
import scipy.io as sio
from torch.autograd import Variable


def saveParams():
    """
    get parameters of the torch model, and save as .hdf5 file
    :return:
    """
    # alexnet = models.alexnet()
    conv = [0,3,6,9,12]
    bn = [1,4,7,10]
    re = [2,5,8,11]
    sig = [13]

    net = load_lua('/home/h/PycharmProjects/chair.t7')  # model path
    b= net.modules

    f = open('Gnet2.txt','w')  # parameters info
    f5 = h5py.File('Gparams2.hdf5','w')  # parameters
    for i in range(14):
        if i in conv:
            # grp = f5.create_group(name='/'+str(i)+str(b[i]))  #创建本层的grp
            grp = f5.create_group(name='/' + '{:02d}'.format(i) + 'conv')  # 创建本层的grp
            bias = grp.create_dataset(name='bias'+'{:02d}'.format(i), data=b[i].bias.numpy())  #存入本层bias
            weight = grp.create_dataset(name='weight' + '{:02d}'.format(i), data=b[i].weight.numpy())  #存入本层weight
            f.write(str(i)+str(b[i])+'\n')
            f.write('bias.shape=%s,weight.shape=%s\n'%(np.array(b[i].bias.size()),
                                                       np.array(b[i].weight.size())))
            f.write('(dH,dT,dW):(%g,%g,%g);(kH,kT,kW):(%g,%g,%g)(padH,padT,padW):(%g,%g,%g)\n'%(b[i].dH, b[i].dT, b[i].dW,
                                                                                              b[i].kH, b[i].kT, b[i].kW,
                                                                                              b[i].padH, b[i].padT, b[i].padW))
            f.write('*******************************************************************************\n')
        if i in bn:
            grp = f5.create_group(name='/' + '{:02d}'.format(i) + 'bn')  # 创建本层的grp
            bias = grp.create_dataset(name='bias' + '{:02d}'.format(i), data=b[i].bias.numpy())  # 存入本层bias
            weight = grp.create_dataset(name='weight' + '{:02d}'.format(i), data=b[i].weight.numpy())  # 存入本层weight
            running_mean = grp.create_dataset(name='running_mean' + '{:02d}'.format(i), data=b[i].running_mean.numpy())  # 存入本层bias
            running_var = grp.create_dataset(name='running_var' + '{:02d}'.format(i), data=b[i].running_var.numpy())  # 存入本层bias
            save_mean = grp.create_dataset(name='save_mean' + '{:02d}'.format(i), data=b[i].save_mean.numpy())  # 存入本层bias
            save_std = grp.create_dataset(name='save_std' + '{:02d}'.format(i), data=b[i].save_std.numpy())  # 存入本层bias
            f.write(str(i) + str(b[i]) + '\n')
            f.write('eps:%g,momentum:%g,nDim:%g,affine:%s\n'%(b[i].eps, b[i].momentum, b[i].nDim, str(b[i].affine)))
            f.write('bias.shape=%s,weight.shape=%s\n' % (np.array(b[i].bias.size()),
                                                         np.array(b[i].weight.size())))
            f.write('running_mean.shape=%s,running_var.shape=%s\n' % (np.array(b[i].running_mean.size()),
                                                                     np.array(b[i].running_var.size())))
            f.write('save_mean.shape=%s,save_std.shape=%s\n' % (np.array(b[i].save_mean.size()),
                                                                np.array(b[i].save_std.size())))
            f.write('*******************************************************************************\n')
        if i in re:
            f.write(str(i) + str(b[i]) + '\n')
            f.write('*******************************************************************************\n')
        if i in sig:
            f.write(str(i) + str(b[i]) + '\n')
            f.write('*******************************************************************************\n')


    f.close()
    f5.close()
    # torch.save(net.state_dict(), 'params.pkl')



def loaddata():
    print("========================")
    print("==> Running with class: chair")
    inputs = sio.loadmat('/mnt/hgfs/shared/3dgan-release-master/demo_inputs/chair.mat')   # z_examples
    inputs = inputs['inputs']
    nb = inputs.shape[0]
    inputs_reshape = inputs.reshape(nb, 200, 1, 1, 1)
    # input = torch.DoubleTensor(nb,200,1,1,1).zero_()
    # input = torch.from_numpy(inputs).view(nb,200,1,1,1) #view并不起作用，不是很懂，pytorch教程太少，太头疼
    input = torch.DoubleTensor(inputs_reshape)
    return input, inputs_reshape

def Gnet():
    conv = [0,3,6,9,12]
    bn = [1,4,7,10]
    re = [2,5,8,11]
    sig = [13]

    # 读入数据
    input,inputs = loaddata()

    # 载入模型
    n = load_lua('/home/h/PycharmProjects/chair.t7')

    # res = n(input)
    n_modules = n.modules

    nz = 200
    ngf = 64
    """
    Shape:
        - Input: :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})` where
          :math:`D_{out} = (D_{in} - 1) * stride[0] - 2 * padding[0] + kernel\_size[0] + output\_padding[0]`
          :math:`H_{out} = (H_{in} - 1) * stride[1] - 2 * padding[1] + kernel\_size[1] + output\_padding[1]`
          :math:`W_{out} = (W_{in} - 1) * stride[2] - 2 * padding[2] + kernel\_size[2] + output\_padding[2]`

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         (in_channels, out_channels, kernel_size[0], kernel_size[1], kernel_size[2])
        bias (Tensor):   the learnable bias of the module of shape (out_channels)
    """
    net = torch.nn.Sequential(
        # 200,512,4,1,0
        torch.nn.ConvTranspose3d(in_channels=nz, out_channels=ngf * 8, kernel_size=4, stride=1, padding=0),
        torch.nn.BatchNorm3d(ngf * 8),
        torch.nn.ReLU(True),
        # 512,256,4,2,1
        torch.nn.ConvTranspose3d(in_channels=ngf * 8, out_channels=ngf * 4, kernel_size=4, stride=2, padding=1),
        torch.nn.BatchNorm3d(ngf * 4),
        torch.nn.ReLU(True),
        # 256,128,4,2,1
        torch.nn.ConvTranspose3d(in_channels=ngf * 4, out_channels=ngf * 2, kernel_size=4, stride=2, padding=1),
        torch.nn.BatchNorm3d(ngf * 2),
        torch.nn.ReLU(True),
        # 128,64,4,2,1
        torch.nn.ConvTranspose3d(in_channels=ngf * 2, out_channels=ngf * 1, kernel_size=4, stride=2, padding=1),
        torch.nn.BatchNorm3d(ngf * 1),
        torch.nn.ReLU(True),
        # 64,1,4,2,1
        torch.nn.ConvTranspose3d(in_channels=ngf * 1, out_channels=1, kernel_size=4, stride=2, padding=1),
        torch.nn.Sigmoid()
    )

    # net_modules = net.modules
    # 导入参数
    for i in range(14):
        if i in conv:
            net[i].bias.data = torch.from_numpy(n_modules[i].bias.numpy())
            net[i].weight.data = torch.from_numpy(n_modules[i].weight.numpy())
            # net[i].train = False
        if i in bn:
            # 因为没有save两个属性，所以可知gamma=weight, beta=bias
            net[i].bias.data = torch.from_numpy(n_modules[i].bias.numpy())
            net[i].weight.data = torch.from_numpy(n_modules[i].weight.numpy())  # torch.nn.parameter.Parameter类型，需要.data才是tensor类型才可以赋值

            net[i]._buffers['running_mean'] = torch.from_numpy(n_modules[i].running_mean.numpy())  # 是doubletensor类型，直接赋值

            net[i]._buffers['running_var'] = torch.from_numpy(n_modules[i].running_var.numpy())
            # net[i].save_mean.data = torch.from_numpy(n_modules[i].save_mean.numpy())  # 没有这个参数
            # net[i].save_std.data = torch.from_numpy(n_modules[i].save_std.numpy())
        net[i].train = False

    print ('get model')
    # res = n(input)
    resnet = net(Variable(input))
    res = resnet.data.numpy()
    sio.savemat('/mnt/hgfs/shared/3dgan-release-master/output/pytorchres.mat',{'inputs':inputs,'voxels':res})  # save results
    print ('are you ok')
Gnet()
# loaddata()
# saveParams()