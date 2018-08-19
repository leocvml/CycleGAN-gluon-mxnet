from mxnet import gluon, image, ndarray
from matplotlib import pyplot as plt
from mxnet.gluon import data as gdata
from mxnet.gluon import nn
import mxnet as mx
from mxnet import nd
import os
from skimage import io
import numpy as np

class conv_inst_relu(gluon.nn.HybridBlock):
    def __init__(self,filters):
        super(conv_inst_relu, self).__init__()
        self.filters = filters
        self.net = nn.HybridSequential()
        with self.net.name_scope():
                self.net.add(
                    nn.Conv2D(self.filters, kernel_size=3, padding=1, strides=2),
                    nn.InstanceNorm(),
                    nn.Activation('relu')

                )

    def hybrid_forward(self, F, x):
        return self.net(x)

class upconv(gluon.nn.HybridBlock):
    def __init__(self,filters):
        super(upconv, self).__init__()
        self.conv = nn.Conv2D(filters, kernel_size=3, padding=1,strides=1)

    def hybrid_forward(self, F, x):
        x = nd.UpSampling(x, scale=2, sample_type='nearest')
        return self.conv(x)

class upconv_inst_relu(gluon.nn.HybridBlock):
    def __init__(self, filters):
        super(upconv_inst_relu, self).__init__()

        self.filters = filters
        self.net = nn.HybridSequential()
        with self.net.name_scope():
                self.net.add(
                    upconv(filters),
                    nn.InstanceNorm(),
                    nn.Activation('relu')
                )

    def hybrid_forward(self, F, x):
        return self.net(x)

class deconv_bn_relu(gluon.nn.HybridBlock):
    def __init__(self, NumLayer, filters):
        super(deconv_bn_relu, self).__init__()
        self.NumLayer = NumLayer
        self.filters = filters
        self.net = nn.HybridSequential()
        with self.net.name_scope():
            for i in range(NumLayer-1):
                self.net.add(
                    nn.Conv2DTranspose(self.filters,kernel_size=4, padding=1, strides=2),
                    nn.InstanceNorm(),
                    nn.Activation('relu')
                )
        self.net.add(
            nn.Conv2DTranspose(3, kernel_size=4, padding=1, strides=2)
        )

    def hybrid_forward(self, F, x):
        return self.net(x)

class ResBlock(gluon.nn.HybridBlock):
    def __init__(self,filters):
        super(ResBlock, self).__init__()
        self.net = nn.HybridSequential()
        with self.net.name_scope():
            self.net.add(
                nn.Conv2D(filters, kernel_size=3, padding=1),
                nn.InstanceNorm(),
                nn.Activation('relu'),
                nn.Conv2D(filters, kernel_size=3, padding=1),
                nn.InstanceNorm(),
                nn.Activation('relu')
            )
    def hybrid_forward(self, F, x):
        out = self.net(x)
        return out + x

class Generator_256(gluon.nn.HybridBlock):
    def __init__(self):
        super(Generator_256, self).__init__()
        self.net = nn.HybridSequential()
        with self.net.name_scope():
            self.net.add(
                nn.ReflectionPad2D(3),
                nn.Conv2D(32, kernel_size=7, strides=1),
                nn.InstanceNorm(),
                nn.Activation('relu'),  #c7s1-32
                conv_inst_relu(64),
                conv_inst_relu(128),
            )
            for _ in range(9):
                self.net.add(
                        ResBlock(128)
                )
            self.net.add(
                upconv_inst_relu(64),
                upconv_inst_relu(32),
                nn.ReflectionPad2D(3),
                nn.Conv2D(3,kernel_size=7,strides=1),
                nn.Activation('sigmoid')
            )

    def hybrid_forward(self, F, x):
        return self.net(x)

class Generator_128(gluon.nn.HybridBlock):
    def __init__(self):
        super(Generator_128, self).__init__()
        self.net = nn.HybridSequential()
        with self.net.name_scope():
            self.net.add(
                nn.ReflectionPad2D(3),
                nn.Conv2D(32, kernel_size=7, strides=1),
                nn.InstanceNorm(),
                nn.Activation('relu'),  #c7s1-32
                conv_inst_relu(64),
                conv_inst_relu(128),
            )
            for _ in range(6):
                self.net.add(
                        ResBlock(128)
                )
            self.net.add(
                upconv_inst_relu(64),
                upconv_inst_relu(32),
                nn.ReflectionPad2D(3),
                nn.Conv2D(3,kernel_size=7,strides=1),
                nn.Activation('sigmoid')
            )

    def hybrid_forward(self, F, x):
        return self.net(x)

class Discriminator(gluon.nn.HybridBlock):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.HybridSequential()
        with self.net.name_scope():
            self.net.add(
                nn.Conv2D(64, kernel_size=3,strides=2,padding=1),
                nn.LeakyReLU(0.2),
                nn.Conv2D(128, kernel_size=3,strides=2,padding=1),
                nn.InstanceNorm(),
                nn.LeakyReLU(0.2),
                nn.Conv2D(256, kernel_size=3,strides=2,padding=1),
                nn.InstanceNorm(),
                nn.LeakyReLU(0.2),
                nn.Conv2D(512, kernel_size=3,strides=2,padding=1),
                nn.InstanceNorm(),
                nn.LeakyReLU(0.2),
                nn.Conv2D(1,kernel_size=1,strides=1),
            )
    def  hybrid_forward(self, F, x):
        return self.net(x)

#####################################################################################################
###
###  Dataloader
###
###
####################################################################################################

class DataSet(gluon.data.Dataset):
    def __init__(self,root,DomainAList, DomainBList):
        self.root = root
        self.DomainAList = DomainAList
        self.DomainBList = DomainBList
        self.load_images()


    def read_images(self, root):
        Aroot = root + 'trainA/'  # left_frames   #data
        Broot = root + 'trainB/'  # labels   #label
        A, B = [None] * len(self.DomainAList), [None] * len(self.DomainBList)
        for i, name in enumerate(self.DomainAList):
            A[i] = image.imread(Aroot + name)
        for i,name in enumerate(self.DomainBList):
            B[i] = image.imread(Broot + name)

        return A, B

    def load_images(self):
        A, B = self.read_images(root=self.root)

        self.A = [self.normalize_image(im) for im in A]
        self.B = [self.normalize_image(im) for im in B]

        print('read ' + str(len(self.A)) + ' examples')
        print('read ' + str(len(self.B)) + ' examples')


    def normalize_image(self, A):
        return A.astype('float32') / 255

    def __getitem__(self, item):
       # A = image.imresize(self.A[item], 256, 256)
       # B = image.imresize(self.B[item], 256, 256)       #resize
        A = self.A[item]
        B = self.B[item]
        return A.transpose((2, 0, 1)), B.transpose((2, 0, 1))


    def __len__(self):
        return len(self.A)

def LoadDataset(dir,A_list, B_list,batchsize):
    dataset = DataSet(dir, A_list, B_list )
    data_iter = gdata.DataLoader(dataset, batchsize, shuffle=True,last_batch ='discard')

    return data_iter




dataset_root = 'apple2orange/'
result_folder = 'result_' + dataset_root
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

DomainA_path ='apple2orange/trainA/'
DomainA_list = [f for f in os.listdir(DomainA_path)]

DomainB_path = 'apple2orange/trainB/'
DomainB_list = [f for f in os.listdir(DomainB_path)]

import random

random.shuffle (DomainB_list )   # if you have pair data you should shuffle it first!
batch_size = 3
train_iter = LoadDataset(dataset_root, DomainA_list, DomainB_list, batchsize=batch_size)


for d, l in train_iter:
     break

print(d.shape)
print(l.shape)
######   check image
# from skimage import io
# d = mx.ndarray.transpose(d, (0,2,3,1))
# io.imshow(d[0,:,:,:].asnumpy())
# io.show()
#
# l = mx.ndarray.transpose(l, (0,2,3,1))
# io.imshow(l[0,:,:,:].asnumpy())
#io.show()



############################################################################################
####
#### Network Arch
####
#####################################################################################
epochs = 0

ctx = mx.gpu()
G_AB = nn.HybridSequential()
with G_AB.name_scope():
    G_AB.add(
        Generator_256()    # input size 256 or 128
    )
G_AB.initialize()

G_BA = nn.HybridSequential()
with G_BA.name_scope():
    G_BA.add(
        Generator_256()
    )
G_BA.initialize()


D_A = nn.HybridSequential()
with D_A.name_scope():
    D_A.add(
        Discriminator()
    )
D_A.initialize()


D_B = nn.HybridSequential()
with D_B.name_scope():
    D_B.add(
        Discriminator()
    )
D_B.initialize()

G_AB.collect_params().reset_ctx(ctx=ctx)
G_BA.collect_params().reset_ctx(ctx=ctx)
D_A.collect_params().reset_ctx(ctx=ctx)
D_B.collect_params().reset_ctx(ctx=ctx)



GAB_trainer = gluon.Trainer(G_AB.collect_params(), 'adam', {'learning_rate': 0.0002})
GBA_trainer = gluon.Trainer(G_BA.collect_params(), 'adam', {'learning_rate': 0.0002})
DA_trainer = gluon.Trainer(D_A.collect_params(), 'adam', {'learning_rate': 0.0002})
DB_trainer = gluon.Trainer(D_B.collect_params(), 'adam', {'learning_rate': 0.0002})

cyc_loss = gluon.loss.L1Loss()
L2_loss = gluon.loss.L2Loss()


GAB_filename = 'GAB.params'
GBA_filename = 'GBA.params'
DA_filename = 'DA.params'
DB_filename = 'DB.params'

G_AB.load_params(GAB_filename, ctx=ctx)
G_BA.load_params(GBA_filename, ctx=ctx)
D_A.load_params(DA_filename, ctx=ctx)
D_B.load_params(DB_filename, ctx=ctx)


import time
from mxnet import autograd



real_label = nd.ones((batch_size,256), ctx=ctx)
fake_label = nd.zeros((batch_size,256), ctx=ctx)
lamda = 10
for epoch in range(epochs):
    tic = time.time()
    for i, (A, B) in enumerate(train_iter):
        A = A.as_in_context(ctx)
        B = B.as_in_context(ctx)


        with autograd.record():   # train A
            # train with real image
            real_A = D_A(A)
            fake_BA = G_BA(B)
            fake_A = D_A(fake_BA)

            real_label = nd.ones_like(real_A,ctx=ctx)
            fake_label = nd.zeros_like(fake_A,ctx=ctx)


            errA_real = L2_loss(real_A, real_label)
            errA_fake = L2_loss(fake_A, fake_label)
            errDA = (errA_real + errA_fake) * 0.5
        errDA.backward()
        DA_trainer.step(A.shape[0])

        with autograd.record():
            real_B = D_B(B)
            fake_AB = G_AB(A)
            fake_B = D_B(fake_AB)



            errB_real = L2_loss(real_B, real_label)
            errB_fake = L2_loss(fake_B, fake_label)
            errDB = (errB_fake + errB_real) * 0.5
        errDB.backward()
        DB_trainer.step(B.shape[0])

        with autograd.record():
            fake_AB = G_AB(A)
            rec_A = G_BA(fake_AB)
            cycA_loss = cyc_loss(rec_A,A)

            fake_B = D_B(fake_AB)


            errG_AB = L2_loss(fake_B,real_label) + lamda * cycA_loss
            errG_AB.backward()
        GAB_trainer.step(A.shape[0])

        with autograd.record():
            fake_BA = G_BA(B)
            rec_B = G_AB(fake_BA)
            cycB_loss = cyc_loss(rec_B, B)

            fake_A = D_A(fake_BA)

            errG_BA = L2_loss(fake_A, real_label) + lamda * cycB_loss
            errG_BA.backward()
        GBA_trainer.step(B.shape[0])


    # Gen_B = G_AB(A)
    # Gen_B = (Gen_B[0].asnumpy().transpose(1, 2, 0)* 255).astype(np.uint8)
    # io.imsave(result_folder+str(epoch)+'_AB.jpg',Gen_B)
    #
    # Gen_A = G_BA(B)
    # Gen_A = (Gen_A[0].asnumpy().transpose(1, 2, 0)* 255).astype(np.uint8)
    # io.imsave(result_folder+str(epoch)+'_BA.jpg',Gen_A)


    Gen_B = G_AB(A)
    Gen_B = mx.ndarray.transpose(Gen_B,(0,2,3,1))
    A = mx.ndarray.transpose(A, (0,2,3,1))

    figsize = (11, 2)
    _, axes = plt.subplots(2, A.shape[0], figsize=figsize)
    for i in range(A.shape[0]):
        axes[0][i].imshow(A[i].asnumpy())
        axes[1][i].imshow(Gen_B[i].asnumpy())
        axes[0][i].axis('off')
        axes[1][i].axis('off')

    plt.savefig(result_folder+str(epoch)+'_AB.jpg')


    Gen_A= G_BA(B)
    Gen_A = mx.ndarray.transpose(Gen_A,(0,2,3,1))
    B = mx.ndarray.transpose(B, (0,2,3,1))

    figsize = (11, 2)
    _, axes = plt.subplots(2, B.shape[0], figsize=figsize)
    for i in range(B.shape[0]):
        axes[0][i].imshow(B[i].asnumpy())
        axes[1][i].imshow(Gen_A[i].asnumpy())
        axes[0][i].axis('off')
        axes[1][i].axis('off')

    plt.savefig(result_folder+str(epoch)+'_BA.png')

    print('Epoch %2d,G_ABloss %.5f ,G_BAloss %.5f ,D_Aloss %.5f ,D_Bloss %.5f ,cycAloss %.5f ,cycBloss %.5f, time %.1f sec' % (
        epoch,
        mx.ndarray.mean(errG_AB).asscalar(),
        mx.ndarray.mean(errG_BA).asscalar(),
        mx.ndarray.mean(errDA).asscalar(),
        mx.ndarray.mean(errDB).asscalar(),
        mx.ndarray.mean(cycA_loss).asscalar(),
        mx.ndarray.mean(cycB_loss).asscalar(),
            time.time() - tic))

    G_AB.save_params(GAB_filename)
    G_BA.save_params(GBA_filename)
    D_A.save_params(DA_filename)
    D_B.save_params(DB_filename)

G_AB.save_params(GAB_filename)
G_BA.save_params(GBA_filename)
D_A.save_params(DA_filename)
D_B.save_params(DB_filename)




for Apple, Orange in train_iter:


    Gen_Orange = G_AB(Apple.as_in_context(ctx))
    Gen_Orange = mx.ndarray.transpose(Gen_Orange,(0,2,3,1))
    Apple = mx.ndarray.transpose(Apple, (0,2,3,1))

    figsize = (11, 2)
    _, axes = plt.subplots(2, Apple.shape[0], figsize=figsize)
    for i in range(Apple.shape[0]):
        axes[0][i].imshow(Apple[i].asnumpy())
        axes[1][i].imshow(Gen_Orange[i].asnumpy())
        axes[0][i].axis('off')
        axes[1][i].axis('off')

    plt.show()

    Gen_Apple = G_BA(Orange.as_in_context(ctx))
    Gen_Apple = mx.ndarray.transpose(Gen_Apple,(0,2,3,1))
    Orange = mx.ndarray.transpose(Orange, (0,2,3,1))

    figsize = (11, 2)
    _, axes = plt.subplots(2, Orange.shape[0], figsize=figsize)
    for i in range(Orange.shape[0]):
        axes[0][i].imshow(Orange[i].asnumpy())
        axes[1][i].imshow(Gen_Apple[i].asnumpy())
        axes[0][i].axis('off')
        axes[1][i].axis('off')

    plt.show()













