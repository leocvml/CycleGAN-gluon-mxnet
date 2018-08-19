# CycleGAN-gluon-mxnet #
this repo attemps to reproduce [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks(CycleGAN)](https://arxiv.org/pdf/1703.10593.pdf) use gluon reimplementation

## Quick start ##
1. download dataset (my sample is samll set apple <-> orange)

you can download complete dataset from this link
[dataset website](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/)
 
2. train 
3. inference ( weighting is trained by complete apple2orange dataset)

## Requirements ##
mxnet 1.1.0 
## Abstract ##
**Image to image translation :** learn the mapping between an input image and an output image using a training set of aligned image pair.However for many tasks, paired training data will not be available.

**present an approach for learning translate an image from a source domain X to a target domain Y in the absence of paired examples.**


## Network architecture ##
### generate ##
```
The network consists of:
c7s1-32,d64,d128,R128,R128,R128,
R128,R128,R128,R128,R128,R128,u64,u32,c7s1-3
```
```python
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
```
### discriminator ###
```
use kernel size = 3
```
```python
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
```
## training step ##
**train Discriminator**
1.  Da aims to distinguish between translated samples G(B) and real smaples A
```python 
with autograd.record():   # train A

    real_A = D_A(A)  # distinguish real image A
    fake_BA = G_BA(B) #generate fake A image from B
    fake_A = D_A(fake_BA)# distinguish fake image

    real_label = nd.ones_like(real_A,ctx=ctx)
    fake_label = nd.zeros_like(fake_A,ctx=ctx)


    errA_real = L2_loss(real_A, real_label)
    errA_fake = L2_loss(fake_A, fake_label)
    errDA = (errA_real + errA_fake) * 0.5
errDA.backward()
DA_trainer.step(A.shape[0])


```
**train generate**
1. generate fake_B from domain A
2. generate reconstruct A from fake B
3. calculate cycle consistency loss(recA,A) (lamda =10)
4. use fake_B image fool disciminator B 
```python
with autograd.record():
    fake_AB = G_AB(A)
    fake_A = G_BA(fake_AB)
    cycA_loss = cyc_loss(fake_A,A)

    fake_B = D_B(fake_AB)


    errG_AB = L2_loss(fake_B,real_label) + lamba * cycA_loss
    errG_AB.backward()
GAB_trainer.step(A.shape[0])
```

# Result #
**apple2Orange**
![](https://github.com/leocvml/CycleGAN-gluon-mxnet/blob/master/result/a2o_infer.PNG)


**Orange2apple**
![](https://github.com/leocvml/CycleGAN-gluon-mxnet/blob/master/result/o2a_infer.PNG)


