import torch
import time
from tqdm import tqdm
import torch.utils.data as data_utils
import utils as utt
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import numpy as np
import math
import torch.nn as nn
import loss_funcs as lf
from skimage.transform import rescale
from .. import image_utils as iu
import torch as T
from .. import utils as ut
import optimizers as opt
from scipy.ndimage.filters import gaussian_filter
#import net2net as n2n

loss_dict = {"categorical_crossentropy":F.nll_loss,
             "binary_crossentropy": F.binary_cross_entropy,
             "dice_loss": lf.dice_loss,
             "dice_loss_modified": lf.dice_loss_modified,
             "seg_binary_cross_entropy": lf.seg_binary_cross_entropy,
             "mse":torch.nn.MSELoss(size_average=False),
             "L1Loss":torch.nn.L1Loss(),
             "bce_localize":lf.bce_localize}

optimizer_dict = {"adadelta":lambda model, lr: optim.Adadelta(model.parameters(), lr=lr),
                  "adam":lambda model, lr: optim.Adam(model.parameters(), lr=lr),
                  "svrg":lambda model, lr: opt.SVRG(model, lr=lr),
                  "sgd":lambda model, lr: optim.SGD(model.parameters(), lr=lr),
                  "rprop":lambda model, lr: optim.Rprop(model.parameters(), lr=lr)}

weight_dict = {0:"weight", 1:"bias"}

class BaseModel(nn.Module):
    """INSPIRED BY KERAS AND SCIKIT-LEARN API"""
    def __init__(self, 
                 problem_type="classification", 
                 loss_name="categorical_crossentropy",
                 optimizer_name="adadelta"):

        super(BaseModel, self).__init__()
        self.loss_name = loss_name
        self.problem_type = problem_type
        self.my_optimizer = None
        self.optimizer_name = optimizer_name
        self.gpu_started = False

    def start_gpu(self):
        if not self.gpu_started:
            if torch.cuda.is_available():
                print ("pytorch running on GPU....")
                self.cuda()
            else:
                print ("pytorch running on CPU....")

            self.gpu_started = True

    def load_weights(self, weights=None):
        self.load_state_dict(torch.load(weights+".pth"))

    def save_weights(self, weights):
        torch.save(self.state_dict(), '%s.pth' % weights)
        print ("weights %s.pth saved..." % weights)
        #net.load_state_dict(torch.load('./net.pth'))

    def reset_optimizer(self, optimizer_name="adadelta",learning_rate=1.0):
        self.optimizer_name = optimizer_name
        self.my_optimizer = optimizer_dict[self.optimizer_name](self, learning_rate)


    def get_weights(self, layer=None, norm_only=False, verbose=0):
        weight_norms = {}
        for key_param in self._modules.keys():
            if layer is not None and layer != key_param:
                continue

            for i, param in enumerate(self._modules[key_param].parameters()):
                weight = utt.get_numpy(param)
               

                if not norm_only and verbose:
                    print ("weight:", weight[:5])
                weight_norm = np.linalg.norm(weight)

                if verbose:
                    print ("\nLAYER %s - WEIGHT %d" % (key_param, i+1))
                    print ("\nweight norm: %.3f" % (weight_norm))
                    print ("min: %.3f, mean: %.3f, max: %.3f" % (weight.min(), weight.mean(), weight.max()))

                    print ("shape: %s" % (str(weight.shape)))

                weight_norms["%s_%s norm" % (key_param, weight_dict[i])] = weight_norm

        return weight_norms


    def compute_loss(self, X, y, batch_size=1000):
        batch_size = min(batch_size, X.shape[0])
        data_loader = utt.get_data_loader(X, y, batch_size, 
                                          problem_type=self.problem_type)
        n = data_loader.sampler.num_samples

        # compute total loss
        total_loss = 0.
        for bi, (xb, yb) in enumerate(data_loader):
            if torch.cuda.is_available():
                    xb = xb.cuda()
                    yb = yb.cuda()

            xb, yb = Variable(xb), Variable(yb)

            y_pred = self(xb)
            loss = loss_dict[self.loss_name](y_pred, yb)

            total_loss += loss.data[0]

        avg_loss = total_loss / float(n)
        #print "Average loss: %.3f" % avg_loss

        return avg_loss

    def compute_accuracy(self, X, y, batch_size=1000):
        y_pred = self.predict(X, batch_size=batch_size)

        acc = (np.argmax(y_pred, axis=1) == np.argmax(y, axis=1)).sum() / float(y.shape[0])

        return acc

    def set_optimizer(self, optimizer_name="adadelta", learning_rate=1.0):
        # INITIALIZE OPTIMIZER
        self.optimizer_name = optimizer_name
        self.my_optimizer = optimizer_dict[self.optimizer_name](self, learning_rate)

    def get_optimizer(self):
        # INITIALIZE OPTIMIZER
        return self.optimizer_name

    def forward_pass(self, XList):
        self.start_gpu()
        self.train()

        if not isinstance(XList, list):
            XList = [XList]

        if isinstance(XList[0], np.ndarray):
            XList = utt.numpy2var(*XList)

        if not isinstance(XList, list):
            XList = [XList]   
        y_pred = self(*XList)

        return y_pred 

    def forward_backward(self, XList, y, loss_function=None, verbose=0):
        y_pred = self.forward_pass(XList)

        if isinstance(y, (np.ndarray, list)):
            y = utt.numpy2var(y)
    
        if self.my_optimizer is None:
            print("optimizer set...")
            self.set_optimizer()


        # ----- 2.OPTIMIZER
        self.my_optimizer.zero_grad()

        if loss_function is None:
            loss = loss_dict[self.loss_name](y_pred, *y)
        else:
            loss = loss_function(y_pred, *y)

        loss.backward()

        self.my_optimizer.step()

        loss_value = loss.data[0]
        if verbose:
            print("loss: %.3f" % loss_value)

        return loss_value

    def fit_batch(self, xb, yb, verbose=0):
        self.start_gpu()
        self.train()

        if self.my_optimizer is None:
            print("optimizer set...")
            self.set_optimizer()
        

        if not isinstance(xb, torch.FloatTensor):
            xb = torch.FloatTensor(xb)
            yb = utt.get_target_format(yb, problem_type=self.problem_type)

        # ----- 1.OPTIMIZER
        if torch.cuda.is_available():
                xb = xb.cuda()
                yb = yb.cuda()

        xb, yb = Variable(xb), Variable(yb)


        # ----- 2.OPTIMIZER
        self.my_optimizer.zero_grad()
        y_pred = self(xb)
        loss = loss_dict[self.loss_name](y_pred, yb)
        loss.backward()

        self.my_optimizer.step()

        loss_value = loss.data[0]
        if verbose:
            print("loss: %.3f" % loss_value)

        return loss_value

    def fit_on_dataloader(self, data_loader):
        self.start_gpu()
        self.train()

        n_batches = len(data_loader)
        for bi, (xb, yb) in enumerate(data_loader):
            loss = self.fit_batch(xb, yb, verbose=0)
            print("[%d/%d] - batch loss: %.3f" % (bi + 1, n_batches, loss))

    def fit(self, X, y, batch_size=20, epochs=1, 
            save_every=10, weights_name=None, 
            until_loss=None, verbose=1, reset_optimizer=False,
            optimizer_name=None, svrg_inner=3, learning_rate=1.0):

        self.start_gpu()

        # SET MODEL TO TRAIN MODE FOR DROPOUT AND BN
        self.train()


        # INITIALIZE DATA
        n_samples = X.shape[0] 
        batch_size = min(batch_size, n_samples)
        data_loader = utt.get_data_loader(X, y, batch_size, 
                                          problem_type=self.problem_type)

        # INITIALIZE OPTIMIZER
        if (optimizer_name is not None) and (optimizer_name != self.optimizer_name):
            self.optimizer_name = optimizer_name
            self.my_optimizer = optimizer_dict[self.optimizer_name](self, learning_rate)
     
        if (reset_optimizer or self.my_optimizer is None):
            self.my_optimizer = optimizer_dict[self.optimizer_name](self, learning_rate)

        if weights_name is not None:
            try:
                self.load_weights(weights=weights_name)

                if verbose:
                    print("SUCCESS: %s loaded..." % weights_name)
            except:
                print("FAILED: %s NOT loaded..." % weights_name)

        for epoch in range(epochs):
            if type(self.my_optimizer).__name__.lower() == "svrg" and (epoch%svrg_inner == 0):
                self.my_optimizer.update_model_outer(data_loader, loss_dict[self.loss_name])

            # INITILIZE VERBOSE
            losses = utt.AverageMeter()
            accs = utt.AverageMeter()
            # if verbose > 1:
            #     pbar = tqdm(total=n_samples)

            # INNER LOOP
            s = time.time()
            n_samples = data_loader.sampler.num_samples

            for bi, (xb, yb) in enumerate(data_loader):
                # ----- 1.GET DATA
                if torch.cuda.is_available():
                    xb = xb.cuda()
                    yb = yb.cuda()

                if self.loss_name == "binary_crossentropy":
                    xb, yb = Variable(xb), Variable(yb).float()

                else:
                    xb, yb = Variable(xb), Variable(yb)

                # ----- 2.OPTIMIZER
                self.my_optimizer.zero_grad()
                y_pred = self(xb)
                loss = loss_dict[self.loss_name](y_pred, yb)
                loss.backward()

                if type(self.my_optimizer).__name__.lower() == "svrg":
                    self.my_optimizer.step(xb, yb, loss_fn=loss_dict[self.loss_name], 
                                           n_samples=n_samples)
                else:
                    self.my_optimizer.step()

                # ----- 3.VERBOSE
                losses.update(loss.data[0], batch_size)
                stdout = ("%d/%d - [%d/%d] - Loss (%s): %f" % (epoch+1, 
                            epochs, (bi+1)*batch_size, 
                            n_samples, self.loss_name, losses.avg))

                if self.problem_type == "classification" and self.loss_name != "binary_crossentropy":
                    accs.update(utt.accuracy(y_pred, yb), 1)
                    stdout += " - Acc: %.3f" % accs.avg

                if verbose > 1:
                    # pbar.set_description(stdout)
                    # pbar.update(batch_size)
                    print(stdout)

            # ------ 4. OUTER LOOP
            stdout += " - Time: %.3f sec" % (time.time() - s)
            stdout += " - Optimizer: %s" % type(self.my_optimizer).__name__
            
            # stdout += " - Operations: %d" % count_operations(self.LayerList)
            if verbose == 1:
                print("")
                print(stdout)
            # else:
            #     pbar.set_description(stdout)      
            
            # ------ 5. SAVE WEIGHTS
            if ((epoch % save_every == 0) and (epoch!=0) and 
                (weights_name is not None)):

                self.save_weights(weights_name)

            if epoch == (epochs - 1) and (weights_name is not None):
                self.save_weights(weights_name)

            # ------ 6. EARLY STOP
            if until_loss is not None:
              if until_loss > losses.avg: 
                return 

    def fit_layers(self, X, y, batch_size=20, epochs=1, verbose=1,
                   layer_opt=((None, None),), svrg_inner=3):
        self.start_gpu()

        if np.array(layer_opt).ndim == 1:
            layer_opt = [layer_opt]

        # SET MODEL TO TRAIN MODE FOR DROPOUT AND BN
        self.train()

        # INITIALIZE DATA
        n_samples = X.shape[0] 
        batch_size = min(batch_size, n_samples)
        data_loader = utt.get_data_loader(X, y, batch_size, 
                                          problem_type=self.problem_type)


        # INITIALIZE OPTIMIZERS
        #optList = [optim.Adadelta(self.fc1.parameters())]
        optList = []
        for layer, opt in layer_opt:
            if opt == "svrg":
                optList += [opt.SVRG(self, layer=layer, lr=1e-5)]
            else:
                optList += [optim.Adadelta(self._modules[layer].parameters())]


        prev_norms = self.get_weights()
        for epoch in range(epochs):
            for opt in optList:

                if type(opt).__name__.lower() == "svrg" and (epoch%svrg_inner == 0):
                    opt.update_model_outer(data_loader, loss_dict[self.loss_name])

            # INITILIZE VERBOSE
            losses = utt.AverageMeter()
            accs = utt.AverageMeter()

            # INNER LOOP
            s = time.time()
            for bi, (xb, yb) in enumerate(data_loader):
                # ----- 1.GET DATA
                xb, yb = Variable(xb), Variable(yb)
                if torch.cuda.is_available():
                    xb = xb.cuda()
                    yb = yb.cuda()

                # ----- 2.OPTIMIZER
                for opt in optList:
                    opt.zero_grad()

                y_pred = self(xb)
                loss = loss_dict[self.loss_name](y_pred, yb)
                loss.backward()

                for opt in optList:
                    if type(opt).__name__.lower() == "svrg":
                        opt.step(xb, yb, loss_fn=loss_dict[self.loss_name])
                    else:
                        opt.step()

                # ----- 3.VERBOSE
 
                losses.update(loss.data[0], batch_size)
                stdout = "%d - Loss (%s): %f" % (epoch, self.loss_name, losses.avg)
                if self.problem_type == "classification":
                    accs.update(utt.accuracy(y_pred, yb), 1)
                    stdout += " - Acc: %.3f" % accs.avg

            # ------ 4. OUTER LOOP
            stdout += " - Time: %.3f sec" % (time.time() - s)
            stdout += " - Optimizer: %s" % self.optimizer_name

            if verbose == 1:
                print(stdout)
                curr_norms = self.get_weights()

                for key in curr_norms:
                    print("%s: %.3f -> %.3f" % (key, prev_norms[key], curr_norms[key]))

                prev_norms = curr_norms


    def get_gradients(self, X, y, batch_size=10, norm_only=True):
        self.start_gpu()

        data_loader = utt.get_data_loader(X, y, batch_size=10)

        for bi, (xb, yb) in enumerate(data_loader):
            print("Batch %d" % bi)

            if torch.cuda.is_available():
                xb = xb.cuda()
                yb = yb.cuda()    

            xb, yb = Variable(xb), Variable(yb)

        

            y_pred = self(xb)
            loss = loss_dict[self.loss_name](y_pred, yb)
            print("loss (%s): %.3f" % (self.loss_name, loss.data[0]))
          
            # Zero the gradients before running the backward pass.
            self.zero_grad()
            loss.backward()

            # Update the weights using gradient descent. Each parameter is a Variable, so
            # we can access its data and gradients like we did before.
            for key_param in self._modules.keys():

                for i, param in enumerate(self._modules[key_param].parameters()):                    
                    grad = utt.get_numpy(param.grad)
                    print("\nLAYER %s - WEIGHT %d" % (key_param, i + 1))

                    if not norm_only:
                        print("grad:", grad)
                    print("gradient norm: %.3f" % (np.linalg.norm(grad)))
                    print("min: %.3f, mean: %.3f, max: %.3f" % (grad.min(), grad.mean(), grad.max()))

                    print("shape: %s" % (str(grad.shape)))

                print("")
        self.zero_grad()

    ## OUTPUT FUNCTIONS
    def predict(self, X, batch_size=10):
        self.start_gpu()

        self.eval()
        bs = batch_size
        if X.ndim == 3 or X.ndim == 1:
            X = X[np.newaxis]
        
        if self.problem_type == "segmentation":
            y_pred = np.zeros((X.shape[0], self.n_outputs, 
                               X.shape[2], X.shape[3]))

        if self.problem_type == "regression":
            y_pred = np.zeros((X.shape[0], self.n_outputs))

        if self.problem_type == "classification":
            y_pred = np.zeros((X.shape[0], self.n_outputs))

        i = 0
        while True:
            s = i*bs
            e = min((i+1)*bs, X.shape[0])
            Xb = torch.FloatTensor(X[s:e])

            if torch.cuda.is_available():
                Xb = Xb.cuda() 

            Xb = Variable(Xb)

            y_pred[s:e] = utt.get_numpy(self(Xb))

            i += 1

            if e == X.shape[0]:
                break

        return y_pred



    def layer_output(self, X, layer, batch_size=10):
        self.eval()

        if X.ndim == 3 or X.ndim == 1:
            X = X[np.newaxis]

        i = 0
        bs = batch_size
        output = []

        while True:
            s = i*bs
            e = min((i+1)*bs, X.shape[0])

            Xb = Variable(torch.FloatTensor(X[s:e]))
            if torch.cuda.is_available():
                result = self._modules[layer](Xb.cuda()).cpu()
            else:
                result = self._modules[layer](Xb)

            output  += [result.data.numpy()]

            i += 1
            if e == X.shape[0]:
                break


        return np.vstack(output)

    


# -------------- BASE EVOLVE
def value2key(dictionary, value):
    for k in dictionary:
        if dictionary[k] == value:
            return k

class BaseEvolve(BaseModel):
    def __init__(self):
        super(BaseEvolve, self).__init__()

    def count_flops(self):
        flops = 0
        x = torch.zeros((1, self.n_channels, self.n_rows, self.n_cols))
        
        if torch.cuda.is_available():
            x = x.cuda()

        x = Variable(x)
        
        n_layers = len(self.LayerList)
        for i in range(n_layers):
            name = self.LayerList[i]
            if "conv" in name:
                layer = self._modules[name]
                x = F.relu(layer(x))

                C1 = int(layer.in_channels)
                C2 = int(layer.out_channels)
                K1, K2 = layer.kernel_size
                
                # image size
                size = x.size()
                H = size[-1]
                W = size[-2]
                flops += C1*C2*K1*K2*H*W

            if "pool" in name:
                layer = self._modules[name]
                x = layer(x)

                flops += 0


        x = x.view(x.size()[0], self.flatten_size)

        for i in range(n_layers):
            name = self.LayerList[i]
            if "fc" in name:
                layer = self._modules[name]
                x = F.relu(layer(x))

                flops += layer.in_features * layer.out_features

            if name == "out":
                layer = self._modules[name]
                x = layer(x)

                flops += layer.in_features * layer.out_features

        return flops

    def forward(self, x):
        n_layers = len(self.LayerList)
        for i in range(n_layers):
            name = self.LayerList[i]
            if "conv" in name:
                layer = self._modules[name]
                x = F.relu(layer(x))

            if "pool" in name:
                layer = self._modules[name]
                x = layer(x)


        x = x.view(x.size()[0], self.flatten_size)

        for i in range(n_layers):
            name = self.LayerList[i]
            if "fc" in name:
                layer = self._modules[name]
                x = F.relu(layer(x))

            if name == "out":
                layer = self._modules[name]
                x = layer(x)

            #print "layer: %s" % name

        return F.log_softmax(x)

    # ------ NET 2 NET OPERATIONS
    def add_convLayer(self, after_layer=None):
        if isinstance(after_layer, str):
            after_layer = value2key(self.LayerList, after_layer)

        n_layers = len(self.LayerList)

        for i in range(n_layers):
            name = self.LayerList[i]

            if "conv" in name:
                chosen_layer = i
                last_conv = name

        # 1.-------- ADD cONV LAYER USING N2N
        pos = last_conv

        layer = self._modules[pos]

        assert isinstance(layer, nn.Conv2d)

        n_filters = layer.out_channels
        kh, kw = layer.kernel_size
        new_layer = nn.Conv2d(n_filters, n_filters, 
                              layer.kernel_size, padding=1)

        # FILL WITH IDENTITY (HAS TO BE ODD)
        new_convWeight = torch.zeros(n_filters, 
                                              n_filters, 
                                              kh, 
                                              kw)

        std = layer.weight.std().data[0]

        
        for i in range(n_filters):
            # ADD NOISE
            new_convWeight[i, i, (kh-1)/2, (kw-1)/2] = 1. + np.random.randn()*std*5e-2
            

        new_layer.weight = torch.nn.Parameter(new_convWeight)
        new_layer.bias = torch.nn.Parameter(torch.zeros(n_filters))

        n_conv = np.max([int(st.replace("conv","")) for st in  self._modules.keys() if "conv" in st]) + 1
        name_new = "conv%d" % n_conv

        self.add_module(name_new, new_layer)


        # 2.---------- GET NAME
        if after_layer is None:
            after_layer = chosen_layer


        for i in range(n_layers, after_layer+1,-1):
            self.LayerList[i] = self.LayerList[i-1]
        
        self.LayerList[i-1] = name_new
        print(self.LayerList)

    def add_fcLayer(self, after_layer=None):

        if isinstance(after_layer, str):
            after_layer = value2key(self.LayerList, after_layer)

        n_layers = len(self.LayerList)

        if after_layer is None:
            for i in range(n_layers):
                name = self.LayerList[i]

                if "fc" in name:
                    after_layer = name
                    chosen_layer = i

        # ADD FC LAYER USING N2N
        pos = after_layer

        layer = self._modules[pos]
        w = layer.weight

        assert isinstance(layer, nn.Linear)

        n_params = w.size()[0]
        new_layer = nn.Linear(n_params, n_params)

        new_layer.weight = torch.nn.Parameter(torch.eye(n_params))
        new_layer.bias = torch.nn.Parameter(torch.zeros(n_params))
        
        n_fc = np.max([int(st[2:]) for st in  self._modules.keys() if "fc" in st]) + 1

        name_new = "fc%d" % n_fc
        self.add_module(name_new, new_layer)

        # GET NAME
        for i in range(n_layers, chosen_layer+1,-1):
            self.LayerList[i] = self.LayerList[i-1]
        
        self.LayerList[i-1] = name_new
        print(self.LayerList)

    def expand_layer(self, layer, n_nodes, add_noise=False):
        if isinstance(layer, str):
            layer = value2key(self.LayerList, layer)

        pos1 = self.LayerList[layer]  

        if "conv" in pos1:
            flag = False
            for pos2 in range(layer+1, len(self.LayerList)):
                if isinstance(self._modules[self.LayerList[pos2]],
                              nn.Conv2d):
                    flag = True
                    break



        if "fc" in pos1:
            flag = False
            for pos2 in range(layer+1, len(self.LayerList)):
                if isinstance(self._modules[self.LayerList[pos2]],
                              nn.Linear):
                    flag = True
                    break
                    
        assert flag

              
        pos2 = self.LayerList[layer+1]
        newWidth = n_nodes
        layers = self._modules
        
        l1 = layers[pos1]
        l2 = layers[pos2]

        w1 = l1.weight
        b1 = l1.bias

        w2 = l2.weight

        # SANITY CHECKS
        assert isinstance(l1, nn.Linear) or isinstance(l1, nn.Conv2d)         
        assert(w2.size()[1] == w1.size()[0])
        assert(newWidth >= w1.size()[0])

        oldWidth = w2.size()[1]

        # NEW WEIGHTS
        if isinstance(l1, nn.Conv2d):
            nw1 = torch.zeros(newWidth, w1.size()[1], w1.size()[2], w1.size()[3])
            nw2 = torch.zeros(w2.size()[0], newWidth, w2.size()[2], w2.size()[3])
        else:
            nw1 = torch.zeros(newWidth, w1.size()[1])
            nw2 = torch.zeros(w2.size()[0], newWidth)

        # NEW BIAS
        nb1 = torch.zeros(newWidth)

        # COPY THE ORIGINAL WEIGHTS
        nw1[:oldWidth] = w1.data
        nw2[:, :oldWidth] = w2.data

        nb1[:oldWidth] = b1.data

        # RANDOM WEIGHT SELECTION
        count = {i:1. for i in range(oldWidth)}
        picks = []

        for node in range(oldWidth, newWidth):
            i = np.random.randint(oldWidth)
            count[i] += 1.
            picks += [i]

            # NEW WEIGHT
            nw1[node] = w1[i].data

            nw2[:,node] = w2[:, i].data 

            # NEW BIAS
            nb1[node:node+1] = b1[i].data

        # NORMALIZE THE WEIGHTS
        for i in range(oldWidth):
            if count[i] != 1.:
                nw2[:, i] /= count[i]

        for i, node in enumerate(range(oldWidth, newWidth)):
            nw2[:, node] /= count[picks[i]]

        if add_noise:

            w2_sub = nw2[:, oldWidth:]
            noise = torch.randn(w2_sub.size()) * w2_sub.std() * 5e-2
            
            nw2[:,oldWidth:] += noise

        if isinstance(l1, nn.Linear):
            l1.out_features = newWidth
            l2.in_features = newWidth

        if isinstance(l1, nn.Conv2d):
            l1.out_channels = newWidth
            l2.in_channels = newWidth
        
        l1.weight = torch.nn.Parameter(nw1)
        l2.weight = torch.nn.Parameter(nw2)

        l1.bias = torch.nn.Parameter(nb1)

        # ZERO GRADIENTS
        l1.zero_grad()
        l2.zero_grad()
