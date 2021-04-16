import torch
from larecnet import LaRecNet
from resnet import BasicBlock
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from datasets import *
from matplotlib import pyplot
from torch.utils.data import DataLoader


DRIVE_PATH = "/gpfsnyu/scratch/yw3752/"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name(0))
print(torch.cuda.is_available())
torch.cuda.empty_cache()


DATASET = "wireframe"
DATASET_PATH = DRIVE_PATH + "Datasets/wireframe/"
GRAPH_PATH = DRIVE_PATH + "Graphs/"
MODEL_PATH = DRIVE_PATH + "Models/"
EPOCHS = 100
LR = 0.00001
BATCH_SIZE = 16
TRAIN_LOSSES = []
TEST_LOSSES = []
USE_APEX = False

'''
try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
    USE_APEX = True
    APEX_OPT_LEVEL = 'O1'
    print("NVIDIA Apex Support: True")
except ImportError:
    print("NVIDIA Apex Support: False")
'''


# %%
def r_f(angle, k, num_params=5):
    result = 0
    for i in range(num_params):
        result += k[i] * angle ** (2 * (i + 1) - 1)

    return result


# %%
class LaRecNetLoss(nn.Module):
    def __init__(self, weights=[], lambda_fus=2, lambda_global=1, lambda_local=1,
                 lambda_m=2, lambda_geo=100, lambda_pix=1, lambda_para=1):
        super(LaRecNetLoss, self).__init__()
        # params for MCM
        self.weights = np.ones(9)
        self.lambda_fus = lambda_fus
        self.lambda_global = lambda_global
        self.lambda_local = lambda_local

        # params for overall network
        self.lambda_m = lambda_m
        self.lambda_geo = lambda_geo
        self.lambda_pix = lambda_pix
        self.lambda_para = lambda_para

        # fix focal length: True ? False
        self.fix_focal = False
        self.focal = 0

    def fix_focal_length(self, f):
        self.fix_focal = True
        self.focal = f

    def forward(self, x, gt):
        k_local, k_global, k_hat = x[0], x[1], x[2]
        loss_global = 1 / 9 * (np.dot(self.weights, (k_global, gt["distortion"]))) ** 2
        loss_local = 1 / 25 * (self.weights[0:5] * np.dot(k_local, gt["distortion"][0:5])) ** 2
        loss_fused = 1 / 9 * (np.dot(self.weights, (k_hat, gt["distortion"]))) ** 2

        # loss_para = self.lambda_fus * loss_fused + self.lambda_global * loss_global + self.lambda_local * loss_local
        #
        # geometric_err = 0
        # image_size = len(gt["img"][0])

        return loss_fused + loss_local + loss_global


# %%
def train(model, inputs, ground_truth):
    # loss_func = LaRecNetLoss(weights)
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    total_loss = 0
    inputs = torch.reshape(inputs, (BATCH_SIZE, 3, 320, 320))
    prediction = model(inputs)
    # print("Model Output:", prediction)
    loss = loss_func(prediction, ground_truth)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    total_loss += loss.item()

    return total_loss


# %%
def save_plot(train_loss, test_loss, e):
    pyplot.plot(train_loss, label='train')
    pyplot.plot(test_loss, label='test')
    pyplot.legend()
    # save to directory
    pyplot.savefig(GRAPH_PATH + "EPOCH%d" % e)
    # pyplot.show()
    pyplot.clf()
    print("\n%%%%%%%%%%%%%%%%%%%%%")
    print("EPOCH %d graph saved!" % e)
    print("%%%%%%%%%%%%%%%%%%%%%\n")
    return


# %%
def main():
    if DATASET == "wireframe":
        wireframe_train = Wireframe(DATASET_PATH + "v1.1/train_4320.txt", DRIVE_PATH + "Datasets/")
        wireframe_test = Wireframe(DATASET_PATH + "v1.1/test_1080.txt", DRIVE_PATH + "Datasets/")
        dataset_loader = DataLoader(dataset=wireframe_train, batch_size=BATCH_SIZE, pin_memory=True, drop_last=True,
                                    shuffle=True)
        test_loader = DataLoader(dataset=wireframe_test, batch_size=BATCH_SIZE, pin_memory=True, drop_last=True,
                                 shuffle=True)
    else:
        dataset_loader = None
        test_loader = None

    model = LaRecNet(block=BasicBlock, layers=[2, 2, 2, 2], batch_size=BATCH_SIZE)
    model.to(device)
    # loss_func = LaRecNetLoss(weights)
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    if USE_APEX:
        model, optimizer = amp.initialize(model, optimizer, opt_level=APEX_OPT_LEVEL)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [30])
    for i in range(EPOCHS):
        print("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print("EPOCH %d STARTED" % (i + 1))
        epoch_loss_train = []
        epoch_loss_test = []
        for idx, data in enumerate(dataset_loader):
            inputs, ground_truth = data
            inputs = torch.reshape(inputs, (BATCH_SIZE, 3, 320, 320))
            inputs = inputs.to(device)
            ground_truth = ground_truth.to(device)
            prediction = model(inputs)
            # print("F_GroundTruth:", torch.reshape(ground_truth ,(1, BATCH_SIZE)))
            loss = loss_func(prediction, ground_truth)
            epoch_loss_train.append(loss)

            optimizer.zero_grad()
            if USE_APEX:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()
            # scheduler.step()

        avg_loss = (sum(epoch_loss_train) / len(epoch_loss_train)).cpu().detach().float()
        TRAIN_LOSSES.append(avg_loss)
        print("##### EPOCH %d train finished." % (i + 1))
        print("##### Average train loss:", avg_loss)

        print("##### Start testing...")
        with torch.no_grad():
            for idx, data in enumerate(test_loader):
                input, ground_truch = data
                inputs = torch.reshape(inputs, (BATCH_SIZE, 3, 320, 320))
                inputs = inputs.to(device)
                ground_truth = ground_truth.to(device)
                prediction = model(inputs)
                loss = loss_func(prediction, ground_truth)
                epoch_loss_test.append(loss)
            avg_loss = (sum(epoch_loss_test) / len(epoch_loss_test)).cpu().float()
            TEST_LOSSES.append(avg_loss)
        print("##### EPOCH %d test finished." % (i + 1))
        print("##### Average test loss:", avg_loss)
        save_plot(TRAIN_LOSSES, TEST_LOSSES, i + 1)

        if (i + 1) % 10 == 0:
            torch.save(model, MODEL_PATH + "NEW_EPOCH%d.pkl" % (i + 1))

        print("EPOCH %d FINISH" % (i + 1))
        print("\n-----------------------------")
    return model


model = main()
torch.save(model, MODEL_PATH + "FINAL.pkl")
