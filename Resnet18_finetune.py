import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data
import os, time
from PIL import Image
from torchvision import transforms
import torchvision.models as models


# Hyper Parameter
BATCH_SIZE = 64
IMAGE_CLASS = 200
EPOCH = 20
LR = 0.01

# Load Pic
def default_loader(path):
    return Image.open(path).convert('RGB').resize((224, 224))

class MyDataset(Data.Dataset):
    def __init__(self, root, label, transform=None, target_transform=None, loader=default_loader):
        imgs = []
        label_txt = open(label, 'rb')
        for line in label_txt.readlines():
            pic_name, kind = line.strip().split()
            # print pic_name, ' ',kind
            pic_name = pic_name.decode('ascii')
            kind = kind.decode('ascii')
            # print (os.path.join(root, pic_name))
            if os.path.isfile(os.path.join(root, pic_name)):
                imgs.append((pic_name, int(kind)))
        label_txt.close()
        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        pic_name, kind = self.imgs[index]
        img = self.loader(os.path.join(self.root, pic_name))
        if self.transform is not None:
            img = self.transform(img)
        return img, kind-1

    def __len__(self):
        return len(self.imgs)

train_txt_path = '/media/tang/code/data/cub/txt/image_class_labels_trainset_jpg.txt'
test_txt_path = '/media/tang/code/data/cub/txt/image_class_labels_testset_jpg.txt'
train_pic_fold = '/media/tang/code/data/cub/img/img_train'
test_pic_fold = '/media/tang/code/data/cub/img/img_test'

train_data = MyDataset(train_pic_fold, train_txt_path, transforms.ToTensor())
test_data = MyDataset(test_pic_fold, test_txt_path, transforms.ToTensor())
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE)


# Model
# alex = models.alexnet(pretrained = True)
model = models.resnet18(pretrained = True)
# inception_v3 = models.inception_v3(pretrained=True)
# densenet121  = models.densenet121(pretrained=True)

# print model
for i, param in enumerate(model.parameters()):
    param.requires_grad = False
    # print i, param.requires_grad

# print model
dim_in = model.fc.in_features
model.fc = nn.Linear(dim_in, IMAGE_CLASS)
model.cuda()
#net = torch.nn.DataParallel(model, device_ids=[0, 1])
#print (model)

# Optimize only the classifier
optimizer = torch.optim.SGD(model.fc.parameters(), lr=LR, momentum=0.9)
loss_func = torch.nn.CrossEntropyLoss()
print (train_data.__len__())


for epoch in range(EPOCH):
#train
    model.train()
    time_s = time.time()
    acc_count = 0
    acc_sum = 0
    for step, (x, y) in enumerate(train_loader):
        x, y = Variable(x).cuda(), Variable(y).cuda()
        out = model(x)
        loss = loss_func(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predict_y = torch.max(F.softmax(out), 1)[1].cuda().data.squeeze()
        target_y = y.data
        acc_sum += int(x.size(0))
        acc_count += sum(predict_y==target_y)
        if step % 10 == 0:
            print(type(acc_count), type(acc_sum))
            print(acc_count/acc_sum)
            accuarcy = 1.0 * acc_count/ acc_sum
            loss = loss.cpu().data.numpy()
            print ('Epoch:%2s Step:%3s | acc_count:%5s   acc_sum:%5s | Accuarcy:%.3f  Loss:%.7f' % (epoch, step, acc_count, acc_sum, accuarcy, loss))
    accuarcy = 1.0 * acc_count / acc_sum
    print ('Epoch:%2s info---- | acc_count:%5s   acc_sum:%5s | Accuarcy:%.3f' % (epoch, acc_count, acc_sum, accuarcy))
    time_e = time.time()
    time_spend = time_e-time_s
    print ('trian time:%.2f mins/epoch  %.2f pics/s' % (1.0*time_spend/60, 1.0*acc_sum/time_spend))

# Test testset
    model.eval()
    time_s = time.time()
    acc_sum = 0
    acc_count = 0
    loss_sum = 0
    for step, (x, y) in enumerate(test_loader):
        # x.ivolatile = True
        x, y = Variable(x).cuda(), Variable(y).cuda()
        out = model(x)
        loss = loss_func(out, y)
        predict_y = torch.max(F.softmax(out), 1)[1].cuda().data.squeeze()
        target_y = y.data
        acc_sum += x.size(0)
        acc_count += sum(predict_y==target_y)
        loss = loss.cpu().data.numpy()
        loss_sum += loss*x.size(0)
    accuarcy = 1.0 * acc_count/ acc_sum
    loss = 1.0 * loss_sum / acc_sum
    time_e = time.time()
    time_spend = time_e-time_s
    print ('test time:%.2f mins   %.2f pics/s' % (1.0*time_spend/60, 1.0*acc_sum/time_spend))
    print ('Test result: acc_count:%s   acc_sum:%s | Accuarcy:%.3f  Loss:%.7f' % (acc_count, acc_sum, accuarcy, loss))
    print (' ')

    torch.save(model, './home/tang/linux/model/classification/cub/cub_resnet18_fc.pkl')
