import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import torch
from skimage import io
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Sampler,TensorDataset
from torch.utils.data.sampler import WeightedRandomSampler
import torchvision
from matplotlib import pyplot as PLT
from torchvision.io import read_image

from torchvision import transforms
import torchvision.models as models
from utils import CropedDataset
from torch.utils.tensorboard import SummaryWriter

#LOAD DATASET
batch_size = 5
path_base = r"/home/fm/Documents/GitHub/em_desenvolvimento/GIS_c/dataset"#_jpeg"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#Pesos do modelo ja treinado
weights_path = r"/home/fm/Documents/GitHub/em_desenvolvimento/GIS_c/VGG16-ROCAN-btachsize40-epochs7-transformn0-acc_test0.7976190476190477.torch"

#Definição de classe do modelo
n_features = 2


epochs = 7

transform = transforms.Compose([  
    transforms.Resize((256, 256))])#,
    #transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])




face_dataset = CropedDataset(root_dir=path_base, transform=transform)
samples = face_dataset.__len__()

n_classes =     face_dataset.n_classes
targets = face_dataset.target()
class_sample_count = targets.value_counts().values

weights = 1 / torch.Tensor(class_sample_count)
weights = weights.double()
sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, batch_size)


train_set, test_set = torch.utils.data.random_split(dataset=face_dataset, lengths = [samples-int(samples*0.4),int(samples*0.4) ])
train_loader = DataLoader(dataset = train_set, batch_size = batch_size, sampler = sampler, num_workers=6)

test_loader = DataLoader(dataset = test_set, batch_size = batch_size, shuffle = False,num_workers=6)



#LOAD MODEL
model = models.vgg16(pretrained=True)
for param in model.parameters():    
    param.requires_grad = False

head_convs = model.classifier[0].in_features
model.classifier = nn.Linear(head_convs,n_features)

#Carregamento do modelo
model.load_state_dict(torch.load(weights_path))


model.double()
#passando modelo para GPU

log_dir = "runs/" + datetime.now().strftime('%Y%m%d-%H%M%S')
tb = SummaryWriter(log_dir)
images,labels = next(iter(test_loader))
image = images[4,:,:,:]

#Read direct file as numpy
path_image = r"/home/fm/Documents/GitHub/em_desenvolvimento/GIS_c/dataset_jpeg/0.0/croppid_936-class_0.0-shp_base_treino_CAN_ROv4-raster_7003-74624.jpeg"
image = io.imread(path_image)
plt.imshow(image)

#Read to tensor then to numpy
img_tensor = read_image(path_image,mode=torchvision.io.image.ImageReadMode.RGB).double()
plt.imshow(img_tensor.type(torch.int8).permute(1, 2, 0))

#Read to numpy then to tensor then to numpy

img_np = io.imread(path_image)
image = img_np.astype(float)
np.moveaxis(image,-1,0).shape
image = torch.tensor(image)
plt.imshow(img_tensor.type(torch.int8).permute(1, 2, 0))

grid = torchvision.utils.make_grid(image)
grid.shape

tb.add_image('image',grid,0)
tb.add_graph(model,images)


hparams_dict = {'epochs':epochs, 'batch_size':batch_size}
class_counters = {'counter_class_'+ str(i):str(class_sample_count[i]) for i in range(len(class_sample_count))}
transforms_used = {"transformer_"+str(i).split('(')[0]: ('('.join(str(i).split('(')[1:]))[:-1] for i in transform.__dict__['transforms']}

hparams_dict.update(class_counters)
hparams_dict.update(transforms_used)


model.to(device)
#Test accuracy

n_correct = 0
n_total= 0

for inputs,targets in train_loader:

    
    inputs,targets = inputs.to(device),targets.to(device)

    outputs = model(inputs)

    _,predictions = torch.max(outputs,1)

    #update max
    n_correct += (predictions ==targets).sum().item()
    n_total += targets.shape[0]
    
    #tb.add_scalar('Accuracy/train', n_correct / n_total,global_step= n_iter)
    

train_acc = n_correct / n_total
dict_metrics = {'Accuracy/train': train_acc}
tb.add_hparams(hparams_dict,metric_dict = dict_metrics)

tb.flush()
tb.close()


#test_acc
n_correct = 0
n_total = 0
n_iter = 0
for inputs,targets in test_loader:

    inputs,targets = inputs.to(device),targets.to(device)

    outputs = model(inputs)

    _,predictions = torch.max(outputs,1)

    #update max
    n_correct += (predictions ==targets).sum().item()
    n_total += targets.shape[0]
    
    writer.add_scalar('Accuracy/train', n_correct / n_total, n_iter)
    n_iter +=1
tb.close()

test_acc = n_correct / n_total

model_file_name = f'VGG16-ROCAN-btachsize{batch_size}-epochs{epochs}-transformn0-acc_test{test_acc}.torch'
torch.save(model.state_dict(),model_file_name)