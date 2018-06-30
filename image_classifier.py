import torch
import utils

from collections import OrderedDict
from torch import nn, optim
from torchvision import models

vgg16 = models.vgg16(pretrained=True)
densenet121 = models.densenet121(pretrained=True)

model_arr    = ['vgg', 'densenet']
model_types  = [vgg16, densenet121]
model_inputs = [25088, 1024]

class model():
    data_path       = 'flowers'
    checkpoint_path = ''
    predict_path    = 'flowers/test/1/image_06743.jpg'
    model           = vgg16
    model_input     = 25088
    model_hidden    = 1024
    model_output    = 102
    drop_p          = 0.5
    norm_means      = (0.485,0.456,0.406)
    norm_stdv       = (0.229,0.224,0.225)
    re_size         = 224
    train_batch     = 64
    valid_batch     = 32
    train_loader    = None
    valid_loader    = None
    learn_rate      = 0.0001
    optimizer       = None
    device          = 'cpu'
    criterion       = nn.NLLLoss()
    epochs          = 3
    print_every     = 40
    running_loss    = 0
    steps           = 0
    top_k           = 5
    category_names  = 'cat_to_name.json'
    
    def __init__(self, d_path, c_path, arch, lr, hidden, epochs, gpu, training, predict, top, cat_names):
        if training:
            self.data_path          = d_path
            self.checkpoint_path    = c_path
            self.model              = model_types[model_arr.index(arch)]
            self.model_input        = model_inputs[model_arr.index(arch)]
            self.model_hidden       = hidden
            self.learn_rate         = lr
            self.epochs             = epochs
            self.device             = torch.device('cuda:0' if gpu else 'cpu')
            self.model.to(self.device);

            for p in self.model.parameters():
                p.requires_grad = False

            classifier = nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(self.model_input, self.model_hidden)),
                ('rl1', nn.ReLU(inplace=True)),
                ('dr1', nn.Dropout(p=self.drop_p)),
                ('fc2', nn.Linear(self.model_hidden, self.model_output)),
                ('out', nn.LogSoftmax(dim=1))
            ]))

            self.model.classifier = classifier
            self.optimizer        = optim.Adam(self.model.classifier.parameters(), lr = self.learn_rate)

            self.create_loaders()

            train_dataset           = utils.create_dataset(self.data_path+'/train', 
                                                           self.re_size, 
                                                           self.norm_means, 
                                                           self.norm_stdv, 
                                                           self.train_batch)
            self.model.class_to_idx = train_dataset.class_to_idx
            
        else:
            self.model           = model_types[model_arr.index(arch)]
            
            
            classifier = nn.Sequential(OrderedDict([
                    ('fc1', nn.Linear(self.model_input, self.model_hidden)),
                    ('rl1', nn.ReLU(inplace=True)),
                    ('dr1', nn.Dropout(p=self.drop_p)),
                    ('fc2', nn.Linear(self.model_hidden, self.model_output)),
                    ('out', nn.LogSoftmax(dim=1))
                ]))

            self.model.classifier = classifier
            self.optimizer        = optim.Adam(self.model.classifier.parameters(), lr = self.learn_rate)
            
            self.checkpoint_path = c_path
            self.predict_path    = predict
            self.top_k           = top
            self.category_names  = cat_names
            self.load_checkpoint()
        
    
    def create_loaders(self):
        self.train_loader = utils.create_loader(self.data_path+'/train', 
                                                self.re_size, 
                                                self.norm_means, 
                                                self.norm_stdv, 
                                                self.train_batch)
        self.valid_loader = utils.create_loader(self.data_path+'/valid', 
                                                self.re_size, 
                                                self.norm_means, 
                                                self.norm_stdv, 
                                                self.valid_batch)
        
    def validate(self):
        acc      = 0
        val_loss = 0
        
        self.model.to(self.device)
        
        for images,labels in self.valid_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            output         = self.model.forward(images)
            val_loss      += self.criterion(output, labels).item()
            ps             = torch.exp(output)
            equality       = (labels.data == ps.max(dim=1)[1])
            acc           += equality.type(torch.FloatTensor).mean()
   
        return acc, val_loss

    def train(self):
        self.model.to(self.device)
        for e in range(self.epochs):
            self.model.train()
            for image, label in self.train_loader:
                self.steps       += 1
                image, label = image.to(self.device), label.to(self.device)
                self.optimizer.zero_grad()
                output = self.model.forward(image)
                loss   = self.criterion(output, label)
                loss.backward()
                self.optimizer.step()
                self.running_loss += loss.item()

                if self.steps%self.print_every == 0:
                    self.model.eval()

                    with torch.no_grad():
                        vali_acc, vali_loss = self.validate()

                    print('Epoch: {}/{}'.format(e + 1,self.epochs),
                          'Training loss: {:.4f}'.format(self.running_loss  / self.print_every),
                          'Validation accuracy {:.4f}'.format(vali_acc / len(self.valid_loader)),
                          'Validation loss: {:.4f}'.format(vali_loss   / len(self.valid_loader)))

                    self.running_loss=0
                    self.model.train()
                
    def save_checkpoint(self):
        checkpoint = {
            'input'                 : self.model_input,
            'hidden'                : self.model_hidden,
            'output'                : self.model_output,
            'epochs'                : self.epochs,
            'model_state_dict'      : self.model.state_dict(),
            'optimizer_state_dict'  : self.optimizer.state_dict(),
            'class_to_idx'          : self.model.class_to_idx
        }

        torch.save(checkpoint, self.checkpoint_path + 'checkpoint.pth')
        print("Checkpoint has been saved at location: " + self.checkpoint_path + self.checkpoint_path)
        
    def load_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_path)
        
        self.model_input            = checkpoint['input']
        self.model_hidden           = checkpoint['hidden']
        self.model_output           = checkpoint['output']
        self.epochs                 = checkpoint['epochs']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model.class_to_idx     = checkpoint['class_to_idx']
            
    def predict(self):
        names = utils.cat_to_name(self.category_names)
        processed_image = utils.process_image(self.predict_path, self.re_size, self.norm_means, self.norm_stdv)
        self.model.to(self.device)
        self.model.eval()
        processed_image = processed_image.float()
        processed_image = processed_image.unsqueeze(0)
        processed_image = processed_image.to(self.device)
        output = self.model.forward(processed_image)
        ps = torch.exp(output)
        probs, index = ps.topk(self.top_k)
        index = index.tolist()[0]
        inv_dict = {v:k for k,v in self.model.class_to_idx.items()}
        
        classes = []
        for i in index:
            classes.append(inv_dict[i])
        probs = probs.tolist()[0]
        
        top_names = []
        for c in classes:
            top_names.append(names[c].title())
        
        probs_per = []
        for p in probs:
            probs_per.append(p*100)
        
        print('resulting [{}] predictions:'.format(self.top_k))
        for name in top_names:
            print(name + '....' + ' {0:.2f} percent'.format(probs_per[top_names.index(name)]))