from models import vgg
import torch
from load_datasets import CifarDataset
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

class Trainer:
    def __init__(self, data_path):
        '''train setting'''
        lr = 1e-3
        self.epochs = 1000
        self.show_iter = 1
        batch_size = 64
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = vgg.VGG16().cuda() if self.device=="cuda" else vgg.VGG16()
        if self.device=="cuda":
            self.criterion = torch.nn.CrossEntropyLoss().cuda()
        else:
            self.criterion = torch.nn.CrossEntropyLoss()
        self.optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        train_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor()
        ])

        '''dataset'''
        train_dataset = CifarDataset(data_path+"/train", train_transform)
        test_dataset = CifarDataset(data_path+"/test", test_transform)
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size)

    def train(self):
        '''training'''
        for epoch in range(self.epochs):
            train_loss = 0
            correct = 0
            self.model.train()
            for x, y in self.train_loader:
                if self.device == "cuda":
                    x, y = x.cuda(), y.cuda()
                pred = self.model(x)
                loss = self.criterion(pred, y)
                train_loss += loss * x.size()[0]

                '''update'''
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                correct += torch.sum(torch.eq(torch.argmax(pred, dim=1), y))

            '''validation'''
            self.model.eval()
            with torch.no_grad():
                val_loss = 0
                val_correct = 0
                for val_x, val_y in self.test_loader:
                    if self.device == "cuda":
                        val_x, val_y = val_x.cuda(), val_y.cuda()
                    val_pred = self.model(val_x)
                    loss = self.criterion(val_pred, val_y)
                    val_loss += loss * val_x.size()[0]
                    val_correct += torch.sum(torch.eq(torch.argmax(val_pred, dim=1), val_y))
            if epoch % self.show_iter == 0:
                epoch_loss = train_loss.item() / len(self.train_loader.dataset)
                epoch_acc = correct.item() / len(self.train_loader.dataset)
                epoch_val_loss = val_loss.item() / len(self.test_loader.dataset)
                epoch_val_acc = val_correct.item() / len(self.test_loader.dataset)
                print("epoch: %d, loss: %.3f, accuracy: %.3f, val_loss: %.3f, val_accuracy: %.3f"
                      %(epoch, epoch_loss, epoch_acc, epoch_val_loss, epoch_val_acc))


if __name__ == "__main__":
    trainer = Trainer("./datasets/cifar100/")
    trainer.train()

