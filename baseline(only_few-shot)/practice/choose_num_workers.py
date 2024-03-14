from time import time
import multiprocessing as mp
import torch
import torchvision
from torchvision import transforms

transform = transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))
])

trainset = torchvision.datasets.MNIST(
    root='dataset/',
    train=True,  # 如果为True，从 training.pt 创建数据，否则从 test.pt 创建数据。
    download=True,  # 如果为true，则从 Internet 下载数据集并将其放在根目录中。 如果已下载数据集，则不会再次下载。
    transform=transform
)

print(f"num of CPU: {mp.cpu_count()}")
for num_workers in range(2, mp.cpu_count(), 2):
    train_loader = torch.utils.data.DataLoader(trainset, shuffle=True, num_workers=num_workers, batch_size=64,
                                               pin_memory=True)
    start = time()
    for epoch in range(1, 3):
        for i, data in enumerate(train_loader, 0):
            pass
    end = time()
    print("Finish with:{} second, num_workers={}".format(end - start, num_workers))
''''
num_workers = 0, train time per DataSet(s): 1168.39468
输出：
num of CPU: 20 
Finish with:12.027751445770264 second, num_workers=2
Finish with:4.964622259140015 second, num_workers=4  实际也慢
Finish with:3.711052417755127 second, num_workers=6
Finish with:3.0533156394958496 second, num_workers=8 
Finish with:2.745054006576538 second, num_workers=10  train time per DataSet(s): 9243.06270
Finish with:2.766202211380005 second, num_workers=12
Finish with:2.8320796489715576 second, num_workers=14
Finish with:2.92264461517334 second, num_workers=16
Finish with:2.94303822517395 second, num_workers=18

'''