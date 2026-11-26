"""Measure ResNet-18 training throughput on this CPU (synthetic data)."""
import sys, time, torch, torchvision
torch.set_num_threads(int(sys.argv[1]))
res, bs = int(sys.argv[2]), 64
m = torchvision.models.resnet18(num_classes=60).to(memory_format=torch.channels_last)
opt = torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.9)
x = torch.randn(bs, 3, res, res).to(memory_format=torch.channels_last)
y = torch.randint(0, 60, (bs,))
lossf = torch.nn.CrossEntropyLoss()
def step():
    opt.zero_grad(); lossf(m(x), y).backward(); opt.step()
for _ in range(3): step()
n, t = 8, time.time()
for _ in range(n): step()
dt = time.time() - t
print(f"threads {sys.argv[1]} res {res}: {n*bs/dt:.1f} train images/s")
