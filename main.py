from teeth_final import  *
from resnet_gnn import  *
import torch
from torch.utils.tensorboard import SummaryWriter
#name of experience
writer = SummaryWriter('runs/dataset_479_2conv')
# %load_ext tensorboard
# tensorboard  --logdir=runs

dataset_path = "/home/ahmed/workspace/data/upper_occlusal_188/raw"
dataset_path = "/home/ahmed/workspace/data/occlusal_upper/pre_processed/raw"
# dataset_path = "/home/ahmed/workspace/data/occlusal_data/raw"

dataset = teeth_final(dataset_path, transform=T.Compose([T.KNNGraph(k=3), T.Cartesian()]))

torch.manual_seed(12345)
dataset = dataset.shuffle()
train_dataset = dataset[:400]
test_dataset = dataset[400:]

print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')


train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Resnet_GNN(num_classes=dataset.num_classes, device=device).to(device)  # , dataset.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()
train_loss = []
test_loss = []


def train():
    model.train()

    losses = 0
    for data in train_loader:
        data = data.to(device=device)

        out = model(data.x, data.edge_index, data.edge_attr)

        loss = criterion(out, data.y)
        losses += loss


        loss.backward()  # Drive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
    mean_loss = losses/len(train_loader)
    print("train loss : ", float(mean_loss))
    return float(mean_loss)




def test(loader):
    model.eval()
    correct = 0
    num_data = 0

    for data in loader:
        data = data.to(device=device)
        out = model(data.x, data.edge_index, data.edge_attr)
        # print(t)

        # print(out.size())

        pred = out.argmax(dim=1)
        # print("*******prediction*******: ")
        # print(pred)

        # print("*****ground_truth*******: ")
        # print(data.y)

        correct += int((pred == data.y).sum())
        num_data+= data.y.size()[0]

    # print("----------------------------------------------------------------")
    # print("correct results:", correct, "sur", num_data, "echattillon")
    # print("----------------------------------------------------------------")

    return correct / num_data  # Derive ratio of correct predictions.


for epoch in range(1, 300):
    train_loss = train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    writer.add_scalar("loss", train_loss, epoch)
    writer.add_scalar("train_acc", train_acc, epoch)
    writer.add_scalar("test_acc", test_acc, epoch)
    print(f'Epoch: {epoch:01d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_loss,
    }, 'output/model_{}.pth'.format(epoch))