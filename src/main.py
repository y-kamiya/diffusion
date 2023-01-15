import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F
from logzero import setup_logger
from torch import Tensor, nn, optim, utils
from torch.utils import tensorboard
from torchvision import datasets, transforms


class ToyDataset(utils.data.Dataset):
    def __init__(self, dim=4, num_max=8):
        self.dim = dim
        self.num_max = num_max

        data = []
        for i in range(num_max - dim + 1):
            data.append(list(range(i, i + dim)))
            data.append(list(range(i + dim - 1, i - 1, -1)))

        self.data = self.normalize(torch.Tensor(data)).repeat(1000, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def normalize(self, x):
        return 2 * x / (self.num_max - 1) - 1

    def denormalize(self, x):
        return (x + 1) * (self.num_max - 1) / 2


class Model(nn.Module):
    def __init__(self, dim, T):
        super(Model, self).__init__()
        self.dim = dim
        self.T = T
        self.fc1 = nn.Linear(dim + 1, 400)
        self.fc2 = nn.Linear(400, 400)
        self.fc3 = nn.Linear(400, 400)
        self.fc4 = nn.Linear(400, dim)

    def forward(self, x, t):
        x = torch.cat((x, t.unsqueeze(-1) / self.T), dim=1)
        x = F.relu(self.fc1(x.view(-1, self.dim + 1)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class Trainer(object):
    def __init__(self, config, logger, input_dim):
        self.config = config
        self.logger = logger

        self.input_dim = input_dim
        self.model = Model(self.input_dim, config.T).to(config.device)
        self.model.apply(self._weights_init)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8
        )

        if os.path.isfile(config.model_path):
            data = torch.load(config.model_path, map_location=self.config.device_name)
            self.model.load_state_dict(data["model"])

        beta0 = 1e-4
        betaT = 2e-2
        self.beta = torch.linspace(beta0, betaT, config.T)

        self.a_cum = torch.cumprod(1 - self.beta, dim=0)
        a_cum_prev = torch.cat((torch.Tensor([0]), self.a_cum[:-1]))
        self.sigma = (1 - a_cum_prev) / (1 - self.a_cum) * self.beta

        self.steps = 0
        self.writer = tensorboard.SummaryWriter(log_dir=config.tensorboard_log_dir)

    def save(self, epoch, model_path):
        data = {
            "model": self.model.state_dict(),
        }
        torch.save(data, model_path)
        print(f"save model to {model_path}")

    def _weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find("Linear") != -1:
            nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0)

    def train(self, dataloader, epoch: int):
        self.model.train()
        self.logger.info(f"start training epoch {epoch}")
        for (x, _) in dataloader:
            x = x.to(self.config.device)
            self.step(x.view(x.shape[0], -1))
            self.step_end()

    def step(self, x):
        e = torch.randn_like(x, device=self.config.device)
        t = torch.randint(0, self.config.T, (x.shape[0],), device=self.config.device)
        input = self.q_sample(x, t, e)
        output = self.model(input, t)

        # print(f"noise: {e}")
        # print(f"x: {x}")
        # print(f"input : {input}")
        # print(f"output: {output}")
        # print(list(self.model.parameters()))

        loss = nn.functional.mse_loss(output, e)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        if self.steps % self.config.log_interval == 0:
            logger.info(f"[train] step: {self.steps}, loss: {loss:.3f}")
        self.writer.add_scalar("loss/train", loss, self.steps, time.time())

    def step_end(self):
        self.steps += 1

    @torch.no_grad()
    def evaluate(self, dataloader, epoch: int):
        self.model.eval()
        self.logger.info(f"evaluate epoch {epoch}")

        self.save(epoch, self.config.model_path)

        samples = self.sample(2)
        print("samples")
        print(dataloader.dataset.denormalize(samples))

    def q_sample(self, x0, t, e=None):
        if e is None:
            e = torch.randn_like(x0, device=self.config.device)
        a_cum = self.a_cum[t].view(-1, 1)
        return torch.sqrt(a_cum) * x0 + torch.sqrt(1 - a_cum) * e

    @torch.no_grad()
    def sample(self, n: int, xt=None, t_start=None):
        if xt is None:
            xt = torch.randn((n, self.input_dim))
            print(xt)

        if t_start is None:
            t_start = self.config.T - 1
        ts = range(t_start, 0, -1)

        for t in ts:
            z = (
                torch.randn((n, self.input_dim), device=self.config.device)
                if t > 1
                else 0
            )
            e = (
                self.beta[t]
                / torch.sqrt(1 - self.a_cum[t])
                * self.model(xt, torch.tensor(t).repeat(n))
            )
            xt = (xt - e) / torch.sqrt(1 - self.beta[t]) + torch.sqrt(self.sigma[t]) * z

        return xt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--device_name",
        default="cpu",
        choices=["cpu", "mps"],
        help="which devices to run on",
    )
    parser.add_argument("--dataroot", default="data", help="path to data")
    parser.add_argument(
        "--name",
        default="default",
        help="name of training, used to model name, log dir name etc",
    )
    parser.add_argument("--epochs", type=int, default=1000, help="epoch count")
    parser.add_argument("--batch_size", type=int, default=2, help="size of batch")
    parser.add_argument(
        "--log_interval", type=int, default=10, help="step num to display log"
    )
    parser.add_argument("--model_path", default=None, help="model path")
    parser.add_argument(
        "--eval_interval_epochs",
        type=int,
        default=5,
        help="evaluate by every this epochs ",
    )
    parser.add_argument(
        "--T", type=int, default=1000, help="time step of diffusion process"
    )
    parser.add_argument("--sample_only", action="store_true")
    parser.add_argument("--sample_from", type=float, nargs="*")
    parser.add_argument("--sample_t_start", type=int, default=None)

    config = parser.parse_args()

    config.device = torch.device(config.device_name)
    if config.model_path is None:
        config.model_path = f"{config.dataroot}/{config.name}.pth"

    config.tensorboard_log_dir = f"{config.dataroot}/runs/{config.name}"
    os.makedirs(config.tensorboard_log_dir, exist_ok=True)

    logger = setup_logger(name=__name__)

    input_dim = 28 * 28
    trainer = Trainer(config, logger, input_dim=input_dim)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1),
    ])
    dataset = datasets.MNIST(config.dataroot, train=True, download=True, transform=transform)
    dataloader = utils.data.DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True
    )

    if config.sample_only:
        logger.info(
            f"sample from: {config.sample_from}, model_path: {config.model_path}"
        )
        n = 10
        xt = (
            None
            if config.sample_from is None
            else dataset.normalize(torch.tensor([config.sample_from] * n))
        )
        x0 = trainer.sample(n, xt, config.sample_t_start)
        print(dataset.denormalize(x0))
        sys.exit()

    for epoch in range(config.epochs):
        trainer.train(dataloader, epoch)

        # if epoch % config.eval_interval_epochs == 0:
        #     trainer.evaluate(dataloader, epoch)
