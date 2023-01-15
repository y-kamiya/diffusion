import argparse
import os
import sys
import time

import torch
from logzero import setup_logger
from torch import nn, optim, utils
from torch.utils import tensorboard
from torchvision import datasets, transforms

import models


class Trainer(object):
    def __init__(self, config, logger, input_shape):
        self.config = config
        self.logger = logger

        in_channels, *self.input_size = input_shape
        self.model = models.Unet(in_channels=in_channels, dim_mults=(1,2,4)).to(config.device)
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

    @staticmethod
    def extract(a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

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
            self.step(x)
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
        a_cum = self.extract(self.a_cum, t, x0.shape)
        return torch.sqrt(a_cum) * x0 + torch.sqrt(1 - a_cum) * e

    @torch.no_grad()
    def sample(self, n: int, xt=None, t_start=None):
        if xt is None:
            xt = torch.randn((n, self.input_size))
            print(xt)

        if t_start is None:
            t_start = self.config.T - 1
        ts = range(t_start, 0, -1)

        for t in ts:
            z = (
                torch.randn((n, self.input_size), device=self.config.device)
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

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1),
    ])
    dataset = datasets.MNIST(config.dataroot, train=True, download=True, transform=transform)
    dataloader = utils.data.DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True
    )

    trainer = Trainer(config, logger, input_shape=(1, 28, 28))

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
