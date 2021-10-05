import sys
import torch
import pickle
import argparse
import numpy as np


def get_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", help="# VAE training epochs", type=int, default=100)
    parser.add_argument("--lr", help="Learning rate", type=float, default=1e-3)
    parser.add_argument("--log_freq", help="Interval for printing losses", type=int, default=3)
    parser.add_argument("--batch_size", help="# samples in mini-batch", type=int, default=32)
    parser.add_argument("--gpuid", help="ID of the GPU", type=int, default=0)
    args = parser.parse_args(argv)
    return args


class BufferDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        states: torch.Tensor,
        next_states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor):
        self.dones = dones

        self.states_actions = torch.cat((states.squeeze(1), actions.squeeze(1)), dim=1)
        self.n_states_rwd = torch.cat((next_states.squeeze(1), rewards), dim=1)

    def __len__(self):
        return len(self.states_actions)
    
    def __getitem__(self, idx):
        return self.states_actions[idx], self.n_states_rwd[idx]


def dataset_from_pickle(buffer: pickle) -> torch.utils.data.Dataset:
    states = torch.tensor(buffer.observations)
    next_states = torch.tensor(buffer.next_observations)
    rewards = torch.tensor(buffer.rewards)
    actions = torch.tensor(buffer.actions)
    dones = torch.tensor(buffer.dones)
    return BufferDataset(states, next_states, actions, rewards, dones) 


class VAE(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)

    def forward(self, x):
        z_mu, z_var = self.encoder(x)
        std = torch.exp(z_var / 2)
        eps = torch.randn_like(std)
        x_sample = eps.mul(std).add_(z_mu)
        recon_x = self.decoder(x_sample)
        return recon_x, z_mu, z_var


class Encoder(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.first = torch.nn.Linear(36, 150)
        self.second = torch.nn.Linear(150, 100)
        self.mu, self.var = torch.nn.Linear(100, 10), torch.nn.Linear(100, 10)

    def forward(self, x):
        x = self.first(x)
        x = torch.nn.functional.leaky_relu(x)
        x = self.second(x)
        x = torch.nn.functional.leaky_relu(x)
        return self.mu(x), self.var(x)

 
class Decoder(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.first = torch.nn.Linear(10, 100)
        self.second = torch.nn.Linear(100, 150)
        self.out = torch.nn.Linear(150, 29)


    def forward(self, x):
        x = self.first(x)
        x = torch.nn.functional.leaky_relu(x)
        x = self.second(x)
        x = torch.nn.functional.leaky_relu(x)
        return self.out(x)


def train(
    args, 
    model: torch.nn.Module, 
    loader: torch.utils.data.DataLoader,
    device: torch.device
    ) -> torch.nn.Module:

    criterion = torch.nn.MSELoss(reduction='sum')
    optim = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        combined_loss, reg_loss, pred_loss = [], [], []
        for state_action, n_state_reward in loader:
            state_action = state_action.to(device)
            n_state_reward = n_state_reward.to(device)
            prediction, z_mu, z_var = model(state_action)
            loss = criterion(n_state_reward, prediction)
            pred_loss.append(loss.item())
            kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1.0 - z_var)
            reg_loss.append(kl_loss.item())
            loss += kl_loss
            combined_loss.append(loss.item())
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
        if epoch % args.log_freq == 0 or epoch == (args.epochs - 1):
            log(f"Epoch: {epoch}/{args.epochs} -- loss: {np.mean(combined_loss):.3f} -- reg loss: {np.mean(reg_loss):.3f} -- pred loss: {np.mean(pred_loss):.3f}")
    return model


def log(text: str):
    print(text)
    print(text, file=open("logfile.txt", "a"))


if __name__ == "__main__":
    args = get_args(sys.argv[1:])
    f = open("logfile.txt", "w")
    device = torch.device(f"cuda:{args.gpuid}") if args.gpuid >= 0 else torch.device("cpu")
    with open('logs/sac/AntBulletEnv-v0_2/replay_buffer.pkl', 'rb') as f:
        buffer = pickle.load(f)
    dataset = dataset_from_pickle(buffer)    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, 
        ((len(dataset) - len(dataset)//10), len(dataset)//10)
    )
    t_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        # batch_size=args.batch_size,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    v_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    vae = VAE(args).to(device)
    vae = train(args, vae, t_dataloader, device)
    validate(args, model, v_dataloader, device)
