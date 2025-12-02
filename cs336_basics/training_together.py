import argparse
import math
import os
import time
import numpy as np
import torch
import wandb
from jaxtyping import Int, Float
from torch import Tensor

import cs336_basics
from cs336_basics import Tokenizer, run_transformer_lm


class LRScheduler:
    def __init__(self, optimizer, schedule_fn):
        self.optimizer = optimizer
        self.schedule_fn = schedule_fn
        self.last_lr = None

    def step(self, t):
        lr = self.schedule_fn(t)
        for g in self.optimizer.param_groups:
            g['lr'] = lr
        self.last_lr = lr
        return lr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Configure the Transformer training run."
    )
    file_dir = os.path.dirname(os.path.abspath(__file__))

    parser.add_argument("--save_model_path", type=str,
                        default=None,
                        help="Path to save model to. If not provided, won't save the model.")

    parser.add_argument("--tokenizer_dir", type=str,
                        default=os.path.join(file_dir, "../tokenizer/owt"),
                        help="Path to tokenizer dir. Two files vocab.txt and merges.txt are expected in there.")

    parser.add_argument("--train_path", type=str,
                        default=os.path.join(file_dir, "../tokenized", "tokenized-owt-TinyStoriesV2-GPT4-train.bin"),
                        help="Path to the training data file.")

    parser.add_argument("--valid_path", type=str,
                        default=os.path.join(file_dir, "../tokenized", "tokenized-owt-TinyStoriesV2-GPT4-valid.bin"),
                        help="Path to the validation data file.")

    parser.add_argument("--context_length", type=int, default=256,
                        help="Number of tokens per training example (default: %(default)s).", )

    parser.add_argument("--d_model", type=int, default=512,
                        help="Transformer hidden size (default: %(default)s).", )
    parser.add_argument("--num_layers", type=int, default=4,
                        help="Number of Transformer blocks (default: %(default)s).", )
    parser.add_argument("--num_heads", type=int, default=16,
                        help="Attention heads per layer (default: %(default)s).", )
    parser.add_argument("--d_ff", type=int, default=1344,
                        help="Feed-forward hidden size (default: %(default)s).", )
    parser.add_argument("--rope_theta", type=float, default=10000.0,
                        help="RoPE theta parameter (default: %(default)s).", )

    # parser.add_argument("--num_iterations", type=int, default=5000,
    #                     help="Number of training iterations to run (default: %(default)s).", )

    parser.add_argument("--max_tokens", type=int, default=100,
                        help="Max number of tokens to decode", )

    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to use for training. examples: cuda:0, mps, cpu", )

    parser.add_argument("--eval_every", type=int, default=200,
                        help="Evaluate every N iterations (default: %(default)s).", )

    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature", )

    parser.add_argument("--p", type=float, default=0.9,
                        help="Nucleus sampling parameter p (default: %(default)s).", )

    parser.add_argument("--use_wandb", action="store_true",
                        help="Whether to use wandb for logging.", )

    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for training (default: %(default)s).", )

    parser.add_argument("--training_budget", type=int, default=327680000,
                        help="Training budget in number of tokens (default: %(default)s).", )

    parser.add_argument("--alpha_max", type=float, default=1e-3,
                        help="Max learning rate (default: %(default)s).", )

    parser.add_argument("--alpha_min", type=float, default=1e-5,
                        help="Min learning rate (default: %(default)s).", )

    return parser.parse_args()


def evaluate(valid_tensor, context_length, model):
    stride = 50000
    acc, count = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(valid_tensor) - context_length - 1, stride):
            x = torch.from_numpy(
                valid_tensor[start:start + context_length]
            ).long().unsqueeze(0).to(model.device)

            pred = valid_tensor[start + context_length:start + context_length + 1]
            targets = torch.from_numpy(pred).long().to(model.device)
            out = model.forward(in_indices=x)

            res = cs336_basics.run_cross_entropy(
                inputs=out[:, -1, :],
                targets=targets, )

            acc += res.item()
            count += 1

    ppl = math.exp(acc / count)
    print(f"Validation perplexity: {ppl:0.3f}")
    return ppl


def next_tokens(logits: Float[Tensor, " bs len vocab_size"], temperature: float, p: float) -> Int[Tensor, " bs"]:
    """ Sample next tokens from the given logits using nucleus sampling (top-p sampling).
    Returns a tensor of shape (batch_size,) with the sampled token indices."""

    # keep only last token, no need to process all previous tokens
    last_logits = logits[:, -1, :]

    probs = cs336_basics.run_softmax_with_temperature(last_logits, dim=-1, temperature=temperature)

    sorted_probs, indices = torch.sort(probs, dim=-1, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)

    mask = (cumsum - sorted_probs) < p
    masked_probs = mask * sorted_probs

    selected = torch.multinomial(masked_probs, num_samples=1)

    result = torch.gather(indices, dim=-1, index=selected)

    return result.reshape((-1,))


def decode(prompt, max_tokens, model, tokenizer, temperature=1.0, p=0.9):
    ids = tokenizer.encode(prompt)
    x = torch.tensor(ids).long().unsqueeze(0)
    end_of_text = tokenizer.encode("<|endoftext|>")[0]
    next_token = None
    lst = []

    while next_token != end_of_text and max_tokens > 0:
        max_tokens -= 1
        out = model.forward(in_indices=x)
        token_indices = next_tokens(out, temperature=temperature, p=p)

        next_token = token_indices[0].item()
        lst.append(next_token)
        # print(f"Next token: {next_token} ('{tokenizer.decode([next_token])}')")

        x = torch.cat([x, torch.tensor([[next_token]])], dim=1)

    print("Decoded text:", tokenizer.decode(lst))


class ModelWrapper:
    def __init__(
            self,
            vocab_size: int,
            context_length: int,
            d_model: int,
            num_layers: int,
            num_heads: int,
            d_ff: int,
            rope_theta: float,
            device,
    ):
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.device = device

        self.weights = self.create_weights()
        assert all(w.requires_grad for w in self.weights.values()), "All weights must require gradients."

    def create_weights(self):
        weights = {
            'token_embeddings.weight': cs336_basics.Embedding(self.vocab_size, self.d_model, self.device).W,
            'ln_final.weight': torch.ones(self.d_model, requires_grad=True, device=self.device),
            'lm_head.weight': cs336_basics.Linear(self.d_model, self.vocab_size, device=self.device).weight,
        }

        for layer_idx in range(self.num_layers):
            weights.update({
                f'layers.{layer_idx}.attn.q_proj.weight': cs336_basics.Linear(self.d_model, self.d_model,
                                                                              device=self.device).weight,
                f'layers.{layer_idx}.attn.k_proj.weight': cs336_basics.Linear(self.d_model, self.d_model,
                                                                              device=self.device).weight,
                f'layers.{layer_idx}.attn.v_proj.weight': cs336_basics.Linear(self.d_model, self.d_model,
                                                                              device=self.device).weight,
                f'layers.{layer_idx}.attn.output_proj.weight': cs336_basics.Linear(self.d_model, self.d_model,
                                                                                   device=self.device).weight,
                f'layers.{layer_idx}.ln1.weight': torch.ones(self.d_model, requires_grad=True, device=self.device),
                f'layers.{layer_idx}.ffn.w1.weight': cs336_basics.Linear(self.d_model, self.d_ff,
                                                                         device=self.device).weight,
                f'layers.{layer_idx}.ffn.w2.weight': cs336_basics.Linear(self.d_ff, self.d_model,
                                                                         device=self.device).weight,
                f'layers.{layer_idx}.ffn.w3.weight': cs336_basics.Linear(self.d_model, self.d_ff,
                                                                         device=self.device).weight,
                f'layers.{layer_idx}.ln2.weight': torch.ones(self.d_model, requires_grad=True, device=self.device),
            })

        return weights

    def forward(self, in_indices: Int[Tensor, "batch_size sequence_length"]) -> Tensor:
        out = run_transformer_lm(
            vocab_size=self.vocab_size,
            context_length=self.context_length,
            d_model=self.d_model,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            rope_theta=self.rope_theta,
            weights=self.weights,
            in_indices=in_indices,
        )
        return out

    def state_dict(self):
        return self.weights


if __name__ == "__main__":
    args = parse_args()

    # Initialize wandb first (before using args for model creation)
    wandb_base_url = os.getenv("WANDB_BASE_URL", "https://wandb.gnlp.io")
    wandb.init(
        project="cs336",
        settings=wandb.Settings(base_url=wandb_base_url),
        mode="disabled" if not args.use_wandb else "online",
    )

    # Override args with sweep config (if running as sweep agent)
    if wandb.run and wandb.config:
        for key, value in wandb.config.items():
            if hasattr(args, key):
                setattr(args, key, value)

    # Read sweep params (with defaults for non-sweep runs)
    alpha_max = wandb.config.get("alpha_max", 1e-3)
    alpha_min = wandb.config.get("alpha_min", 1e-5)


    num_iterations = args.training_budget // (args.batch_size * args.context_length)
    print(f"Will run training for {num_iterations} iterations. Training budget: {args.training_budget} tokens. ")

    wandb.config.update({
        "context_length": args.context_length,
        "batch_size": args.batch_size,
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "d_ff": args.d_ff,
        "alpha_max": alpha_max,
        "alpha_min": alpha_min,
    })

    train_path = args.train_path
    valid_path = args.valid_path

    print(f"Using files: \ntrain_path={train_path} \nvalid_path={valid_path}")

    print(f"Using context_length={args.context_length}, d_model={args.d_model}, "
          f"num_layers={args.num_layers}, num_heads={args.num_heads}, d_ff={args.d_ff}, rope_theta={args.rope_theta}")

    tokenizer = Tokenizer.from_dir(args.tokenizer_dir, special_tokens=["<|endoftext|>"])

    vocab_size = tokenizer.vocab_size()
    print(f"Loaded tokenizer ({args.tokenizer_dir}) with vocab size {vocab_size}.")

    device = args.device
    model = ModelWrapper(
        vocab_size=vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
    )

    train_tensor = np.memmap(train_path, dtype=np.uint16, mode='r')
    valid_tensor = np.memmap(valid_path, dtype=np.uint16, mode='r')
    print(len(train_tensor), len(valid_tensor))

    optim = cs336_basics.MyAdamW(params=[w for w in model.weights.values()])

    Tw = 100
    Tc = num_iterations
    scheduler = LRScheduler(optim, lambda t: cs336_basics.run_get_lr_cosine_schedule(t, alpha_max, alpha_min, Tw, Tc))

    for it in range(1, num_iterations + 1):
        iter_start = time.perf_counter()

        x, y = cs336_basics.get_batch(dataset=train_tensor, context_length=args.context_length, batch_size=args.batch_size,
                                      device=device)
        lr = scheduler.step(it)
        optim.zero_grad()
        out = model.forward(in_indices=x)

        loss = cs336_basics.run_cross_entropy(out, targets=y)
        loss.backward()

        g_norm = cs336_basics.run_gradient_clipping(model.weights.values(), 1.0)

        if it and it % 10 == 0:
            wandb.log({
                "train/loss": loss.item(),
                "train/lr": lr,
                "train/grad_norm": g_norm,
                "iteration": it,
            })

        if args.eval_every >= 10 and it and it % (args.eval_every // 10) == 0:
            print(".", end="", flush=True)

        if it and it % args.eval_every == 0:
            elapsed = time.perf_counter() - iter_start
            print(
                f"{it}/{num_iterations}: lr {lr:.7f}, g_norm: {g_norm:0.5f}, loss {loss.item():0.5f}. {elapsed:0.3f} sec/iter")

            ppl = evaluate(valid_tensor, args.context_length, model)
            wandb.log({"eval/perplexity": ppl, "iteration": it})

            print("Decoding sample prompt...")
            decode("Once upon a time", args.max_tokens, model, tokenizer, temperature=args.temperature, p=args.p)

            if args.save_model_path is not None:
                data = (model.state_dict(), optim.state_dict(), it)
                print(f"Saving model to {args.save_model_path}...", end="", flush=True)
                torch.save(data, args.save_model_path)
                print(" done.")

        optim.step()

    wandb.finish()
