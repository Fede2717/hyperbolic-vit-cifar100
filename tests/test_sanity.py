import torch
from torch.utils.data import TensorDataset, DataLoader

from scripts.train import Config, build_model, accuracy, evaluate

def _tiny_loader(cfg: Config):
    x = torch.randn(8, 3, cfg.img_size, cfg.img_size)
    y = torch.randint(0, cfg.num_classes, (8,))
    return DataLoader(TensorDataset(x, y), batch_size=cfg.batch_size, shuffle=False)

def test_forward_euclid():
    cfg = Config()
    model = build_model(cfg, variant="euclid", progressive=False).eval()
    x = torch.randn(2, 3, cfg.img_size, cfg.img_size)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, cfg.num_classes)

def test_accuracy_and_evaluate():
    cfg = Config()
    model = build_model(cfg, variant="euclid", progressive=False).eval()
    loader = _tiny_loader(cfg)
    device = torch.device("cpu")
    model.to(device)
    xb, yb = next(iter(loader))
    xb, yb = xb.to(device), yb.to(device)
    logits = model(xb)
    a1, = accuracy(logits, yb, topk=(1,))
    assert 0.0 <= a1 <= 100.0
    loss, top1, top5 = evaluate(model, loader, device)
    assert loss >= 0.0 and 0.0 <= top1 <= 100.0 and 0.0 <= top5 <= 100.0

def test_progressive_builds():
    cfg = Config()
    # hyp-mlp + progressive = head+pos(+res quando lo colleghi)+mlp
    model = build_model(cfg, variant="hyp-mlp", progressive=True)
    assert model is not None

