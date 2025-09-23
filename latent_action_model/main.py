from lightning.pytorch.cli import LightningCLI
from genie.dataset import LightningOpenX
from genie.model import DINO_LAM
from genie.model import LAPA_LAM

cli = LightningCLI(
    # DINO_LAM,
    LAPA_LAM,
    LightningOpenX,
    seed_everything_default=42,
)
