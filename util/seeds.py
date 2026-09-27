import json
import numpy as np
from pathlib import Path

ARQUIVO_SEEDS = "seeds.json"

def gerar_seeds(quantidade):
    return np.random.randint(0, 100000, size = quantidade).tolist()

class GerenciadorSeeds:

    def __init__(self):
        self.seed_base = None
        self.seeds = None

    def carregar(self):
        with open(ARQUIVO_SEEDS, "r") as f:
            dados = json.load(f)
        self.seed_base = dados["seed base"]
        self.seeds = dados["seeds"]

    def salvar(self):
        with open(ARQUIVO_SEEDS, "w") as f:
            json.dump({
                "seed base": self.seed_base,
                "seeds": self.seeds
            }, f, indent=4)

    def regerar_seeds(self, quantidade, seed_base):
        np.random.seed(seed_base)
        self.seed_base = seed_base
        self.seeds = gerar_seeds(quantidade)
        self.salvar()

    def iniciar(self, quantidade, seed_base):
        if not Path(ARQUIVO_SEEDS).is_file():
            self.regerar_seeds(quantidade, seed_base)
        self.carregar()
        if seed_base != self.seed_base:
            self.regerar_seeds(quantidade, seed_base)

    def incrementar_seeds(self, quantidade):
        self.seeds += gerar_seeds(quantidade)
        self.salvar()

    def get_seed_i(self, i, seed_base):
        if (not self.seeds) or (not self.seed_base):
            self.iniciar(i + 1, seed_base)
        elif seed_base != self.seed_base:
            self.regerar_seeds(i + 1, seed_base)
        elif i >= len(self.seeds):
            self.incrementar_seeds(i + 1 - len(self.seeds))
        return self.seeds[i]
