"""Centralised settings loaded from .env via pydantic-settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Run mode
    MODE: str = "fast"
    SEED: int = 42

    # Data
    DATA_PATH: str = "data/AI_Human.csv"
    KAGGLE_DATASET: str = "shanegerami/ai-vs-human-text"
    RESULT_DIR: str = "result"
    TRAIN_RATIO: float = 0.7
    VAL_RATIO: float = 0.15
    TEST_RATIO: float = 0.15

    # Embeddings. The corpus path is RouterAI Qwen; s03 refuses any other backend.
    EMBEDDING_BACKEND: str = "routerai"
    OLLAMA_URL: str = "http://127.0.0.1:11434"
    VLLM_URL: str = "http://localhost:8000"
    ROUTERAI_BASE_URL: str = "https://routerai.ru/api/v1"
    ROUTERAI_API_KEY: SecretStr = SecretStr("")
    EMBEDDING_MODEL: str = "qwen/qwen3-embedding-8b"
    EMBEDDING_BATCH_SIZE: int = 32
    EMBEDDING_MAX_CONCURRENT: int = 4

    # DNN. DNN_LABEL_NOISE — скорость отдельного шумового плеча, не сравнение с CatBoost.
    DNN_WIDTHS: str = "8,16,32,64,128,256,512,1024,2048,4096"
    DNN_MODELWISE_EPOCHS: int = 800
    DNN_EPOCHWISE_EPOCHS: int = 4000
    DNN_LR: float = 0.001
    DNN_LABEL_NOISE: float = 0.1
    DNN_RUN_NOISE_ARM: bool = False
    DNN_CONVERGENCE_REL: float = 0.05

    # CatBoost. Оси сложности выключены на корпусе: их сначала проверяет малый пилот.
    CB_ITERATIONS: int = 20000
    CB_LR: float = 0.01
    CB_DEPTH: int = 8
    CB_L2_LEAF_REG: float = 50.0
    CB_RANDOM_STRENGTH: float = 20.0
    CB_BORDER_COUNT: int = 254
    CB_RUN_COMPLEXITY_AXES: bool = False
    CB_AXIS_ITERATIONS: int = 500
    CB_AXIS_DEPTHS: str = "4,6,8,10"
    CB_AXIS_BORDER_COUNTS: str = "32,128,254"
    CB_AXIS_ENSEMBLE_SIZES: str = "100,500,2000"

    # Независимые запуски малого протокола. Один SEED корпуса не даёт интервал.
    REPLICATE_SEEDS: str = "42,43,44"
    PCA_VARIANCE_TARGET: float = 0.95

    # Derived helpers
    @property
    def result_path(self) -> Path:
        return Path(self.RESULT_DIR)

    @property
    def dnn_widths_list(self) -> List[int]:
        return [int(w.strip()) for w in self.DNN_WIDTHS.split(",")]

    @property
    def replicate_seeds_list(self) -> List[int]:
        return [int(part.strip()) for part in self.REPLICATE_SEEDS.split(",") if part.strip()]

    @property
    def cb_axis_depths_list(self) -> List[int]:
        return [int(part.strip()) for part in self.CB_AXIS_DEPTHS.split(",") if part.strip()]

    @property
    def cb_axis_border_counts_list(self) -> List[int]:
        return [int(part.strip()) for part in self.CB_AXIS_BORDER_COUNTS.split(",") if part.strip()]

    @property
    def cb_axis_ensemble_sizes_list(self) -> List[int]:
        return [int(part.strip()) for part in self.CB_AXIS_ENSEMBLE_SIZES.split(",") if part.strip()]

    @property
    def is_fast(self) -> bool:
        return self.MODE == "fast"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
