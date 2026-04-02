"""Centralised settings loaded from .env via pydantic-settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

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

    # Embeddings
    EMBEDDING_BACKEND: str = "ollama"
    OLLAMA_URL: str = "http://localhost:11434"
    VLLM_URL: str = "http://localhost:8000"
    EMBEDDING_MODEL: str = "embeddinggemma:latest"
    EMBEDDING_BATCH_SIZE: int = 64
    EMBEDDING_MAX_CONCURRENT: int = 16

    # DNN
    DNN_WIDTHS: str = "8,16,32,64,128,256,512,1024,2048,4096"
    DNN_MODELWISE_EPOCHS: int = 800
    DNN_EPOCHWISE_EPOCHS: int = 4000
    DNN_LR: float = 0.001
    DNN_LABEL_NOISE: float = 0.1

    # CatBoost
    CB_ITERATIONS: int = 20000
    CB_LR: float = 0.01
    CB_DEPTH: int = 8
    CB_L2_LEAF_REG: float = 50.0
    CB_RANDOM_STRENGTH: float = 20.0

    # Derived helpers
    @property
    def result_path(self) -> Path:
        return Path(self.RESULT_DIR)

    @property
    def dnn_widths_list(self) -> List[int]:
        return [int(w.strip()) for w in self.DNN_WIDTHS.split(",")]

    @property
    def is_fast(self) -> bool:
        return self.MODE == "fast"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
