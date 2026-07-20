from enum import Enum
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TConfig:
    kBlockM: int
    kBlockN: int
    kStage: int
    producer_reg_dealloc: int
    consumer_reg_alloc: int
    p_smem_k_tiles: int
    q_reg_k_tiles: int
    num_consumer: int
    use_scheduler_barrier: int
    rescale_o_before_gemm: int

    def name(self) -> str:
        return (
            f"bm{self.kBlockM}_bn{self.kBlockN}_st{self.kStage}"
            f"_prd{self.producer_reg_dealloc}_cra{self.consumer_reg_alloc}"
            f"_p{self.p_smem_k_tiles}_q{self.q_reg_k_tiles}"
            f"_nc{self.num_consumer}_sb{self.use_scheduler_barrier}"
            f"_rb{self.rescale_o_before_gemm}"
        )

    def base_name(self) -> str:
        return (
            f"bm{self.kBlockM}_bn{self.kBlockN}_st{self.kStage}"
            f"_p{self.p_smem_k_tiles}_q{self.q_reg_k_tiles}"
            f"_nc{self.num_consumer}"
        )

@dataclass
class CompileResult:
    config: TConfig
    exec_file: Path

@dataclass
class BenchResult:
    rank: int
    config: TConfig
    exec_file: Path
    time_ms: float
    tflops: float
    smem_size: int
    reg_size: int

class Mode(Enum):
    RADICAL = 0
    KEEP = 1

class DType(Enum):
    FP8 = 1
    FP16 = 2

Shape = tuple[int, int, int, int]
