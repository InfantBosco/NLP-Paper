import time
from typing import Callable, Dict, Any

def benchmark_inference_speed(func: Callable, num_sentences: int, warmups: int = 2, runs: int = 5) -> Dict[str, Any]:
    """Measures inference throughput (sentences/sec) and total latency."""
    for _ in range(warmups):
        func()

    latencies = []
    for _ in range(runs):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        latencies.append(end - start)

    avg_latency = float(sum(latencies) / len(latencies))
    sents_per_sec = float(num_sentences / avg_latency) if avg_latency > 0 else 0.0

    return {
        "num_sentences": num_sentences,
        "avg_latency_sec": avg_latency,
        "sentences_per_sec": sents_per_sec
    }
