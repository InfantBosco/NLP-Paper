import os
import matplotlib.pyplot as plt

def plot_efficiency_curve(corpus_sizes, latencies, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(corpus_sizes, latencies, marker="o", label="SBERT Bi-Encoder")
    plt.xlabel("Corpus Size (Sentences)")
    plt.ylabel("Latency (Seconds)")
    plt.title("Corpus Size vs Search Latency")
    plt.grid(True)
    plt.legend()
    plt.savefig(output_path)
    plt.close()
