# SlothCLIP

**SlothCLIP** is a lightweight and high-performance framework for fine-tuning OpenAI CLIP models using **LoRA**, **custom Triton kernels**. The project focuses on maximizing training efficiency while minimizing GPU memory usage, making CLIP adaptation practical on consumer GPUs.

The framework combines fused GPU kernels, custom autograd implementations, and optimized attention modules to accelerate training without modifying the original CLIP architecture.

---

## Features

* 🚀 Efficient CLIP fine-tuning with LoRA
* ⚡ Custom Triton implementation of Cross Entropy Loss
* 🧠 Custom LoRA Autograd for reduced memory consumption
* 🔥 PyTorch `torch.compile` optimization
* 📉 Lower GPU memory footprint than standard implementations
* 🎯 Support for image encoder, text encoder, or both

---

## Installation

Clone the repository

```bash
git clone https://github.com/canhdinhtien/slothclip.git
cd slothclip
```

Install dependencies

```bash
pip install -e .
```
---


## Optimizations

### Fast LoRA

SlothCLIP replaces the standard autograd implementation with a custom fused LoRA operator.

Benefits:

* Reduced intermediate tensor allocation
* Lower memory usage
* Faster backward propagation
* Better compatibility with `torch.compile`

---

### Triton Cross Entropy

The framework provides a custom Triton implementation of Cross Entropy Loss.

Characteristics:

* Numerically stable LogSumExp computation
* Ignore index support
* Fused forward computation
* Custom backward kernel
* Reduced memory overhead

---

### Torch Compile

The project is designed for PyTorch 2.x compiler optimizations.

Example configuration

```python
torch.compile(
    model,
    mode="max-autotune"
)
```

---

## Supported CLIP Models

Examples include

* ViT-B/16
* ViT-B/32
* ViT-L/14
* RN50

Additional CLIP-compatible models can be added with minimal modification.

---

## Supported Datasets

Current examples include

* ImageNet
* ImageNet-100
* Stanford Cars
* Oxford Pets
* Flowers102
* Food101
* FGVC Aircraft
* EuroSAT
* DTD
* SUN397
* Caltech101

---

## Performance Goals

SlothCLIP aims to provide:

* Faster CLIP fine-tuning
* Lower GPU memory consumption
* Efficient LoRA training
* High compatibility with modern PyTorch compilation

Performance benchmarks will be added in future releases.

---

## Roadmap

* [ ] Benchmark against PEFT
* [ ] Benchmark against Unsloth
* [ ] FlashAttention integration
* [ ] Multi-GPU training
* [ ] QLoRA support
* [ ] Hugging Face integration
* [ ] Benchmark on larger CLIP models

---

## Citation

If you use SlothCLIP in your research, please cite this repository.

```bibtex
@software{slothclip,
  title = {SlothCLIP},
  author = {Canh Dinh Tien},
  year = {2026},
  url = {https://github.com/canhdinhtien/slothclip}
}
```

---

## License

This project is released under the MIT License.

---

## Acknowledgements

This project builds upon the excellent work of:

* OpenAI CLIP
* LoRA
* Triton
* PyTorch
* timm
* loralib
* Unsloth
