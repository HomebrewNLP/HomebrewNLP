# SiPaR

SiPaR, the Single-Pass RNN, is an RNN that requires only one forward pass over the entire sequence. No matter what. Idea based on [this](https://arxiv.org/pdf/1909.00021.pdf) work.\
Combined with the Reversible RNN (RevRNN), SiPaR not only uses O(n) memory but instead only has ~30MB + input size as special requirement.\
Note that it currently has gradient issues, as it's a linear function wrapper in a [reversible](https://arxiv.org/pdf/1707.04585.pdf) manner.