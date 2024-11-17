## Todo
- Examine why LAGAT is performing worse?
    - probably because the implementation of GAT is not as good as PyG ones 
- Implement Neighbor Sampler to prevent lable leakage in training (though not a big influence)
- Modulize the code and run more experiments
- Parallelly run a graphformer if possible
- Add documentation for LAGAT Conv Class

## Done
- Improve LAGAT Conv Layer for efficiency
    - Precompute alpha value instead of concatenation in message
    - Use torch operation to speed up indexing
- Fix bugs caused by "hint comment" (used by TorchScript JIT to infer the propagate type):
- Implement LAGAT on top of current GATConv from PyG
    - Add label attention mechanism to current GAT



- 