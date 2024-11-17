## Todo
- Examine why LAGAT is performing worse?
    - probably because the implementation of GAT is not as good as PyG ones 
- Implement Neighbor Sampler to prevent lable leakage in training (though not a big influence)
- Modulize the code and run more experiments
- Parallelly run a graphformer if possible

## Done
- Improve LAGAT Conv Layer for efficiency
    - precompute alpha value instead of concatenation in message
    - use torch operation to 