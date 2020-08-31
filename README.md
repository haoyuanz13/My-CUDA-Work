# My CUDA Work
A repository to store my cuda codes, including some common-used kernels. 

## Deps
You are supposed to get a GPU workspace first, below is one of the succeeded examples, 
- `Ubuntu` 16.04
- `Opencv` ≥ 3.1
- `CUDA` 10.0
- `CuDNN` v7.6.5
- `Eigen` ≥ 3.3


## Execution
```bash
git clone https://github.com/haoyuanz13/My-CUDA-Work.git
cd My-CUDA-Work
```

Before compiling, change the `DCUDNN_LIBRARY` in the file `compile.sh`, then, 
```bash
sh compile.sh
sh run.sh
```


## Clarification
All the demo main codes are stored in the folder `src` that are implemented via `cpp`, and the kernels also other further header files will be ordered under the `common`.

In addition, some of the cuda kernel implementation topics are listed below, will keep learning and uploading, <br>
- [x] The gray image normalization, check `src/main_gray_normalize.cpp`
- [ ] The rgb image normalization, including channels flip, e.g. bgr to rgb
- [ ] The center-aligned based image padding 
- [ ] The biliner interpolation based image resize
- [ ] The feature map postprocess for the detection scenario
- [ ] The feature map postprocess for the classification scenario, using reducing algorithm
- [ ] The 3D pointclouds generation based on the depth image and camera models


## More things
Just feel free to check this repository, also will be pleasure if these stuffs would do a favor to you.
