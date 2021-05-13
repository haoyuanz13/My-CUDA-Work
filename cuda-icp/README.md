# CUDA ICP Tool

Here we provide a CUDA-ICP Tool, using the GPU to speed up the icp implementation. <br>
The whole tool is developed based on a open-source repo [pose_refine](https://github.com/meiqua/pose_refine), also you can check a detailed reference via the [ZhiHu link](https://zhuanlan.zhihu.com/p/58757649)

This tool provides a cuda-icp `shared lib`, estimating a transformation matrix from the `src frame` to the `dst frame`.


## Depends
You are supposed to get a GPU workspace first, below is one of the succeeded dep combinations,
- `Linux` Ubuntu 16.04, x86
- `GPU` 2080 ti
- `CUDA` 10.0
- `Opencv` 3.4.7
- `PCL` 1.8
- `Eigen3`

If there was a `vtk version issue`, please check your PCL install approach. In addition, check the [solution](https://answers.ros.org/question/251156/unable-to-solve-nvcc-fatal-a-single-input-file-is-required-for-a-non-link-phase-when-an-outputfile-is-specified-error/) once the gcc conflicts with nvcc in your compile environment. 


## Execution
Below shows how to compile and execute this lib tool. 

#### 1. Compile
After the repo clone, execute the below shell scripts, 
```bash
sh compile.sh 
```

Once the compile task succeeded，you are supposed to get three `shared libs` and one `exe demo` under the folder `build`, 
- `libcpp_icp.so` the cpp-icp shared lib
- `libcuda_icp.so` the cuda-icp cuda shared lib
- `libicp_worker.so` the integrated icp functional worker shared lib
- `demo_icp_worker` the cuda icp demo exe file


#### 2. Run Demo
Here we provide two demo 3d pointcloud data under the folder `data`, 
```bash
sh demo.sh
```

`NOTE` in our demo we estimate two same dataset in loop times, since the very first time estimation will cost much more time to malloc memories. 


In addition，we provide a `visualization shell script` to help check the estimation result mainly using the open-source tool `open3d`, 
```bash
sh vis_res.sh
```

Then you are supposed to get a visual result like below(the red pcl is the src and the green pcl is the dst)
<div align=center>
  <img width="640" height="360" src="./docs/cuda_icp_demo.gif", alt="res demo"/>
</div>


#### 3. How to use this plugin
If you want to use this tool for other tasks or your own project, 
- put all compiled shared libs into your project
- put all involved header files into your project
- modify your project makefile or cmakelists to make sure it can find all above functional libs and headers



