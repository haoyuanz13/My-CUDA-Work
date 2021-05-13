# CUDA ICP

这里提供一个CUDA ICP的工具，利用GPU来加速ICP的实现，主要是基于开源代码进行的整理和二次开发，参考资料如下，
- 开源代码: https://github.com/meiqua/pose_refine
- 参考资料: https://zhuanlan.zhihu.com/p/58757649

而当前工具会将cuda icp实现一个`功能动态库`，实现`src点云 到 dst点云`的转换矩阵的求取，从而用于适配其他需要的任务


## Depends
该工具需要依赖GPU的编程环境，下面给出一个成功编译和运行的环境，
- `Linux` Ubuntu 16.04, x86
- `GPU` 2080 ti
- `CUDA` 10.0
- `Opencv` 3.4.7
- `PCL` 1.8
- `Eigen3`

在编译的过程中如果有遇到`vtk`版本的问题，请检查`PCL`的安装方式。另外，如果在编译过程中遇到了gcc和nvcc的冲突，请参考[相关solution](https://answers.ros.org/question/251156/unable-to-solve-nvcc-fatal-a-single-input-file-is-required-for-a-non-link-phase-when-an-outputfile-is-specified-error/)


## Execution
下面简单介绍一下如何使用当前的工具库。

#### 1. 编译
在完成代码环境的准备后，运行脚本，
```bash
sh compile.sh 
```

成功编译后，在项目路径的`build`文件夹下，你可以获得三个动态库和一个可执行demo文件，
- `libcpp_icp.so` icp cpp功能的动态库
- `libcuda_icp.so` icp cuda功能的动态库
- `libicp_worker.so` 封装得到的icp worker动态库
- `demo_icp_worker` 一个演示cuda icp demo的可执行文件


#### 2. demo 运行
这里在`data`文件夹下提供了两个测试的点云数据，你也可以准备自己的点云数据，然后运行脚本，
```bash
sh demo.sh
```

成功运行，你可以看到cuda icp几个主要模块所需要的时间，以及icp获得的变换矩阵。<br>
p.s. demo中同一份数据循环了10次，主要是第一次运行会进行内存显存的分配，速度较慢，后续的循环才为实际的cuda icp的速度。


另外，这里也提供了`可视化的脚本`来帮助判断icp的效果，主要使用到了open3d的相关工具，运行脚本如下，
```bash
sh vis_res.sh
```

运行成功后可以得到如下的可视化效果，其中src点云是红色，dst点云是绿色，
<div align=center>
  <img width="640" height="360" src="./docs/cuda_icp_demo.gif", alt="res demo"/>
</div>


#### 3. 适配其他项目
如果要将当前cuda icp功能库用于其他的需求，则需要，
- 将编译获得的三个动态库都加入到项目中
- 将工程中涉及到的头文件加入到项目中
- 在项目的CMakeLists.txt中加入正确的动态库和头文件地址



