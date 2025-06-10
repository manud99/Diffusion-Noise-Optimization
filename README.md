# Second-order Optimization Techniques for Diffusion Noise Optimization

In this repository, we extend the work of [Diffusion Noise Optimization (DNO)](https://arxiv.org/abs/2312.11994) by introducing second-order optimization techniques to improve the efficiency of DNO for motion editing tasks.

## 🛠️ Setup

### 🚀 Setup environment

First, clone this repository. Then, create and activate a conda environment using:

```bash
conda env create -f environment.yml
conda activate dno
```

### 📥 Download dependencies and pretrained models

Let's download the necessary dependencies for the project:

```bash
bash prepare/download_smpl_files.sh
bash prepare/download_glove.sh
bash prepare/download_t2m_evaluators.sh
```

Next, clone [HumanML3D](https://github.com/EricGuo5513/HumanML3D.git) and copy the data dir to our repository:

```shell
cd ..
git clone https://github.com/EricGuo5513/HumanML3D.git
unzip ./HumanML3D/HumanML3D/texts.zip -d ./HumanML3D/HumanML3D/
cp -r HumanML3D/HumanML3D Diffusion-Noise-Optimization/dataset/HumanML3D
cd Diffusion-Noise-Optimization
```

You're all set! 🎉

## 🚦 Usage

### 🛤️ Trajectory Editing

To edit a trajectory of a motion, run one of the following command for the respective optimizer:

```bash
python -m dno_optimized.generate config/trajectory_editing_adam.yml
python -m dno_optimized.generate config/trajectory_editing_lbfgs.yml
python -m dno_optimized.generate config/trajectory_editing_lm.yml
```

The results will be saved in a timestamped subfolder in `save/mdm_avg_dno/trajectory_editing`. They contain a video of the edited motions, log files to run with TensorBoard, and the edited motions in `.npy` format.

### 🕺 Pose Editing

To edit a pose of a motion, run one of the following command for the respective optimizer:

```bash
python -m dno_optimized.generate config/pose_editing_adam.yml
python -m dno_optimized.generate config/pose_editing_lbfgs.yml
python -m dno_optimized.generate config/pose_editing_lm.yml
```

The results will be saved in a timestamped subfolder in `save/mdm_avg_dno/pose_editing`. They contain a video of the edited motions, log files to run with TensorBoard, and the edited motions in `.npy` format.

### 📝 Modify configurations

You can modify the configurations either directly in the YAML files in the `config` folder or by passing command line arguments in the dotlist format. For example, to change the text prompt and the number of optimization steps for trajectory editing using Adam, you can run:

```bash
python -m dno_optimized.generate config/trajectory_editing_adam.yml text_prompt="a person is walking forward" dno.num_opt_steps=150
```

In `config/full_config.yml` you can find a list with the most useful configuration options.

### 📈 TensorBoard

To analyze the training process, you can use TensorBoard. Start it by running:

```bash
tensorboard --logdir_spec save/mdm_avg_dno
```
### 📊 Analyze results

To analyze the videos and results of many runs in a folder, e.g. `save/mdm_avg_dno/trajectory_editing`, run the following command:

```bash
python -m dno_optimized.analyze save/mdm_avg_dno/trajectory_editing
```

Open http://localhost:8000 in your browser to see the videos and results in a table.
To extract the results open http://localhost:8000/results where the results are nicely formatted in a table that can be copied to a spreadsheet.

## ⚡ Controversial Example

To produce the results for the controversial example discussed in section 4.5 our our report, run the following commands:

```bash
python -m dno_optimized.generate config/controversial_example_adam.yml
python -m dno_optimized.generate config/controversial_example_lbfgs.yml
python -m dno_optimized.generate config/controversial_example_lm.yml
```

The results will be saved in a timestamped subfolder in `save/mdm_avg_dno/controversial_example`. To see the videos have a look at the corresponding subfolders for the tree optimizers in `save/mdm_avg_dno/controversial_example`.

## 🖥️ Platform Details

This code was tested on ETH Zurich's [student cluster](https://www.isg.inf.ethz.ch/Main/HelpClusterComputingStudentCluster) using:

-   Ubuntu 24.04.2 LTS
-   Python 3.12
-   miniconda version 25.3.1
-   GPU: NVidia GTX 1080 Ti with 3584 CUDA cores and 11 GB RAM

## ℹ️ More Information on DNO

For more information on how to use the other DNO files, please have a look at the [main repo](https://www.github.com/korrawe/diffusion-noise-optimization).

## 📄 License

This code is distributed under an [MIT LICENSE](LICENSE).

Note that our code depends on other libraries, including CLIP, SMPL, SMPL-X, PyTorch3D, and uses datasets that each have their own respective licenses that must also be followed.
