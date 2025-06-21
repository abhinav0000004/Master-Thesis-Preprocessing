# 📱 Smartphone Pose Estimation

This project provides preprocessing, visualization, and benchmarking tools for the **MaD Pose Challenge (MPC)** dataset. It focuses on cleaning and formatting marker data, comparing interpolation methods, and preparing the data for integration with the [AddBiomechanics](https://addbiomechanics.org/) platform.

> ⚠️ Note: Model training and evaluation are handled separately in a [shared repository](https://mad-srv.informatik.uni-erlangen.de/empkins/student-projects/gap-0/pose-estimation-benchmarking).

---

## 📁 Project Structure

```bash
.
├── implementation/             # Python scripts for preprocessing and experiments
│   ├── experiments/            # Scripts for plotting and error benchmarking
│   ├── helper/                 # Utility scripts
│   └── preprocessing/          # Dataset preparation scripts
├── markerset/                  # OpenSim marker set model (PlugInGait adjusted)
│   └── PlugingaitFullBody_fixedmarkers.osim
├── output/                     # Script output: processed files, CSVs, etc.
├── Scripts/                    # PowerShell scripts for bulk renaming and conversion
│   ├── rename_c3ds.ps1
│   ├── rename_mp4s.ps1
│   └── vid_to_img.ps1
└── conda_env.yml               # Conda environment file with dependencies
```

---

## 🧪 Experiments

### 📊 Plot C3D Marker Data  
**Script:** `implementation/experiments/plot.py`  
- Visualize individual or multiple C3D marker sets  
- Compare ground truth vs interpolated outputs

### 📉 Benchmark GPR vs Linear Interpolation  
**Script:** `implementation/experiments/lerp_vs_gpr.py`  
- Compares reconstruction error using Linear Interpolation vs Gaussian Process Regression (GPR)  
- ⚙️ Set the `DATA_DIR` variable to your MPC dataset location before running

---

## 🏗️ MPC Dataset Creation Workflow

This section outlines the full process to create a clean and structured MPC dataset.

### 1. 📂 Folder Structure
- Rename subject folders to `S1`, `S2`, ..., `S8`
- Create subfolders inside each subject:
  ```
  S1/
  ├── c3ds/
  └── videos/
  ```
- Move C3D and video files into these folders
- Delete unused or static trial files
- Add `participant_infos.csv` to the root

### 2. 📝 Rename Files
Use the PowerShell scripts from the `Scripts/` directory:
```powershell
# From inside 'S1/videos'
<Path>\rename_mp4s.ps1 -SubjectId 1

# From inside 'S1/c3ds'
<Path>\rename_c3ds.ps1 -SubjectId 1
```

### 3. 🎥 Convert Videos to Images
Extract image frames from videos using:
```powershell
<Path>\vid_to_img.ps1
```

### 4. 🧹 Preprocess C3D Data
1. Fix Jumping Jack files for subject S3:
   ```bash
   python implementation/preprocessing/handle_S3_jumpingjacks.py
   ```

2. Preprocess all C3D files:
   ```bash
   python implementation/preprocessing/fix_c3d_folder.py "<input_c3d_path>" "<output_path>"
   ```

   Example:
   ```bash
   python fix_c3d_folder.py "E:/Dataset/S1/raw_c3d" "E:/Dataset/S1/preprocessed_c3d"
   ```

---

## 🧬 AddBiomechanics Pipeline

1. Use the [AddBiomechanics](https://addbiomechanics.org/) tool to process preprocessed C3D files  
2. Use the custom OpenSim model:  
   `markerset/PlugingaitFullBody_fixedmarkers.osim`

3. Extract JSON outputs from AddBiomechanics:
   ```bash
   python implementation/preprocessing/osim_to_json.py
   ```

4. ⚠️ **Note:** Bounding box generation not yet implemented
