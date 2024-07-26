#!/bin/bash
# Generate a table of scan event information (three columns: onset, duration, trial_type) and save under subject func folder in BIDS format.
# Input mat log file: '/data/local/u0151613/Qiuhan_Projects/Human_Prep_GLM_Qiuhan/HumanData_Preprocessing_SeedFixed/experiment/matlab_log/sub-01/run-01/sub-01_run-01_log.mat'
# Output tsv file: '/data/local/u0151613/Qiuhan_Projects/Human_Prep_GLM_Qiuhan/HumanData_Preprocessing_SeedFixed/bids/sub-01/ses-1/func/sub-01_ses-1_task-PAO_run-01_events.tsv'
# @Qiuhan Jin 29/05/2023

# Define subject, run list and seq list information
SurfaceLH='/data/local/u0151613/Human_Brain_Atlases/fsaverage_LR32k/fsaverage_LR32k/fsaverage.L.midthickness.32k_fs_LR.surf.gii'
SurfaceRH='/data/local/u0151613/Human_Brain_Atlases/fsaverage_LR32k/fsaverage_LR32k/fsaverage.R.midthickness.32k_fs_LR.surf.gii'



RootPath='/data/local/u0151613/Qiuhan_Projects/Encoding_Qiuhan/rsa_082023/results/output_refined_batch-1_human-27-r-5-ed_RSA-regout-v2/corrected'
OutputPath='/data/local/u0151613/Qiuhan_Projects/Encoding_Qiuhan/rsa_082023/results/output_refined_batch-1_human-27-r-5-ed_RSA-regout-v2/corrected'
mkdir -p $OutputPath
cd $RootPath
for ImageFile in $(printf '*.nii.gz'); do
	echo $ImageFile
	OutputImageName=${ImageFile%*.nii.gz}   # remove extension
	OutputImageLH=$(printf '%s/%s.LH.func.gii' $OutputPath $OutputImageName)
	OutputImageRH=$(printf '%s/%s.RH.func.gii' $OutputPath $OutputImageName)
	echo $OutputImageLH
	wb_command -volume-to-surface-mapping $ImageFile $SurfaceLH $OutputImageLH -trilinear
	wb_command -volume-to-surface-mapping $ImageFile $SurfaceRH $OutputImageRH -trilinear
done

