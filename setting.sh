# Make Log Dir
mkdir ./_Output
mkdir ./_Output/log
mkdir ./_PoseEstimation/mmlab/mmpose/checkpoints
mkdir ./_HAR/HRI/models
mkdir ./_HAR/CSDC/models

# RTMO Checkpoints
wget "https://dguackr-my.sharepoint.com/:u:/g/personal/qqaazz0222_dgu_ac_kr/EbTvNVxo52REgaFHwtfuJokB5wEMOdyHbHgebBxW7OMT-w?e=jHzCnf&download=1" -O ./_PoseEstimation/mmlab/mmpose/checkpoints/rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.pth
wget "https://dguackr-my.sharepoint.com/:u:/g/personal/qqaazz0222_dgu_ac_kr/ERduj7hEcgFDiK7vUMZ0QF0BjrXez9ID3ifrdTt9Z_A6WQ?e=7Z9oBj&download=1" -O ./_PoseEstimation/mmlab/mmpose/checkpoints/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth

# PLASS Checkpoints
wget "https://dguackr-my.sharepoint.com/:u:/g/personal/qqaazz0222_dgu_ac_kr/EdjNyXlMOcNLjB8wwa0P_GkBKWfNIT6qlyF-DGJiqQSmmQ?e=cnEdn1&download=1" -O ./_HAR/PLASS/models/checkpoint.pth

# HRI Checkpoints
wget "https://dguackr-my.sharepoint.com/:u:/g/personal/qqaazz0222_dgu_ac_kr/EfyxWktMAahMk-6wJ2RfpcoBE39nuIBp41WI0wUIthdd0Q?e=BxrvKa&download=1" -O ./_HAR/HRI/models/Resnet50_Final.pth
wget "https://dguackr-my.sharepoint.com/:u:/g/personal/qqaazz0222_dgu_ac_kr/Eaa778hjqadBq_536FGvcJgBzOV5JbSRnz7bZRuidiKOXw?e=H4JNQD&download=1" -O ./_HAR/HRI/models/model_state.pth
wget "https://dguackr-my.sharepoint.com/:u:/g/personal/qqaazz0222_dgu_ac_kr/EfItQVH89CpCiIUCuAXF9fkBdD55mgG-7yU7dp33kNkSqA?e=yPueyM&download=1" -O ./_HAR/HRI/models/enet_b2_7.pt
cp /System_Integration/_HAR/HRI/models/Resnet50_Final.pth /root/.cache/torch/hub/checkpoints/

# CSDC Checkpoints
wget "https://dguackr-my.sharepoint.com/:u:/g/personal/qqaazz0222_dgu_ac_kr/Ea1MusgVUiZOtU4wVkgYW3ABZDiW8pEz4P5jNucZNbMA0Q?e=KYBlk0&download=1" -O ./_HAR/CSDC/models/tsstg-model-best-1.pth

cp /System_Integration/_HAR/HRI/models/Resnet50_Final.pth /root/.cache/torch/hub/checkpoints/