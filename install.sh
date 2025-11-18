mkdir -p pre_trained/OPENAI_CLIP/
mkdir -p pre_trained/META_DINOV2/
mkdir -p pre_trained/META_DINOV3/LVD/
mkdir -p pre_trained/META_DINOV3/SAT/

sudo apt install unzip
pip uninstall torchaudio -y
pip install --upgrade -r requirements.txt

# gdown "https://drive.google.com/drive/folders/1fm3Jd8lFMiSP1qgdmsxfqlJZGpr_bXsx?usp=drive_link" -O pre_trained/OPENAI_CLIP/ --folder
wget "https://www.dropbox.com/scl/fi/9e1fdt4u3g9npeixsg61d/OPENAI_ViT_L_14.zip?rlkey=vz5fn421z0mvh4rvapnim30fg&st=drtu25ka&dl=0" -O pre_trained/OPENAI_ViT_L_14.zip
unzip pre_trained/OPENAI_ViT_L_14.zip -d pre_trained/OPENAI_CLIP/
mv pre_trained/OPENAI_CLIP/OPENAI_ViT_L_14/** pre_trained/OPENAI_CLIP/
rm -rf pre_trained/OPENAI_ViT_L_14.zip
rm -rf pre_trained/OPENAI_CLIP/OPENAI_ViT_L_14/

hf download "facebook/dinov2-large" --local-dir pre_trained/META_DINOV2/ --cache-dir pre_trained/META_DINOV2/temp/
hf download "facebook/dinov3-vitl16-pretrain-lvd1689m" --local-dir pre_trained/META_DINOV3/LVD/ --cache-dir pre_trained/META_DINOV3/LVD/temp/
hf download "facebook/dinov3-vitl16-pretrain-sat493m" --local-dir pre_trained/META_DINOV3/SAT/ --cache-dir pre_trained/META_DINOV3/SAT/temp/

rm -rf pre_trained/APPLE_DFN_CLIP/temp/
rm -rf pre_trained/META_DINOV2/temp/
rm -rf pre_trained/META_DINOV3/LVD/temp/
rm -rf pre_trained/META_DINOV3/SAT/temp/
pip cache purge
git update-index --skip-worktree configs/default_config.yaml
mkdir -p logs/