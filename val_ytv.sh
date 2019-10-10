name=train_ytv
echo $name
tgt_dir='val_dir/'$name
if [ ! -d $tgt_dir ]; then
	mkdir -p $tgt_dir
fi
python3 infer_ytv.py \
	--root-data='data/youtube_vos/valid/' \
	--root-all-data='data/youtube_vos/valid_all_frames/' \
	--list-path='data/youtube_vos/valid/meta.json' \
	--restore='checkpoints/train_ytv/model_4.pth' \
	--batch-size=1 \
	--start-epoch=0 \
	--epoch=1 \
	--sample-size=20 \
	--lr=1e-5 \
	--gpu='4' \
	--sample-dir=$tgt_dir'/sample' \
	--save-dir=$tgt_dir'/save' \
	--test-mode=1 \
	--show-step=1 \
	2>&1 | tee $tgt_dir/val.log
