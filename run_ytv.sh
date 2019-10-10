name=train_ytv
echo $name
tgt_dir=Outputs/$name
if [ ! -d $tgt_dir ]; then
	mkdir -p $tgt_dir
fi
python3 train_ytv.py \
	--root-data='data/youtube_vos/train/' \
	--meta-list='data/youtube_vos/train/meta.json' \
	--restore='checkpoints/weights.pth' \
	--batch-size=1 \
	--start-epoch=0 \
	--epoch=5 \
	--random-ref \
	--lr-atn \
	--loss-iou-maxmin \
	--sample-size=8 \
	--lr=1e-5 \
	--gpu='6' \
	--sample-dir=$tgt_dir'/sample' \
	--snapshot-dir=$tgt_dir'/snapshot' \
	2>&1 | tee $tgt_dir/train.log
