name=train_davis
echo $name
tgt_dir='val_dir/'$name
if [ ! -d $tgt_dir ]; then
	mkdir -p $tgt_dir
fi
python3 infer_davis.py \
	--root-data='data/davis2017/trainval' \
	--root-all-data='data/davis2017/trainval' \
	--list-path='data/davis2017/trainval/val_meta.json' \
	--restore='checkpoints/train_davis/model_199.pth' \
	--batch-size=1 \
	--start-epoch=0 \
	--epoch=1 \
	--sample-size=20 \
	--lr=1e-5 \
	--gpu='7' \
	--sample-dir=$tgt_dir'/sample' \
	--save-dir=$tgt_dir'/save' \
	--test-mode=1 \
	2>&1 | tee $tgt_dir/val.log
