name=train_davis
echo $name
tgt_dir='test_dir/'$name
if [ ! -d $tgt_dir ]; then
	mkdir -p $tgt_dir
fi
python3 infer_davis.py \
	--root-data='data/davis2017/test' \
	--root-all-data='data/davis2017/test' \
	--list-path='data/davis2017/test/test_meta.json' \
	--restore='checkpoints/train_davis/model_199.pth' \
	--batch-size=1 \
	--start-epoch=0 \
	--epoch=1 \
	--sample-size=20 \
	--lr=1e-5 \
	--gpu='2' \
	--sample-dir=$tgt_dir'/sample' \
	--save-dir=$tgt_dir'/save' \
	--test-mode=1 \
	2>&1 | tee $tgt_dir/val.log
