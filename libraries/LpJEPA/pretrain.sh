set -e
set -u

echo "Pretraining launched $(date)"
python main.py pretrain -t sample -e 100 -b 64 -p 32 -w 1 -c
python main.py plot -o loss