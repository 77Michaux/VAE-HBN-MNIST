set -u

echo "Data prep launched: $(date)"
python main.py prep -t sample
python main.py prep -t full