set -u
set -e

echo "Checking PyTorch device:"
IS_GPU=$(python -c "import torch; print(torch.cuda.is_available())")
DEVICE=$(python -c "import torch; print(torch.cuda.get_device_name(torch.cuda.current_device()))")
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
if test "$IS_GPU" = "True"
then
  DEVICE_TYPE="GPU"
else
  DEVICE_TYPE=""
fi
echo "PyTorch version: $TORCH_VERSION."
echo "$DEVICE_TYPE: $DEVICE"
echo "Press Enter to close this window."
read -r
exit