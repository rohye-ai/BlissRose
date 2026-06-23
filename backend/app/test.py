import torch; 

print('torch:', torch.__version__); 
print('cuda available:', torch.cuda.is_available()); 
print('cuda version:', torch.version.cuda);
print('device:', torch.cuda.get_device_name(0)); 
x=torch.randn(3,3,device='cuda'); 
print('GPU tensor ok:', x.device, float(x.sum()))