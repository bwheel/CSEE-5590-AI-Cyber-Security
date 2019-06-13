#!/usr/bin/env python

try:
    import torch
    
    device_id = torch.cuda.current_device()
    print(f"Current device id {device_id}")

    # set the working device
    torch.cuda.device(device_id)

    device_count = torch.cuda.device_count()
    print(f"Total number of available-cuda GPUs {device_count}")

    device_name = torch.cuda.get_device_name(device_id)
    print(f"The name of the current device {device_name}")
    
    # check if cuda is working.
    if torch.cuda.is_available():
        print("Success")
    else:
        print("Error")
except Exception as ex:
    print("Cuda is not properly installed: " + str(ex))