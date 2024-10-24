import random
import os 
import numpy as np
import torch
import platform
import cpuinfo
import pickle
from torch_geometric.loader import DataLoader
from model import SegmentGNN,train_model,predict
import pandas as pd
import yaml
from sklearn.metrics import rand_score

seed=0
train_size=0.8
mini_batch_size=32
in_channels=3
hidden_channels=512
nclasses=1
ntrials=100
epochs=20
lr=0.001

def set_seed(seed: int) -> None:
    """
    Sets the random seed for the random, numpy, and PyTorch libraries.

    Args:
        - seed (int): The seed value to be used for random number generation.

    Returns:
        None: This function does not return any value. It only sets the seed for the specified libraries.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)


def identify_device():
    """
    Identifies the device (CPU or GPU) and returns the device and its name.

    Args:
        None

    Returns:
        Tuple[torch.device, str]: A tuple containing the device and its name.

    Raises:
        None

    The function first checks the system type. If it's a Darwin system (macOS), it checks if the MPS (Metal Performance Shaders) is available. If so, it sets the device to MPS, otherwise it sets it to CPU. If the system is not a Darwin system, it checks if the CUDA (Compute Unified Device Architecture) is available. If so, it sets the device to CUDA, otherwise it sets it to CPU.

    After setting the device, it retrieves the device name. If the device is a CUDA device, it gets the device name using torch.cuda.get_device_name(). If the device is a CPU, it gets the CPU information using cpuinfo.get_cpu_info()["brand_raw"].

    Finally, it returns a tuple containing the device and its name.
    """
    so = platform.system()
    if so == "Darwin":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        dev_name = cpuinfo.get_cpu_info()["brand_raw"]
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        d = str(device)
        if d == 'cuda':
            dev_name = torch.cuda.get_device_name()
            set_seed(seed)
        else:
            dev_name = cpuinfo.get_cpu_info()["brand_raw"]
    return device, dev_name




def load_patient_names():
    """
    Loads the names of the patients from the specified directory.

    Args:
        ns (str): The name of the directory containing the patient graphs.

    Returns:
        List[str]: A list of patient names sorted in alphabetical order.
    """
    path = f"../graphs"
    patients = sorted(os.listdir(path))
    print(f"Number of patients: {len(patients)}")
    return patients

def create_train_test_sets(patients):
    """
    This function creates training and testing datasets from a list of patients.

    Parameters:
    - patients (list): A list of patient names or IDs.
    - ns (str): A string representing the directory path where the patient data is stored.

    Returns:
    - trainloader (DataLoader): A DataLoader object containing the training dataset.
    - testloader (DataLoader): A DataLoader object containing the testing dataset.

    The function first shuffles the list of patients and then splits it into a training set and a testing set. It then loads the data for each patient from the specified directory, creates DataLoader objects for both the training and testing datasets, and returns them.
    """
    random.shuffle(patients)
    train_patients = patients[:int(train_size * len(patients))]
    test_patients = patients[int(train_size * len(patients)):]
    print(f"Train patients: {len(train_patients)}")
    print(f"Test patients: {len(test_patients)}")

    prefix = f"../graphs/"
    data = []

    for patient in train_patients:
        patient_path = os.path.join(prefix, patient)
        with open(patient_path, 'rb') as file:
            aus = pickle.load(file)
            data.extend(aus)
    trainloader = DataLoader(data, batch_size=mini_batch_size, shuffle=True)

    print(f"Training images: {len(data)}")

    data = []
    for patient in test_patients:
        patient_path = os.path.join(prefix, patient)
        with open(patient_path, 'rb') as file:
            aus = pickle.load(file)
            data.extend(aus)
    testloader = DataLoader(data, batch_size=1, shuffle=False)
    print(f"Test images: {len(data)}")
    return trainloader, testloader


def create_mask(segments, predicted_mask):
    """
    This function creates a binary mask from the given segmentation and predicted mask.

    Parameters:
    - segments (torch.Tensor): A tensor containing the segmentation of the input data. Each element in the tensor represents the segmentation of a pixel or a node in the graph.
    - predicted_mask (torch.Tensor): A tensor containing the predicted mask for each segment. Each element in the tensor represents the predicted label for a specific segment.

    Returns:
    - num_mask (torch.Tensor): A binary mask tensor where each element is set to the predicted label of the corresponding segment in the input segmentation tensor.

    The function iterates through each segment in the input segmentation tensor and sets the corresponding elements in the binary mask tensor to the predicted label of that segment. The binary mask tensor is initialized with zeros and has the same shape as the input segmentation tensor.
    """
    num_mask = np.zeros_like(segments,dtype=np.int32)
    for segment_id in range(1, segments.max() + 1):
        mask = (segments == segment_id)
        label = predicted_mask[segment_id - 1] 
        num_mask[mask] = label
    return num_mask

def create_predicted_masks(predictions, testloader):
    """
    This function creates predicted masks for each patient in the testloader based on the given predictions.

    Parameters:
    - predictions (List[torch.Tensor]): A list of predicted segmentation labels for each patient in the testloader.
    - testloader (torch.utils.data.DataLoader): A DataLoader object containing the test dataset.

    Returns:
    - true_masks (List[numpy.ndarray]): A list of true masks for each patient in the testloader.
    - pred_masks (List[numpy.ndarray]): A list of predicted masks for each patient in the testloader.

    The function iterates through each patient in the testloader and creates true masks and predicted masks for each patient. The true masks are obtained from the testloader, while the predicted masks are created using the given predictions and the segmentation labels from the testloader. The true masks and predicted masks are then returned as lists.
    """
    true_masks = []
    pred_masks = []
    for graph, pred in zip(testloader, predictions):
        num_mask = graph.num_mask.detach().cpu().numpy()
        segments = graph.segments.detach().cpu().numpy()
        true_masks.append(num_mask)
        pred_mask = create_mask(segments, np.array(pred))
        pred_masks.append(pred_mask)
    return true_masks, pred_masks


def dice_coefficient(target, prediction, smooth=1e-6):
    intersection = np.sum(np.logical_and(target, prediction))
    
    # Add smoothing to avoid division by zero
    dice_score = (2 * intersection + smooth) / (np.sum(target) + np.sum(prediction) + smooth)
    
    return dice_score



def pixel_accuracy(target, prediction):
    # Calculate the number of correctly predicted pixels
    correct_pixels = np.sum(target == prediction)
    
    # Calculate the total number of pixels
    total_pixels = target.size
    
    # Compute pixel accuracy
    accuracy = correct_pixels / total_pixels
    
    return accuracy

def pixel_error(target, prediction):
    assert prediction.shape == target.shape, "Shape mismatch between predicted and ground truth"
    
    # Count the number of pixels that are different
    incorrect_pixels = np.sum(prediction != target)
    
    # Calculate total number of pixels
    total_pixels = prediction.size
    
    # Compute pixel-wise error
    pixel_error = incorrect_pixels / total_pixels
    
    return pixel_error

def evaluate_model(trial,target,predictions):
    dices=[]
    rands=[]
    rand_errors=[]
    pixel_acc=[]
    pixel_errors=[]
    for t,p in zip(target, predictions):
        dice=dice_coefficient(t,p)
        dices.append(dice)
        r=rand_score(t.flatten(),p.flatten())
        rands.append(r)
        re=1-r
        rand_errors.append(re)
        pa=pixel_accuracy(t,p)
        pixel_acc.append(pa)
        pe=pixel_error(t,p)
        pixel_errors.append(pe)
        
    mean_dice = np.mean(dices)
    mean_rand=np.mean(rands)
    mean_rand_err=np.mean(rand_errors)
    mean_pixel_acc=np.mean(pixel_acc)
    mean_pixel_error=np.mean(pixel_errors)    
    
    print(f"Dice {mean_dice}")
    print(f"Rand {mean_rand}")
    print(f"Rand Error {mean_rand_err}")
    print(f"Pixel Accuracy {mean_pixel_acc}")
    print(f"Pixel Error {mean_pixel_error}")
    
    return [trial,mean_dice, mean_rand, mean_rand_err, mean_pixel_acc, mean_pixel_error]


def save_reports(devname,reports):
    columns=["Trial","Dice Coefficient","Rand Index","Rand Error","Pixel Accuracy","Pixel Error","Training Time (s)"]
    df=pd.DataFrame(reports,columns=columns)
    path=f"../experiments/gnn/metrics.csv"
    df.to_csv(path,index=False)
    
    columns=["Dice Coefficient","Rand Index","Rand Error","Pixel Accuracy","Pixel Error","Training Time (s)"]
    
    data={}
    data["Device"]=devname
    for c in columns:
        values=df[c].values
        mean=np.mean(values)
        sigma=np.std(values)
        print(f"{c}: Mean={mean:.4f}, Std={sigma:.4f}")
        data[c]={"Mean":float(mean),"Std":float(sigma)}
    
    path=f"../experiments/gnn/results.yaml"
    
    with open(path, 'w') as file:
        yaml.dump(data, file)
    


def run_experiment(device,devname):
    patients=load_patient_names()
    reports=[]
    for trial in range(1,ntrials+1):
        print(f"========Trial {trial}/{ntrials}==========")
        trainloader,testloader=create_train_test_sets(patients)
        model=SegmentGNN(in_channels, hidden_channels, nclasses).to(device)
        model,train_time=train_model(device,model,trainloader,epochs,lr)
        predictions=predict(device, model, testloader)
        true_masks,pred_masks=create_predicted_masks(predictions, testloader)
        rep=evaluate_model(trial,true_masks, pred_masks)
        rep.append(train_time)
        reports.append(rep)
        print("==========================================")
    save_reports(devname,reports)   
    


def main():
    device,devname=identify_device()
    print(f"\n======================================================================")
    print(f"Unsing {device} - {devname}")
    print(f"Segmenting MRI brain images using graphs with felzenszwalb alg. and GNN")
    run_experiment(device,devname)
    print(f"======================================================================\n")
    print("Experiment completed successfully.")
    exit(0)

if __name__ == "__main__":
    main()
