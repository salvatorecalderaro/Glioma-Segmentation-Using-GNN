import random
import os 
import numpy as np
import torch
import platform
import cpuinfo
import pickle
from torch_geometric.loader import DataLoader
from model import SegmentGNN,train_model,predict
from PIL import Image
import matplotlib.pyplot as plt
dpi=1000

plt.rcParams["text.usetex"]=True

plt.rcParams.update({'font.size': 20})

seed=0
train_size=0.8
mini_batch_size=32
in_channels=3
hidden_channels=512
nclasses=1
ntrials=100
epochs=20
lr=0.001
dpi=300

#plt.rcParams["text.usetex"]=True

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
    Create a mask by assigning the predicted label to each segment.

    Parameters:
    - segments (numpy.ndarray): Segmentation labels (same shape as the image).
    - predicted_mask (numpy.ndarray): Predicted labels for each segment.

    Returns:
    - num_mask (numpy.ndarray): A binary mask with the same shape as segments.
    """
    num_mask = np.zeros_like(segments, dtype=np.int32)
    unique_segments = np.unique(segments)

    for segment_id in unique_segments:
        mask = (segments == segment_id)
        label = predicted_mask[segment_id - 1] if segment_id - 1 < len(predicted_mask) else 0
        num_mask[mask] = label

    return num_mask


def save_image(image,true_mask, pred_mask,index):
    fig, axes = plt.subplots(1, 3, figsize=(15, 7))

    # Plot original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Plot true mask
    axes[1].imshow(true_mask, cmap='gray')
    axes[1].set_title('True Mask')
    axes[1].axis('off')

    # Plot predicted mask
    axes[2].imshow(pred_mask, cmap='gray')
    axes[2].set_title('Predicted Mask')
    axes[2].axis('off')
    plt.tight_layout()
    
    path=f"../experiments/gnn/results/image_{index}.png"
    plt.savefig(path,dpi=dpi)
    plt.close()
    
def plot_masks(predictions, testloader):
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
    index=1
    for graph, pred in zip(testloader, predictions):
        num_mask = graph.num_mask.detach().cpu().numpy()
        segments = graph.segments.detach().cpu().numpy()
        pred_mask = create_mask(segments, np.array(pred))
        if 1 in np.unique(pred_mask):
            path=graph.path[0]
            img = Image.open(path)
            img=np.array(img)
            save_image(img, num_mask, pred_mask, index)
            index += 1


def main():
    device,devname=identify_device()
    print(f"Unsing {device} - {devname}")
    print(f"Segmenting MRI brain images using graphs with SQuickshift and GNN")
    patients=load_patient_names()
    trainloader,testloader=create_train_test_sets(patients)
    model=SegmentGNN(in_channels, hidden_channels, nclasses).to(device)
    model,train_time=train_model(device,model,trainloader,epochs,lr)
    predictions=predict(device, model, testloader)
    plot_masks(predictions,testloader)
    exit(0)

if __name__ == "__main__":
    main()
