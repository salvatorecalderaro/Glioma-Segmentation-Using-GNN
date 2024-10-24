import os
from glob import glob
from img2graph import create_graph
import pickle


 

scale=1
sigma=0.8
min_size=20

def process_patient(id, scale,sigma,min_size):
    """
    Process a single patient's images and create corresponding graphs.

    Args:
    - id (str): The identifier of the patient.
    - ns (int): Number of superpixels to create (default: 500).
    - c (float): Compactness parameter (default: 30.0).
    - sigma (float): Gaussian smoothing parameter (default: 1.0).

    Returns:
    - None: This function does not return a value, but instead saves the created graphs to a file.
    """
    path = f"../dataset/{id}"
    mask_files = glob(path + '*/*_mask*')
    image_files = [file.replace('_mask', '') for file in mask_files]
    graphs = []
    for img_file, mask_file in zip(image_files, mask_files):
        g = create_graph(img_file, mask_file, scale,sigma,min_size)
        graphs.append(g)

    path = f"../graphs/{id}"

    with open(path, "wb") as file:
        pickle.dump(graphs, file)
    
def process_patients(scale,sigma,min_size):
    data_dir="../dataset"
    patients=sorted(os.listdir(data_dir))
    for p in patients:
        print(f"Creating graphs for patient {p}...")
        process_patient(p, scale,sigma,min_size)


def main():
    print("\n=================================================================")
    print("Creating graphs using felzenszwalb alg. with the following parameters:")
    print(f"Scale: {scale}")
    print(f"Sigma: {sigma}")
    print(f"Min size: {min_size}")
    process_patients(scale,sigma,min_size)
    print("\nFinished creating graphs for all patients.")
    print("===================================================================\n")
    exit(0)


if __name__ == "__main__":
    main()
