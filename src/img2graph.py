from PIL import Image
import numpy as np
from collections import Counter
import networkx as nx
from torch_geometric.utils import from_networkx
import torch
import matplotlib.pyplot as plt
from io import BytesIO
import skimage.segmentation as segmentation
import skimage.graph as graph
from skimage import measure

dpi=1000

plt.rcParams["text.usetex"]=True

plt.rcParams.update({'font.size': 20})

def load_mri(image_path, mask_path):
    """
    Load the MRI image and its corresponding mask from the specified paths.

    Parameters:
    image_path (str): The path to the MRI image file.
    mask_path (str): The path to the mask file.

    Returns:
    Tuple[np.ndarray, np.ndarray]: A tuple containing the MRI image and its corresponding mask as NumPy arrays.

    This function opens the MRI image and mask files using the PIL library, converts them to NumPy arrays, and then processes the mask to create a binary mask where 1 represents the region of interest and 0 represents the background.
    """
    image = np.array(Image.open(image_path))
    mask = np.array(Image.open(mask_path))
    num_mask = np.where(mask == 255, 1, 0)
    return image, num_mask

def create_segments(image,scale,sigma,min_size):
    segments = segmentation.felzenszwalb(image,scale=scale,sigma=sigma,min_size=min_size)
    segments = segments + 1  # Adjust the offset as needed
    boundaries = segmentation.mark_boundaries(image, segments, color=(1, 1, 0))  # Yellow boundaries
    return segments, boundaries


def assign_labels_to_superpixels(segments, mask):
    """
    Assigns labels to the superpixels based on the most common label in the corresponding mask region.

    Parameters:
    segments (np.ndarray): The superpixel segments as a NumPy array.
    mask (np.ndarray): The corresponding mask as a NumPy array.

    Returns:
    np.ndarray: A NumPy array containing the assigned labels for each superpixel.

    This function iterates through each superpixel segment and its corresponding mask region. It then assigns the most common label in the mask region to the superpixel segment. The resulting array contains the assigned labels for each superpixel.
    """
    labels = np.zeros(segments.max())
    for segment_id in range(segments.max()):
        segment_mask = (segments == segment_id)
        segment_labels = mask[segment_mask]
        if len(segment_labels) == 0:
            continue
        most_common_label = Counter(segment_labels).most_common(1)[0][0]
        labels[segment_id] = most_common_label
    return labels

def craete_rag(image, segments):
    """
    Creates a Region Adiajency Graph (RAG) from the input image and its corresponding superpixel segments.

    Parameters:
    image (np.ndarray): The input image as a NumPy array.
    segments (np.ndarray): The superpixel segments as a NumPy array.

    Returns:
    Tuple[torch.Tensor, List[Tuple[int, int]], nx.Graph]: A tuple containing the RAG data, a list of centers for the superpixels, and the corresponding graph as a NetworkX object.

    This function first creates a RAG using the `graph.rag_mean_color` function from the `skimage.segmentation` module. It then constructs a graph using the NetworkX library, where each node represents a superpixel and each edge represents a connection between two superpixels. The `features` attribute of each node in the graph is set to the mean color of the corresponding superpixel. Finally, the function returns the RAG data, a list of centers for the superpixels, and the corresponding graph as a NetworkX object.
    """
    rag = graph.rag_mean_color(image, segments, mode='similarity')
    G = nx.Graph()
    

    for node in rag.nodes:
        mean_color = rag.nodes[node]['mean color']
        G.add_node(node, features=list(mean_color))

    for edge in rag.edges:
        node1, node2 = edge
        G.add_edge(node1, node2)


    data=from_networkx(G)
    if data.edge_index.max() >= data.num_nodes:
        raise ValueError(f"Edge index out of bounds: max index {data.edge_index.max()} exceeds number of nodes {data.num_nodes}")

    centers = []
    properties = measure.regionprops(segments)
    for p in properties:
        centers.append(p.centroid)
    return data,centers,G


def draw_graph_on_image(image, graph, centers, figsize=(10, 10), dpi=100):
    """
    Draw nodes and edges of the graph on the image using NetworkX and remove white borders.

    Parameters:
    image (np.ndarray): The input image as a NumPy array.
    graph (nx.Graph): The graph to be drawn on the image.
    centers (List[Tuple[int, int]]): A list of centers for the superpixels, where each center is a tuple of two integers representing the coordinates of the center.
    figsize (Tuple[int, int], optional): The size of the figure in inches. Defaults to (10, 10).
    dpi (int, optional): The dots per inch for the figure. Defaults to 100.

    Returns:
    np.ndarray: A NumPy array containing the image with the graph drawn on it, after removing the white borders.

    This function first computes the positions for the nodes based on the superpixel centroids. It then draws the nodes and edges of the graph on the image using NetworkX, with the specified options. The function then removes the white borders from the image and returns the resulting image as a NumPy array.
    """
    # Compute positions for nodes based on superpixel centroids
    positions = {i + 1: [c[1], c[0]] for i, c in enumerate(centers)}  # Adjust i/home/salvatorecalderaro/Dropbox/AIxKDE/LGG SEGMENTATION/ndexing and coordinates

    options = {
        'node_color': 'red',
        'node_size': 20,
        'width': 2,
        'with_labels': False,
        'arrows': False,
        'edge_color': 'blue'
    }

    buf = BytesIO()
    plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(image)
    nx.draw_networkx(graph, positions, **options)
    plt.axis('off')
    plt.tight_layout()

    # Save the figure to the buffer
    plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Go to the beginning of the buffer
    buf.seek(0)

    # Open the image from the buffer
    img = Image.open(buf).convert("RGB")

    # Convert the image to a NumPy array
    img_array = np.array(img)

    # Remove white border
    white = np.array([255, 255, 255])
    mask = np.all(img_array != white, axis=-1)
    coords = np.argwhere(mask)
    y1, x1 = coords.min(axis=0)
    y2, x2 = coords.max(axis=0) + 1

    # Crop the image to the bounding box
    img_cropped = img_array[y1:y2, x1:x2]

    # Resize the image to 512x512 pixels
    img_resized = Image.fromarray(img_cropped)
    img_resized = img_resized.resize((512, 512), Image.LANCZOS)
    #img_resized.save("input_graph.png")
    # Convert the resized image back to a NumPy array
    img_array_resized = np.array(img_resized)

    return img_array_resized


def make_plot(original_image, segmentation_image, superpixel_image, graphplot):
    """
    Display the images in a 2x2 grid and save the result.

    Parameters:
    original_image (np.ndarray): The original input image as a NumPy array.
    segmentation_image (np.ndarray): The color mask (segmentation image) as a NumPy array.
    superpixel_image (np.ndarray): The superpixel boundaries image as a NumPy array.
    graphplot (np.ndarray): The graph visualization image on the superpixel image as a NumPy array.
    plot (bool, optional): A boolean flag indicating whether to display the plot. Defaults to False.

    Returns:
    None

    This function displays the original image, color mask (segmentation image), superpixel boundaries image, and graph visualization image in a 2x2 grid. It then saves the resulting plot as a PDF file named "report.pdf" with a high DPI value. If the `plot` parameter is set to `True`, the function will also display the plot.
    """
    plt.figure(figsize=(12, 12))  # Adjust the figure size as needed

    # Original Image
    plt.subplot(2, 2, 1)
    plt.title('Original Image')
    plt.imshow(original_image)
    plt.axis('off')

    # Color Mask (Segmentation Image)
    plt.subplot(2, 2, 2)
    plt.title('Mask')
    plt.imshow(segmentation_image,cmap="gray")
    plt.axis('off')

    # Superpixel Boundaries
    plt.subplot(2, 2, 3)
    plt.title('Felzenszwalb Algorithm Superpixels')
    plt.imshow(superpixel_image)
    plt.axis('off')

    # Graph Visualization on Superpixel Image
    plt.subplot(2, 2, 4)
    plt.title('Graph')
    plt.imshow(graphplot)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("../images/graph.png", dpi=dpi)

    
def create_graph(img_path,mask_path,scale,sigma,min_size, plot=False):
    """
    Creates a Region Adjacency Graph (RAG) from an input MRI image and its corresponding mask, using superpixels and a specified SLIC algorithm parameters.

    Parameters:
    img_path (str): The path to the MRI image file.
    mask_path (str): The path to the mask file.
    ns (int): The desired number of superpixels to create.
    c (int): The compactness parameter for the SLIC algorithm, which controls the size and shape of the superpixels.
    sigma (float): The Gaussian smoothing parameter for the SLIC algorithm, which affects the spatial coherence of the superpixels.
    plot (bool, optional): A flag indicating whether to display the plot of the original image, color mask (segmentation image), superpixel boundaries image, and graph visualization image in a 2x2 grid. Defaults to False.

    Returns:
    RAGGraph: The Region Adjacency Graph (RAG) object containing the superpixel segments, labels, mask, and graph data.

    This function loads the MRI image and its corresponding mask, creates superpixels using the SLIC algorithm, assigns labels to the superpixels based on the most common label in the corresponding mask region, creates a RAG from the input image and its corresponding superpixel segments, sets the labels, mask, and segments attributes of the RAGGraph object, and optionally displays the plot.
    """
    # Load the input MRI image and its corresponding mask
    image, mask = load_mri(img_path, mask_path)

    # Create superpixels from the input image using the SLIC algorithm
    segments, boundaries = create_segments(image,scale,sigma,min_size)

    # Assign labels to the superpixels based on the most common label in the corresponding mask region
    labels = assign_labels_to_superpixels(segments, mask)

    # Create a Region Adjacency Graph (RAG) from the input image and its corresponding superpixel segments
    rag_graph, centers, nx_graph = craete_rag(image, segments)

    # Set the labels, mask, and segments attributes of the RAGGraph object
    rag_graph.y = torch.tensor(labels, dtype=torch.long)
    rag_graph.num_mask = torch.tensor(mask)
    rag_graph.segments = torch.tensor(segments)
    rag_graph.path=img_path

    # If the plot parameter is set to True, display the plot of the original image, color mask (segmentation image), superpixel boundaries image, and graph visualization image in a 2x2 grid
    if plot:
        graphplot = draw_graph_on_image(image, nx_graph, centers)
        make_plot(image, mask, boundaries, graphplot)
    return rag_graph
