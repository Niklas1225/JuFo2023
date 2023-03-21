import pathlib
import numpy as np
import nibabel as nib
from skimage import measure
import xmltodict
import plotly.graph_objects as go

DATA_PATH = pathlib.Path(__file__).parent.joinpath("data").resolve()

default_colorscale = [
    [0, "rgb(12,51,131)"],
    [0.25, "rgb(10,136,186)"],
    [0.5, "rgb(242,211,56)"],
    [0.75, "rgb(242,143,56)"],
    [1, "rgb(217,30,30)"],
]


def read_mniobj(file):
    """
    Parses an obj file.
    
    :params file: file name in data folder
    :returns: a tuple
    """

    def triangulate_polygons(list_vertex_indices):
        for k in range(0, len(list_vertex_indices), 3):
            yield list_vertex_indices[k : k + 3]

    with open(DATA_PATH.joinpath(file)) as fp:
        num_vertices = 0
        matrix_vertices = []
        k = 0
        list_indices = []

        for i, line in enumerate(fp):
            if i == 0:
                num_vertices = int(line.split()[6])
                matrix_vertices = np.zeros([num_vertices, 3])
            elif i <= num_vertices:
                matrix_vertices[i - 1] = list(map(float, line.split()))
            elif i > 2 * num_vertices + 5:
                if not line.strip():
                    k = 1
                elif k == 1:
                    list_indices.extend(line.split())

    list_indices = [int(i) for i in list_indices]
    faces = np.array(list(triangulate_polygons(list_indices)))
    return matrix_vertices, faces


def plotly_triangular_mesh(
    vertices,
    faces,
    intensities=None,
    colorscale="Viridis",
    flatshading=False,
    showscale=False,
    plot_edges=False,
    names=None,
    opacity=1,
):

    x, y, z = vertices.T
    I, J, K = faces.T

    if intensities is None:
        intensities = z

    mesh = {
        "type": "mesh3d",
        "x": x,
        "y": y,
        "z": z,
        "colorscale": colorscale,
        "intensity": intensities,
        "flatshading": flatshading,
        "i": I,
        "j": J,
        "k": K,
        "name": "",
        "hovertemplate": "x=%{x}<br>y=%{y}<br>z=%{z}<extra></extra>",
        "showscale": showscale,
        "opacity": opacity,
        "lighting": {
            "ambient": 0.18,
            "diffuse": 1,
            "fresnel": 0.1,
            "specular": 1,
            "roughness": 0.1,
            "facenormalsepsilon": 1e-6,
            "vertexnormalsepsilon": 1e-12,
        },
        "lightposition": {"x": 100, "y": 200, "z": 0},
    }
    if names is not None:
        mesh["text"] = names
        mesh["hovertemplate"] = "x=%{x}<br>y=%{y}<br>z=%{z}<br>color=%{intensity}<br>region=%{text}<extra></extra>"

    if showscale:
        mesh["colorbar"] = {"thickness": 20, "ticklen": 4, "len": 0.75}

    if plot_edges is False:
        return [mesh]

    lines = create_plot_edges_lines(vertices, faces)
    return [mesh, lines]

def create_plot_edges_lines(vertices, faces):
    tri_vertices = vertices[faces]
    Xe = []
    Ye = []
    Ze = []
    for T in tri_vertices:
        Xe += [T[k % 3][0] for k in range(4)] + [None]
        Ye += [T[k % 3][1] for k in range(4)] + [None]
        Ze += [T[k % 3][2] for k in range(4)] + [None]

    # define the lines to be plotted
    lines = {
        "type": "scatter3d",
        "x": Xe,
        "y": Ye,
        "z": Ze,
        "mode": "lines",
        "name": "",
        "line": {"color": "rgb(70,70,70)", "width": 1},
    }
    return lines

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def create_mesh_data(option, z, img=None, img_labeled=None, cs=None, opacity=0.6):
    data = []
    names = None
    mesh = []

    if option == "human_mrt":
        nifti_file = nib.load(DATA_PATH.joinpath("1103_3-test.nii"))
        data = nifti_file.get_fdata()
        max_z = data.shape[2]
        data = np.rot90(data, k=-1, axes=(1,2))

        #create a sliced matrix
        rest = data.shape[2]-z
        if rest > 0:
            data = data[:, :, :z]
            data = np.concatenate((data, np.zeros([data.shape[0], data.shape[1], rest])), axis=2)
        
        vertices, faces, normals, intensities = measure.marching_cubes(data, 0,  method='lorensen', allow_degenerate=False)

        data = plotly_triangular_mesh(
            vertices, faces, intensities, colorscale=default_colorscale, names=names
        )
        mesh.extend(data)

        mesh[0]["name"] = option

    elif option == "mrt_regions":
        #load file
        nifti_file = nib.load(DATA_PATH.joinpath("1103_3_glm-test.nii"))
        #get data
        data = nifti_file.get_fdata()
        max_z = data.shape[2]
        #rotate
        data = np.rot90(data, k=-1, axes=(1,2))

        #create a sliced matrix
        rest = data.shape[2]-z
        if rest > 0:
            if img is not None:
                data = data[:, :, :z-1]
                data = np.concatenate((data, np.zeros([data.shape[0], data.shape[1], rest+1])), axis=2)
            else:
                data = data[:, :, :z]
                data = np.concatenate((data, np.zeros([data.shape[0], data.shape[1], rest])), axis=2)

        #transform 
        vertices, faces, normals, intensities = measure.marching_cubes(data, 0,  method='lorensen', allow_degenerate=False)

        #create labels
        with open(DATA_PATH.joinpath('1103_3_glm_LabelMap.xml'), 'r', encoding='utf-8') as file:
            my_xml = file.read()
        my_dict = xmltodict.parse(my_xml)
        new_dict = {"0": ""}
        for i in range(len(my_dict["LabelList"]["Label"])):
            number = my_dict["LabelList"]["Label"][i]["Number"]
            name = my_dict["LabelList"]["Label"][i]["Name"]
            new_dict[str(number)] = name
        names = []
        for i in intensities:
            names.append(new_dict[str(int(i))])
        names = np.array(names)

        mesh_obj = plotly_triangular_mesh(
            vertices, faces, intensities, colorscale=default_colorscale, names=names
        )
        mesh.extend(mesh_obj)

        mesh[0]["name"] = option

        #add image if needed
        if img is not None:
            matrix1 = np.ndarray(data.shape)
            img_gray = rgb2gray(img)
            matrix1[:, :, z] = img_gray.T
            matrix2 = np.ndarray(data.shape)
            matrix2[:, :, z] = img_labeled.T
            vertices, faces, normals, intensities = measure.marching_cubes(matrix1, 0,  method='lorensen', allow_degenerate=False)

            matrix3 = np.ndarray(data.shape)
            matrix3[:, :, z] = img_labeled.T
            matrix3[matrix3 == 0] = 1
            matrix4 = matrix1.copy()
            matrix4[matrix4 > 0] = 1
            matrix3 = matrix3 * matrix4
            _, _, _, intensities_for_names = measure.marching_cubes(matrix3, 0,  method='lorensen', allow_degenerate=False)

            img_names = []
            for i in intensities_for_names:
               img_names.append(new_dict[str(int(i))])
            img_names = np.array(img_names)

            mesh_obj = plotly_triangular_mesh(
                vertices, faces, intensities, colorscale="inferno", names=img_names, opacity=opacity
            )
            mesh.extend(mesh_obj)
            mesh[1]["name"] = "img"

    elif option == "labeled_atlas":
        #load file
        nifti_file = nib.load(DATA_PATH.joinpath("aal.nii.gz"))
        #get data
        data = nifti_file.get_fdata()
        max_z = data.shape[2]
        #rotate
        #data = np.rot90(data, k=-1, axes=(1,2))

        #create a sliced matrix
        rest = data.shape[2]-z
        if rest > 0:
            data = data[:, :, :z]
            data = np.concatenate((data, np.zeros([data.shape[0], data.shape[1], rest])), axis=2)

        #transform 
        vertices, faces, normals, intensities = measure.marching_cubes(data, 0,  method='lorensen', allow_degenerate=False)
        
        #create labels
        with open(DATA_PATH.joinpath('aal.nii.txt')) as f:
            lines = f.readlines()

        value_list = [""]
        for count, line in enumerate(lines):
            value_list.append(line.split()[1])
        
        names = []
        for i in intensities:
            names.append(value_list[int(i)])
        names = np.array(names)

        mesh_obj = plotly_triangular_mesh(
            vertices, faces, intensities, colorscale=default_colorscale, names=names
        )
        mesh.extend(mesh_obj)

        mesh[0]["name"] = option

        #add image if needed
        if img is not None: 
            matrix = np.ndarray(data.shape)
            img_gray = rgb2gray(img)
            matrix[:, :, z] = img_gray.T
            vertices, faces, normals, intensities = measure.marching_cubes(matrix, 0,  method='lorensen', allow_degenerate=False)
            mesh_obj = plotly_triangular_mesh(
                vertices, faces, intensities, colorscale=default_colorscale, names=names
            )
            mesh_obj[0]["color"] = "purple"
            mesh.extend(mesh_obj)

    elif option == "all_regions":
        #load file
        nifti_file = nib.load(DATA_PATH.joinpath('aal.nii.gz'))
        #get data
        data = nifti_file.get_fdata()
        max_z = data.shape[2]
        #rotate
        #data = np.rot90(data, k=-1, axes=(1,2))

        #create a sliced matrix
        rest = data.shape[2]-z
        if rest > 0:
            data = data[:, :, :z]
            data = np.concatenate((data, np.zeros([data.shape[0], data.shape[1], rest])), axis=2)

        for i in np.unique(data):
            this_region = data.copy()
            this_region[this_region != i] = 0
            
            if np.sum(this_region) != 0:
                vertices, faces, normals, intensities = measure.marching_cubes(this_region, 0,  method='lorensen', allow_degenerate=False)
                #create labels
                with open(DATA_PATH.joinpath('aal.nii.txt')) as f:
                    lines = f.readlines()

                value_list = [""]
                for count, line in enumerate(lines):
                    value_list.append(line.split()[1])
                
                names = []
                for i in intensities:
                    names.append(value_list[int(i)])
                names = np.array(names)

                one_mesh = plotly_triangular_mesh(
                    vertices, faces, intensities, colorscale=default_colorscale, names=names, opacity=0.5
                )
                mesh.extend(one_mesh)

    else:
        raise ValueError

    
    
    return mesh
