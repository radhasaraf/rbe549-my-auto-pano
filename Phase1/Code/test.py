import numpy as np


imgs_graph = np.array(
        [[ 0.,21.,10., 9.],
         [20., 0.,23., 6.],
         [12.,24., 0.,14.],
         [17.,17.,12., 0.]])

homography_inds_mat = np.array(
        [[-1,  0,  1,  2],
         [ 3, -1,  4,  5],
         [ 6,  7, -1,  8],
         [ 9, 10, 11, -1]])
homography_mats_list = [23,45,2,352,12,51,78,12,123,41,76,32]
## expected output:41*123,12,0,123

print(f"before:\n{imgs_graph}")

imgs_graph[imgs_graph < 10 ] = 0
print(f"after thresholding:\n{imgs_graph}")

imgs_graph[imgs_graph < imgs_graph.T] = 0
print(f"after fixing directions:\n{imgs_graph}")

imgs_graph[np.max(imgs_graph,axis=0,keepdims=True)!=imgs_graph]=0
print(f"after taking max per column:\n{imgs_graph}")

ref_image_id = np.where(~imgs_graph.any(axis=0))[0]
print(f"ref_image_id:{ref_image_id}")

eff_H_list = []
eff_H_list_ids = np.full(imgs_graph.shape[0],fill_value=-1)
nonzero_list_id = np.nonzero(imgs_graph)
print(nonzero_list_id)
for k,j in enumerate(nonzero_list_id[1]):
    eff_H = homography_mats_list[homography_inds_mat[nonzero_list_id[0][k],j]]
    eff_H_list.append(eff_H)
    eff_H_list_ids[j] = len(eff_H_list) - 1
print(eff_H_list)
print(eff_H_list_ids)

def get_effective_homography(I,graph,reference_id,eff_homography_list,eff_homography_ids):
    visited = False
    print(f"I:{I}")
    if I == reference_id:
        return None
    for j in range(graph.shape[0]):
        if visited:
            print("something is wrong")
            return
        if j == reference_id:
            continue
        if graph[j,I] != 0:
            visited = True
            I_d = eff_homography_ids[I]
            val = get_effective_homography(j,graph,reference_id,eff_homography_list,eff_homography_ids)
            if val is None:
                continue
            eff_homography_list[I_d] -= val
    return eff_homography_list[eff_homography_ids[I]]

eff_H_vals = np.full(imgs_graph.shape[0],fill_value=-1)
for i in range(imgs_graph.shape[0]):
    print(f"i:{i}")
    val = get_effective_homography(i,imgs_graph,ref_image_id,eff_H_list,eff_H_list_ids)
    if val is None:
        continue
    eff_H_vals[i] = val
print(eff_H_vals)
