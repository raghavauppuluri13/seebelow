import seebelow.utils.constants as seebelow_const
import numpy as np
from seebelow.algorithms.grid import SurfaceGridMap
import open3d as o3d
from seebelow.utils.pcd_utils import scan2mesh, mesh2polyroi, visualize_pcds, inverse_crop, pick_polygon_bbox

ROI = np.array([[0.5325745344161987, 0.005577473901212215, 0.0685637779533863],
                [0.5281904935836792, 0.00452816765755415, 0.06843702122569084],
                [0.5247376561164856, 0.0020533567294478416, 0.0689111165702343],
                [0.5222628116607666, -0.0008339229971170425, 0.06852295622229576],
                [0.5202004909515381, -0.004693150520324707, 0.0678398534655571],
                [0.5193755626678467, -0.011558104306459427, 0.06682687997817993],
                [0.5202004909515381, -0.014857852831482887, 0.06484301388263702],
                [0.5222628116607666, -0.01803570706397295, 0.06506950035691261],
                [0.5251501202583313, -0.02053224854171276, 0.06536503881216049],
                [0.5300997495651245, -0.021869818679988384, 0.06668160483241081],
                [0.5375241637229919, -0.020632412284612656, 0.06790422648191452],
                [0.5421661734580994, -0.018570070154964924, 0.06432020664215088],
                [0.5457735061645508, -0.014857853297144175, 0.06309395655989647],
                [0.5483525991439819, -0.009083293378353119, 0.06289023160934448],
                [0.5471891164779663, -0.004133671522140503, 0.06520787253975868],
                [0.544123649597168, 0.004115698859095573, 0.06293772161006927],
                [0.5457735061645508, 0.00041433796286582947, 0.0641762875020504],
                [0.5388736128807068, 0.005765574052929878, 0.06536503881216049]])

BO = 'dataset_02-23-2024_13-50-42'
RANDOM = 'dataset_02-22-2024_14-07-30'

if __name__ == "__main__":
    pcd = o3d.io.read_point_cloud(
        str(seebelow_const.SEEBELOW_MESH_PATH.parent / 'eval_old' / f'eval_{RANDOM}' /
            'mesh_without_CF.ply'))
    #surface_mesh = scan2mesh(pcd)

    bbox: o3d.visualization.SelectionPolygonVolume = pick_polygon_bbox(surface_mesh,
                                                                       polybox_pts=ROI)
    inverse_cropped_pcd = inverse_crop(bbox, pcd)
    cropped_pcd = mesh2polyroi(surface_mesh, ROI)
    v = inverse_cropped_mesh.get_center() - cropped_pcd.get_center()
    T = np.eye(4)
    v[2] = 0
    T[:3, 3] = v
    cropped_pcd.transform(T)
    cropped_pcd += inverse_cropped_mesh
    o3d.visualization.draw_geometries([cropped_mesh])
